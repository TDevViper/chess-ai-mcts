"""
AlphaZero-style Monte Carlo Tree Search (MCTS)
"""

import math
import torch
import chess
from typing import Dict, Tuple


# =========================
# MCTS NODE
# =========================
class MCTSNode:
    def __init__(self, board: chess.Board, parent=None, prior: float = 0.0):
        self.board = board
        self.parent = parent
        self.prior = prior
        self.children: Dict[chess.Move, "MCTSNode"] = {}
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return len(self.children) > 0


# =========================
# MCTS
# =========================
class MCTS:
    def __init__(self, engine, simulations: int = 100, c_puct: float = 1.5):
        self.engine = engine
        self.simulations = simulations
        self.c_puct = c_puct

    # -------------------------
    def search(
        self, board: chess.Board, temperature: float = 1.0
    ) -> Tuple[chess.Move, torch.Tensor, float]:

        root = MCTSNode(board.copy())

        for _ in range(self.simulations):
            node = root
            path = [node]

            # SELECTION
            while node.is_expanded() and not node.board.is_game_over():
                node = self._select_child(node)
                path.append(node)

            # EXPANSION / EVALUATION
            if node.board.is_game_over():
                value = self._terminal_value(node.board)
            else:
                value = self._expand(node, add_noise=(node is root))

            # BACKPROP
            self._backpropagate(path, value)

        policy = self._policy_from_visits(root)
        move = self._pick_move(root, temperature)

        return move, policy, root.value

    # -------------------------
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        total_visits = sum(c.visit_count for c in node.children.values())

        def ucb(child: MCTSNode):
            q = child.value
            u = (
                self.c_puct
                * child.prior
                * math.sqrt(total_visits + 1)
                / (1 + child.visit_count)
            )
            return q + u

        return max(node.children.values(), key=ucb)

    # -------------------------
    def _expand(self, node: MCTSNode, add_noise=False) -> float:
        board = node.board

        board_tensor = self.engine.encoder.encode_board(board).to(self.engine.device)
        legal_mask = self.engine.encoder.get_legal_move_mask(board).to(self.engine.device)

        with torch.no_grad():
            policy_logits, value = self.engine.model(board_tensor.unsqueeze(0))
            logits = policy_logits.squeeze(0)
            logits[legal_mask == 0] = -1e9
            policy = torch.softmax(logits, dim=0)

        # Dirichlet noise (AlphaZero)
        if add_noise:
            alpha = 0.03
            epsilon = 0.40
            noise = torch.distributions.Dirichlet(
                torch.full_like(policy, alpha)
            ).sample()
            policy = (1 - epsilon) * policy + epsilon * noise

        for move in board.legal_moves:
            idx = self.engine.encoder.encode_move(move, board)
            prior = policy[idx].item()

            new_board = board.copy()
            new_board.push(move)

            node.children[move] = MCTSNode(new_board, node, prior)

        return float(value.item())

    # -------------------------
    def _backpropagate(self, path, value):
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    # -------------------------
    def _terminal_value(self, board: chess.Board) -> float:
        result = board.result()
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        else:
            return 0.0

    # -------------------------
    def _policy_from_visits(self, root: MCTSNode) -> torch.Tensor:
        policy = torch.zeros(4096)

        for move, child in root.children.items():
            idx = self.engine.encoder.encode_move(move, root.board)
            policy[idx] = child.visit_count

        if policy.sum() > 0:
            policy /= policy.sum()

        return policy

    # -------------------------
    def _pick_move(self, root: MCTSNode, temperature: float) -> chess.Move:
        moves = list(root.children.keys())
        visits = torch.tensor(
            [root.children[m].visit_count for m in moves],
            dtype=torch.float32
        )

        if temperature == 0:
            return moves[torch.argmax(visits).item()]

        probs = visits ** (1.0 / temperature)
        probs /= probs.sum()

        return moves[torch.multinomial(probs, 1).item()]


# =========================
# MCTS ENGINE
# =========================
class MCTSEngine:
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        simulations: int = 100,
        c_puct: float = 1.5,
    ):
        from src.engine.chess_uci_engine import SimpleEngine

        self.base_engine = SimpleEngine(model_path, device)
        self.encoder = self.base_engine.encoder
        self.model = self.base_engine.model
        self.device = self.base_engine.device

        self.mcts = MCTS(self.base_engine, simulations, c_puct)

    def get_move(self, board: chess.Board, temperature: float = 0.0) -> chess.Move:
        move, _, _ = self.mcts.search(board, temperature)
        return move

    def get_move_with_analysis(self, board: chess.Board, temperature: float = 0.0):
        move, policy, value = self.mcts.search(board, temperature)
        return {
            "move": move,
            "policy": policy,
            "value": value,
            "simulations": self.mcts.simulations,
        }
