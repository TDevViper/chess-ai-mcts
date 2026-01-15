import torch
import chess

from src.network.chess_net import ChessNet
from src.core.board_encoder import BoardEncoder
from src.engine.mcts import MCTS


class SimpleEngine:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device

        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )

        config = checkpoint.get("model_config", {
            "input_channels": 18,
            "num_filters": 64,
            "num_residual_blocks": 2,
        })

        self.model = ChessNet(
            input_channels=config["input_channels"],
            num_filters=config["num_filters"],
            num_residual_blocks=config["num_residual_blocks"],
        ).to(device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.encoder = BoardEncoder()
        print(f"✅ SimpleEngine loaded on {device.upper()}")


    # --------------------------------------------------
    # NN INFERENCE
    # --------------------------------------------------

    def predict_with_policy(self, board, temperature=1.0):
        board_tensor = self.encoder.encode_board(board).to(self.device)
        legal_mask = self.encoder.get_legal_move_mask(board).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.model(board_tensor.unsqueeze(0))
            logits = policy_logits.squeeze(0)

            logits[legal_mask == 0] = -1e9
            probs = torch.softmax(logits / temperature, dim=0)

        move_idx = torch.multinomial(probs, 1).item()
        move = self.encoder.decode_move(move_idx, board)

        return move, probs.cpu(), value.item()

    # --------------------------------------------------
    # MCTS API (🔥 IMPORTANT)
    # --------------------------------------------------

    def get_move_mcts_with_policy(self, board, sims=50):
        mcts = MCTS(self, simulations=sims)
        return mcts.search(board)


    # --------------------------------------------------
    # DEBUG
    # --------------------------------------------------

    def top_k_moves(self, board, k=5):
        _, probs, value = self.predict_with_policy(board)

        topk = torch.topk(probs, k)
        moves = []

        for p, idx in zip(topk.values, topk.indices):
            move = self.encoder.decode_move(idx.item(), board)
            moves.append((move, float(p)))

        return moves, value
    def get_move_mcts_with_policy(self, board, sims=50, temperature=1.0):
        from src.engine.mcts import MCTS
        mcts = MCTS(self, simulations=sims)
        return mcts.search(board, temperature)


