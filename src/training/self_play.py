import chess
import torch


class SelfPlayGenerator:
    def __init__(self, engine, encoder, max_moves=200):
        self.engine = engine
        self.encoder = encoder
        self.max_moves = max_moves

    # ==================================================
    # GENERATE MULTIPLE GAMES
    # ==================================================
    def generate_games(self, num_games=5, sims=50):
        all_states, all_policies, all_values = [], [], []
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

        for g in range(num_games):
            states, policies, values, result = self.play_one_game(sims)

            all_states.extend(states)
            all_policies.extend(policies)
            all_values.extend(values)

            results[result] += 1
            print(f"🔥 Game {g + 1}/{num_games} finished: {result}")

        return {
            "board_states": torch.stack(all_states),
            "policy_targets": torch.stack(all_policies),
            "value_targets": torch.tensor(all_values).unsqueeze(1),
            "results": results,
            "num_positions": len(all_states),
        }

    # ==================================================
    # PLAY ONE GAME (AGGRESSIVE + MATERIAL AWARE)
    # ==================================================
    def play_one_game(self, sims):
        board = chess.Board()

        board_states = []
        policy_targets = []
        player_turns = []      # True = WHITE to move
        board_snapshots = []   # Board copies for material eval

        move_count = 0

        # --------------------------------------------------
        # NO RESIGN | NO EARLY DRAW | MAX_MOVES ONLY
        # --------------------------------------------------
        while not board.is_game_over() and move_count < self.max_moves:

            # 🔥 AGGRESSIVE TEMPERATURE SCHEDULE
            if move_count < 15:
                temperature = 1.5
            elif move_count < 40:
                temperature = 1.0
            else:
                temperature = 0.3

            move, mcts_policy, _ = self.engine.get_move_mcts_with_policy(
                board,
                sims=sims,
                temperature=temperature,
            )

            board_states.append(self.encoder.encode_board(board))
            policy_targets.append(mcts_policy)

            # ✅ FIX: explicit boolean (no ambiguity)
            player_turns.append(board.turn == chess.WHITE)

            board_snapshots.append(board.copy())

            board.push(move)
            move_count += 1

        # ==================================================
        # FINAL RESULT (NORMALIZED)
        # ==================================================
        result = board.result()
        if result == "*" or result is None:
            result = "1/2-1/2"

        # ==================================================
        # GAME VALUE (WIN / LOSS / DRAW)
        # ==================================================
        if result == "1-0":
            game_value = 1.0
        elif result == "0-1":
            game_value = -1.0
        else:
            game_value = 0.0

        # ==================================================
        # VALUE TARGETS (GAME + MATERIAL BLEND)
        # ==================================================
        value_targets = []

        for b, is_white in zip(board_snapshots, player_turns):

            # ---------- MATERIAL BALANCE ----------
            material = (
                1 * (len(b.pieces(chess.PAWN, chess.WHITE)) -
                     len(b.pieces(chess.PAWN, chess.BLACK)))
              + 3 * (len(b.pieces(chess.KNIGHT, chess.WHITE)) -
                     len(b.pieces(chess.KNIGHT, chess.BLACK)))
              + 3 * (len(b.pieces(chess.BISHOP, chess.WHITE)) -
                     len(b.pieces(chess.BISHOP, chess.BLACK)))
              + 5 * (len(b.pieces(chess.ROOK, chess.WHITE)) -
                     len(b.pieces(chess.ROOK, chess.BLACK)))
              + 9 * (len(b.pieces(chess.QUEEN, chess.WHITE)) -
                     len(b.pieces(chess.QUEEN, chess.BLACK)))
            )

            # Clamp & normalize material signal
            material = max(-10, min(10, material)) / 10.0

            # 🔥 FINAL VALUE TARGET
            final_value = 0.7 * game_value + 0.3 * material

            value_targets.append(
                final_value if is_white else -final_value
            )

        return board_states, policy_targets, value_targets, result
