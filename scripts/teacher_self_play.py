import chess
import chess.engine
import torch

class StockfishTeacherGenerator:
    def __init__(self, encoder, stockfish_path, max_moves=200):
        self.encoder = encoder
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.max_moves = max_moves

    def generate_game(self, depth=8):
        board = chess.Board()

        states = []
        policies = []
        values = []

        while not board.is_game_over() and len(states) < self.max_moves:
            # Stockfish move
            result = self.engine.analyse(
                board,
                chess.engine.Limit(depth=depth)
            )

            move = result["pv"][0]
            score = result["score"].white().score(mate_score=10000)

            # normalize value
            value = max(min(score / 1000, 1), -1)

            # one-hot policy
            policy = torch.zeros(4096)
            idx = self.encoder.encode_move(move, board)
            policy[idx] = 1.0

            states.append(self.encoder.encode_board(board))
            policies.append(policy)
            values.append(value)

            board.push(move)

        return (
            torch.stack(states),
            torch.stack(policies),
            torch.tensor(values).unsqueeze(1)
        )

    def close(self):
        self.engine.quit()
