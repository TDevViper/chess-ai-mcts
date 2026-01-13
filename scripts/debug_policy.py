import chess
import numpy as np
from src.engine.chess_uci_engine import SimpleEngine

FILES = "abcdefgh"

def print_policy_heatmap(board, policy):
    heat = np.zeros((8, 8))

    for move in board.legal_moves:
        from_sq = move.from_square
        to_sq = move.to_square
        prob = policy[from_sq * 64 + to_sq].item()

        r = 7 - (from_sq // 8)
        c = from_sq % 8
        heat[r, c] += prob

    print("\nPolicy heatmap (from-square strength):\n")
    print("    a    b    c    d    e    f    g    h")
    for r in range(8):
        row = f"{8-r} "
        for c in range(8):
            row += f"{heat[r,c]:.2f} "
        print(row)
    print()

def main():
    engine = SimpleEngine("checkpoints/fresh_model.pt", device="cpu")
    board = chess.Board()

    move, policy, value = engine.predict_with_policy(board)

    print("Chosen move:", move)
    print("Value:", round(value, 3))

    print_policy_heatmap(board, policy)

if __name__ == "__main__":
    main()
