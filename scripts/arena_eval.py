import os
import argparse
import chess
import torch

from src.engine.mcts import MCTS
from src.engine.chess_uci_engine import SimpleEngine

# ==================================================
# CONFIG (EARLY TRAINING)
# ==================================================
GAMES = 20

SIMS = 50                # 🔥 weaker arena
DEVICE = "cpu"

EARLY_TEMP = 0.6          # 🔥 exploration
LATE_TEMP = 0.05

MAX_ARENA_MOVES = 120     # 🔥 force decisions

# ==================================================
# LOAD ENGINE
# ==================================================
def load_engine(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    base = SimpleEngine(model_path, device=DEVICE)
    return MCTS(base, simulations=SIMS)

# ==================================================
# PLAY ONE ARENA GAME
# ==================================================
def play_arena_game(engine_new, engine_old):
    board = chess.Board()

    # Randomize colors
    new_is_white = bool(torch.randint(0, 2, (1,)).item())
    move_count = 0

    # Optional opening diversity
    if torch.rand(1).item() < 0.3:
        try:
            board.push(chess.Move.from_uci("e2e4"))
        except:
            pass

    while not board.is_game_over() and move_count < MAX_ARENA_MOVES:

        # 🔥 Early exploration
        temperature = EARLY_TEMP if move_count < 15 else LATE_TEMP

        # Select engine
        if board.turn == chess.WHITE:
            engine = engine_new if new_is_white else engine_old
        else:
            engine = engine_old if new_is_white else engine_new

        # 🔥 Get move + value
        move, _, value = engine.search(board, temperature=temperature)

        # 🔥 Arena resign logic (CRITICAL)
        if value < -0.85:
            return "old" if new_is_white else "new"

        # Safety
        if move not in board.legal_moves:
            return "draw"

        board.push(move)
        move_count += 1

    # Forced draw
    if not board.is_game_over():
        return "draw"

    result = board.result()

    if result == "1-0":
        return "new" if new_is_white else "old"
    elif result == "0-1":
        return "old" if new_is_white else "new"
    else:
        return "draw"


# ==================================================
# ARENA EVALUATION
# ==================================================
def arena_eval(new_model_path, old_model_path):
    print("⚔️ Arena evaluation...")
    print(f"NEW: {new_model_path}")
    print(f"OLD: {old_model_path}")

    engine_new = load_engine(new_model_path)
    engine_old = load_engine(old_model_path)

    results = {"new": 0, "old": 0, "draw": 0}

    for i in range(GAMES):
        outcome = play_arena_game(engine_new, engine_old)
        results[outcome] += 1
        print(f"Game {i+1}/{GAMES}: {outcome}")

    print("Arena results:", results)
    return results

# ==================================================
# PROMOTION RULE
# ==================================================
def should_promote(results):
    return results["new"] > results["old"]

# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arena evaluation")
    parser.add_argument("--candidate", type=str, required=True)
    parser.add_argument("--baseline", type=str, required=True)
    args = parser.parse_args()

    results = arena_eval(
        new_model_path=args.candidate,
        old_model_path=args.baseline,
    )

    if should_promote(results):
        print("✅ Candidate promoted")
    else:
        print("❌ Candidate rejected")
