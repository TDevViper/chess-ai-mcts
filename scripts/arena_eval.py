import os
import chess
import torch

from src.engine.mcts import MCTS
from src.engine.chess_uci_engine import SimpleEngine

# ==================================================
# CONFIG (EARLY TRAINING)
# ==================================================
OLD_MODEL_PATH = "checkpoints/stockfish_bootstrap.pt"
DEFAULT_NEW_MODEL_PATH = "checkpoints/stockfish_bootstrap.pt"

GAMES = 20

# 🔥 LOWER ARENA STRENGTH (IMPORTANT)
SIMS = 120          # was 200 → too draw-heavy
DEVICE = "cpu"
TEMPERATURE = 0.0

# 🔥 HARD STOP (PREVENT INFINITE GAMES)
MAX_ARENA_MOVES = 300

# ==================================================
# LOAD ENGINE
# ==================================================
def load_engine(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    base = SimpleEngine(model_path, device=DEVICE)
    return MCTS(base, simulations=SIMS)

# ==================================================
# PLAY ONE ARENA GAME (SAFE + GUARANTEED RETURN)
# ==================================================
def play_arena_game(engine_new, engine_old):
    board = chess.Board()

    # Randomize colors
    new_is_white = bool(torch.randint(0, 2, (1,)).item())

    move_count = 0

    while not board.is_game_over() and move_count < MAX_ARENA_MOVES:
        if board.turn == chess.WHITE:
            engine = engine_new if new_is_white else engine_old
        else:
            engine = engine_old if new_is_white else engine_new

        move, _, _ = engine.search(board, temperature=TEMPERATURE)

        # ---------- SAFETY ----------
        if move not in board.legal_moves:
            return "draw"

        board.push(move)
        move_count += 1

    # ---------- FORCED DRAW ----------
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
# PROMOTION RULE (EARLY TRAINING)
# ==================================================
def should_promote(results):
    """
    EARLY PHASE RULE:
    Promote if new beats old even by 1 game.
    Tighten later.
    """
    return results["new"] > results["old"]

# ==================================================
# MAIN (STANDALONE TEST)
# ==================================================
if __name__ == "__main__":
    results = arena_eval(
        new_model_path=DEFAULT_NEW_MODEL_PATH,
        old_model_path=OLD_MODEL_PATH,
    )

    if should_promote(results):
        print("✅ Candidate promoted")
    else:
        print("❌ Candidate rejected")
