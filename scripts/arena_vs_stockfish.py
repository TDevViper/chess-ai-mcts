import chess
import chess.engine

from src.engine.mcts import MCTS
from src.engine.chess_uci_engine import SimpleEngine

# ==================================================
# CONFIG
# ==================================================
MODEL_PATH = "checkpoints/mcts_iter_5.pt"
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"  # macOS brew

GAMES = 10                 # total games
SIMS = 200                 # higher sims ONLY for arena
STOCKFISH_DEPTH = 3        # 🔎 TEST 4: depth-3 survival
DEVICE = "cpu"
TEMPERATURE = 0.0          # greedy best play

# ==================================================
# LOAD ENGINES
# ==================================================
print("🔹 Loading NN engine...")
base_engine = SimpleEngine(MODEL_PATH, device=DEVICE)

print("🔹 Initializing MCTS...")
mcts = MCTS(base_engine, simulations=SIMS)

print("🔹 Loading Stockfish...")
sf_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

results = {"win": 0, "loss": 0, "draw": 0}

# ==================================================
# PLAY MATCHES
# ==================================================
for g in range(1, GAMES + 1):
    board = chess.Board()

    # Alternate colors every game
    nn_is_white = (g % 2 == 1)

    print(f"\n🎮 Game {g}/{GAMES} | NN as {'WHITE' if nn_is_white else 'BLACK'}")

    while not board.is_game_over():
        nn_turn = (
            (board.turn == chess.WHITE and nn_is_white) or
            (board.turn == chess.BLACK and not nn_is_white)
        )

        if nn_turn:
            # ---- YOUR MCTS ENGINE ----
            move, _, _ = mcts.search(board, temperature=TEMPERATURE)
        else:
            # ---- STOCKFISH (DEPTH 3) ----
            sf_result = sf_engine.play(
                board,
                chess.engine.Limit(depth=STOCKFISH_DEPTH)
            )
            move = sf_result.move

        board.push(move)

    outcome = board.result()
    print("Result:", outcome)

    # Normalize result from NN perspective
    if outcome == "1-0":
        if nn_is_white:
            results["win"] += 1
        else:
            results["loss"] += 1
    elif outcome == "0-1":
        if nn_is_white:
            results["loss"] += 1
        else:
            results["win"] += 1
    else:
        results["draw"] += 1

# ==================================================
# CLEANUP + SUMMARY
# ==================================================
sf_engine.quit()

print("\n📊 FINAL RESULTS vs STOCKFISH")
print(f"Stockfish depth: {STOCKFISH_DEPTH}")
print(f"MCTS sims: {SIMS}")
print(results)
