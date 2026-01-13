import argparse
import torch

from src.training.stockfish_bootstrap import generate_stockfish_data
from src.training.trainer import train_from_selfplay

# ==================================================
# CLI
# ==================================================
parser = argparse.ArgumentParser()
parser.add_argument("--elo", type=int, default=1200)
parser.add_argument("--games", type=int, default=50)
parser.add_argument("--epochs", type=int, default=5)
args = parser.parse_args()

# ==================================================
# CONFIG
# ==================================================
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
OUTPUT_MODEL = "checkpoints/stockfish_bootstrap.pt"

# ==================================================
# GENERATE DATA
# ==================================================
print(f"🎓 Generating Stockfish data (Elo {args.elo})...")

data = generate_stockfish_data(
    stockfish_path=STOCKFISH_PATH,
    games=args.games,
    multipv=5,
    temperature=1.0,
)

print(f"📦 Generated {data['board_states'].shape[0]} positions")

# ==================================================
# TRAIN
# ==================================================
print("🧠 Training from Stockfish bootstrap...")

train_from_selfplay(
    data=data,
    base_model_path=None,        # fresh bootstrap
    save_path=OUTPUT_MODEL,
    epochs=args.epochs,
    batch_size=64,
)

print("✅ Saved model to:", OUTPUT_MODEL)
