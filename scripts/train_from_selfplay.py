# scripts/train_from_selfplay.py

import torch
import chess

from src.engine.chess_uci_engine import SimpleEngine
from src.core.board_encoder import BoardEncoder
from src.training.self_play import SelfPlayGenerator
from src.training.chess_trainer import ChessTrainer

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "checkpoints/fresh_model.pt"
DEVICE = "cpu"
NUM_GAMES = 3        # small on purpose
EPOCHS = 1           # JUST 1 epoch (sanity check)
BATCH_SIZE = 64

# ----------------------------
# LOAD ENGINE
# ----------------------------
print("🔄 Loading engine...")
engine = SimpleEngine(MODEL_PATH, device=DEVICE)
encoder = BoardEncoder()

# ----------------------------
# SELF PLAY
# ----------------------------
print("\n🎮 Generating self-play games...")
self_play = SelfPlayGenerator(engine, encoder)

data = self_play.generate_games(
    num_games=NUM_GAMES,
    temperature=1.5
)

print("\n📊 Self-play summary:")
print("Positions:", data["num_positions"])
print("Results:", data["results"])

# ----------------------------
# TRAINING
# ----------------------------
print("\n🧠 Training model (1 epoch only)...")

trainer = ChessTrainer(engine.model, device=DEVICE)

trainer.train(
    training_data=data,
    num_epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# ----------------------------
# SAVE MODEL
# ----------------------------
SAVE_PATH = "checkpoints/selfplay_step1.pt"
trainer.save_checkpoint(SAVE_PATH, epoch=1)

print("\n✅ Training complete")
print("Model saved to:", SAVE_PATH)
