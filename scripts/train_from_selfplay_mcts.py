# scripts/train_from_selfplay_mcts.py

import torch
from src.engine.chess_uci_engine import SimpleEngine
from src.core.board_encoder import BoardEncoder
from src.training.self_play import SelfPlayGenerator
from src.training.chess_trainer import ChessTrainer

DEVICE = "cpu"   # change to "cuda" later
MODEL_IN = "checkpoints/selfplay_iter_10.pt"
MODEL_OUT = "checkpoints/mcts_iter_1.pt"

NUM_GAMES = 5
MCTS_SIMS = 25
EPOCHS = 2
BATCH_SIZE = 64

print("\n🔄 Loading engine...")
engine = SimpleEngine(MODEL_IN, device=DEVICE)
encoder = BoardEncoder()

print("\n🎮 Generating MCTS self-play games...")
sp = SelfPlayGenerator(engine, encoder)
data = sp.generate_games(num_games=NUM_GAMES, sims=MCTS_SIMS)

print("\n📊 Self-play summary:")
print("Positions:", data["num_positions"])
print("Results:", data["results"])

print("\n🧠 Training model...")
trainer = ChessTrainer(engine.model, device=DEVICE)
trainer.train(
    training_data=data,
    num_epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

trainer.save_checkpoint(MODEL_OUT, epoch=1)

print("\n✅ MCTS training complete")
print("Saved:", MODEL_OUT)
