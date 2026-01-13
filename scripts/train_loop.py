# scripts/train_loop.py

import torch
from src.engine.chess_uci_engine import SimpleEngine
from src.training.self_play import SelfPlayGenerator
from src.training.chess_trainer import ChessTrainer
from src.core.board_encoder import BoardEncoder

DEVICE = "cpu"  # change to "cuda" later
ITERATIONS = 10
GAMES_PER_ITER = 5
EPOCHS = 2
BATCH_SIZE = 64

model_path = "checkpoints/selfplay_step1.pt"

encoder = BoardEncoder()

for it in range(1, ITERATIONS + 1):
    print(f"\n🔥 ITERATION {it}")

    # Load engine
    engine = SimpleEngine(model_path, device=DEVICE)

    # Self-play
    sp = SelfPlayGenerator(engine, encoder)
    data = sp.generate_games(num_games=GAMES_PER_ITER)

    print("Positions:", data["num_positions"])
    print("Results:", data["results"])

    # Train
    trainer = ChessTrainer(engine.model, device=DEVICE)
    trainer.train(
        training_data=data,
        num_epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # Save new model
    model_path = f"checkpoints/selfplay_iter_{it}.pt"
    trainer.save_checkpoint(model_path, epoch=it)

    print(f"✅ Saved {model_path}")
