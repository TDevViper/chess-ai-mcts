# scripts/train_loop.py

import torch
from src.engine.chess_uci_engine import SimpleEngine
from  src.training.anti_collapse_self_play import SafeAntiCollapseSelfPlayGenerator
from src.training.chess_trainer import ChessTrainer
from src.core.board_encoder import BoardEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ITERATIONS = 20
GAMES_PER_ITER = 40
EPOCHS = 4
BATCH_SIZE = 128

model_path = "checkpoints/stockfish_bootstrap.pt"

encoder = BoardEncoder()

for it in range(1, ITERATIONS + 1):
    print(f"\n🔥 ITERATION {it}")

    # Load engine
    engine = SimpleEngine(model_path, device=DEVICE)

    # Self-play
    sp = SafeAntiCollapseSelfPlayGenerator(engine, encoder)
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
