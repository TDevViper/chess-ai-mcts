import torch
from src.training.trainer import train_from_tensors

data = torch.load("data/vs_stockfish.pt")

train_from_tensors(
    states=data["states"],
    policy_targets=data["policies"],
    value_targets=data["values"],
    base_model_path="checkpoints/mcts_iter_5.pt",
    save_path="checkpoints/vs_stockfish.pt",
    epochs=5,
    batch_size=64,
)
