import torch
from pathlib import Path
from src.network.chess_net import ChessNet

# ensure checkpoints folder exists
Path("checkpoints").mkdir(exist_ok=True)

# create REAL ChessNet (same architecture engine expects)
model = ChessNet(
    input_channels=18,
    num_filters=64,
    num_residual_blocks=2
)

checkpoint = {
    "model_state_dict": model.state_dict(),
    "model_config": {
        "input_channels": 18,
        "num_filters": 64,
        "num_residual_blocks": 2
    }
}

torch.save(checkpoint, "checkpoints/model.pt")
print("✅ Real base ChessNet model saved: checkpoints/model.pt")
