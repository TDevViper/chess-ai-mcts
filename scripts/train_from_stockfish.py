from src.training.teacher_self_play import StockfishTeacherGenerator
from src.core.board_encoder import BoardEncoder
from src.training.trainer import train_from_batches
from src.engine.chess_uci_engine import SimpleEngine

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
OUTPUT_MODEL = "checkpoints/stockfish_bootstrap.pt"

encoder = BoardEncoder()
teacher = StockfishTeacherGenerator(
    encoder,
    STOCKFISH_PATH
)

all_states = []
all_policies = []
all_values = []

print("🎓 Generating Stockfish games...")

for i in range(20):
    s, p, v = teacher.generate_game(depth=8)
    all_states.append(s)
    all_policies.append(p)
    all_values.append(v)
    print(f"Game {i+1}/20 done")

teacher.close()

data = {
    "board_states": torch.cat(all_states),
    "policy_targets": torch.cat(all_policies),
    "value_targets": torch.cat(all_values),
}

print("🧠 Training from Stockfish...")
train_from_batches(
    data,
    save_path=OUTPUT_MODEL,
    epochs=5,
    batch_size=64
)

print("✅ Saved:", OUTPUT_MODEL)
