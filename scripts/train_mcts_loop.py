import os
import argparse

from src.engine.chess_uci_engine import SimpleEngine
from src.training.self_play import SelfPlayGenerator
from src.training.trainer import train_from_selfplay
from scripts.arena_eval import arena_eval


# =========================
# DEFAULT CONFIG
# =========================
CHECKPOINT_DIR = "checkpoints"


# =========================
# MAIN LOOP
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--sims", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    device = args.device
    current_model = args.base_model

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for it in range(1, args.iterations + 1):
        print(f"\n🔥 MCTS ITERATION {it}")
        print(f"Current model: {current_model}")
        print(f"Running on device: {device}")

        # ---------- LOAD ENGINE ----------
        engine = SimpleEngine(current_model, device=device)

        # ---------- SELF PLAY ----------
        sp = SelfPlayGenerator(
            engine=engine,
            encoder=engine.encoder,
            max_moves=200,
        )

        data = sp.generate_games(
            num_games=args.games,
            sims=args.sims,
        )

        print("📊 Positions:", data["num_positions"])
        print("📊 Results:", data["results"])

        # ---------- TRAIN CANDIDATE ----------
        candidate_path = os.path.join(
            CHECKPOINT_DIR, f"candidate_iter_{it}.pt"
        )

        print("🧠 Training candidate model...")
        train_from_selfplay(
            data=data,
            base_model_path=current_model,
            save_path=candidate_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        # ---------- ARENA ----------
        print("\n⚔️ Arena evaluation...")
        results = arena_eval(
            new_model_path=candidate_path,
            old_model_path=current_model,
        )

        print("Arena results:", results)

        # ---------- PROMOTION (EARLY TRAINING) ----------
        if results["new"] > results["old"]:
            print("🚀 New model PROMOTED")
            promoted_path = os.path.join(
                CHECKPOINT_DIR, f"mcts_iter_{it}.pt"
            )
            os.replace(candidate_path, promoted_path)
            current_model = promoted_path
        else:
            print("❌ Candidate rejected")
            os.remove(candidate_path)

    print("\n✅ TRAINING COMPLETE")
    print("Final best model:", current_model)


if __name__ == "__main__":
    main()
