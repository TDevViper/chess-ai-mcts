import os
from src.engine.chess_uci_engine import SimpleEngine
from src.training.self_play import SelfPlayGenerator
from src.training.trainer import train_from_selfplay
from scripts.arena_eval import arena_eval, should_promote


# =========================
# CONFIG  (EARLY TRAINING SAFE)
# =========================
START_MODEL = "checkpoints/stockfish_bootstrap.pt"
CHECKPOINT_DIR = "checkpoints"

ITERATIONS = 5

# 🔥 MOST IMPORTANT CHANGE
GAMES_PER_ITER = 32        # was 8 (too low)

MCTS_SIMS = 50

# 🔥 REDUCE OVERFITTING
EPOCHS = 2                # was 5

BATCH_SIZE = 64


# =========================
# MAIN LOOP
# =========================
def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    current_model = START_MODEL

    for it in range(1, ITERATIONS + 1):
        print(f"\n🔥 MCTS ITERATION {it}")
        print(f"Current model: {current_model}")

        # ---------- LOAD ENGINE ----------
        engine = SimpleEngine(current_model, device="cpu")

        # ---------- SELF PLAY ----------
        sp = SelfPlayGenerator(
            engine=engine,
            encoder=engine.encoder,
            max_moves=200,
        )

        data = sp.generate_games(
            num_games=GAMES_PER_ITER,
            sims=MCTS_SIMS,
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
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
        )

        # ---------- ARENA ----------
        print("\n⚔️ Arena evaluation...")
        results = arena_eval(
            new_model_path=candidate_path,
            old_model_path=current_model,
        )

        print("Arena results:", results)

        # ---------- PROMOTION ----------
        # 🔥 TEMPORARILY LOOSEN FOR FIRST PROMOTION
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
