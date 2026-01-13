import torch
from torch.utils.data import DataLoader, TensorDataset
from src.network.chess_net import ChessNet


def train_from_selfplay(
    data,
    base_model_path: str | None,
    save_path: str,
    epochs: int = 2,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
):
    """
    Train model from self-play data and save checkpoint.
    """

    # -----------------------------
    # MODEL CONFIG
    # -----------------------------
    default_config = {
        "input_channels": 18,
        "num_filters": 64,
        "num_residual_blocks": 2,
    }

    # -----------------------------
    # LOAD / INIT MODEL
    # -----------------------------
    if base_model_path is None:
        print("🆕 Initializing fresh model")
        config = default_config
        model = ChessNet(**config).to(device)

    else:
        print(f"📦 Loading base model from {base_model_path}")
        checkpoint = torch.load(base_model_path, map_location=device)

        config = checkpoint.get("model_config", default_config)
        model = ChessNet(**config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -----------------------------
    # DATASET
    # -----------------------------
    dataset = TensorDataset(
        data["board_states"],
        data["policy_targets"],
        data["value_targets"],
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    # -----------------------------
    # TRAIN LOOP
    # -----------------------------
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        batches = 0

        for boards, target_policies, target_values in loader:
            boards = boards.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)

            optimizer.zero_grad()

            policy_logits, values = model(boards)

            # Policy loss (cross-entropy with soft targets)
            log_probs = torch.log_softmax(policy_logits, dim=1)
            policy_loss = -(target_policies * log_probs).sum(dim=1).mean()

            # Value loss (MSE)
            value_loss = torch.mean((values - target_values) ** 2)

            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_policy += policy_loss.item()
            total_value += value_loss.item()
            batches += 1

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Loss: {total_loss / batches:.4f} | "
            f"Policy: {total_policy / batches:.4f} | "
            f"Value: {total_value / batches:.4f}"
        )

    # -----------------------------
    # SAVE CHECKPOINT
    # -----------------------------
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config,
        },
        save_path,
    )

    print(f"✅ Model saved to {save_path}")
