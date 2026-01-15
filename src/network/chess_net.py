import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChessNet(nn.Module):
    def __init__(self, input_channels=18, num_filters=64, num_residual_blocks=2):
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, 3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 4096)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_blocks(x)

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value

    def predict_move(self, board_tensor, legal_mask, temperature=1.0):
        """
        REQUIRED by engine
        """
        with torch.no_grad():
            policy_logits, value = self(board_tensor.unsqueeze(0))

            policy = policy_logits.squeeze(0)
            policy = policy.masked_fill(legal_mask == 0, -1e9)
            probs = F.softmax(policy / temperature, dim=0)

            move_idx = torch.argmax(probs).item()

        return move_idx, probs, value.item()
