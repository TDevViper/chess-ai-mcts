# Self-Trained Chess Neural Network Engine

Project skeleton.🎯 Self-Trained Chess Neural Network Engine
A complete chess-playing AI system that learns through neural networks and reinforcement learning, then plays against real humans on Lichess.org.

🌟 Features
Neural Network Architecture: Deep convolutional network with residual blocks (inspired by AlphaZero)
Reinforcement Learning: Learns through self-play with policy and value heads
UCI Protocol: Standard chess engine interface
Lichess Integration: Plays real humans online
Modular Design: Easy to extend and experiment with
Training Pipeline: Complete training infrastructure with checkpointing
📋 Project Structure
chess-ai/
├── chess_board_encoder.py      # Board state encoding
├── chess_neural_network.py     # Neural network architecture
├── chess_self_play.py          # Self-play game generation
├── chess_trainer.py            # Training loop
├── chess_uci_engine.py         # UCI engine wrapper
├── lichess_bot.py              # Lichess bot integration
├── requirements.txt            # Python dependencies
├── checkpoints/                # Saved models
└── logs/                       # Training logs
🛠️ Installation
1. Prerequisites
Python 3.8+
CUDA-capable GPU (optional, but recommended for training)
2. Install Dependencies
bash
pip install torch torchvision
pip install python-chess
pip install requests
pip install numpy
Or use the requirements file:

bash
pip install -r requirements.txt
3. Verify Installation
python
python -c "import chess; import torch; print('Setup successful!')"
🚀 Quick Start
Phase 1: Train the Model
python
from chess_neural_network import ChessNet
from chess_board_encoder import BoardEncoder
from chess_trainer import train_chess_model

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ChessNet(num_filters=128, num_residual_blocks=10)
encoder = BoardEncoder()

# Train (this will take several hours)
trained_model, trainer = train_chess_model(
    model=model,
    encoder=encoder,
    num_iterations=50,          # More iterations = better play
    games_per_iteration=100,    # More games = more data
    epochs_per_iteration=10,    # Training epochs per iteration
    batch_size=256,
    device=device
)

# Save final model
trainer.save_checkpoint("checkpoints/chess_model_v1.pt", epoch='final')
Phase 2: Test the Engine
python
from chess_uci_engine import SimpleEngine
import chess

# Load trained model
engine = SimpleEngine("checkpoints/chess_model_v1.pt")

# Play a game
board = chess.Board()
move = engine.get_move(board)
print(f"Best move: {move}")
Phase 3: Deploy to Lichess
3.1 Create Lichess Bot Account
Create a new account on Lichess.org (this will become your bot)
Upgrade to bot account: Settings → Bot → Upgrade to Bot
Generate API token: Settings → API Access Tokens → Create new token
Select scopes: bot:play, challenge:read, challenge:write
3.2 Run the Bot
bash
python lichess_bot.py \
    --token YOUR_LICHESS_API_TOKEN \
    --model checkpoints/chess_model_v1.pt \
    --device cuda  # or cpu
3.3 Challenge Your Bot
Go to lichess.org/@/YOUR_BOT_USERNAME
Click "Challenge to a game"
Your bot will accept and play!
📊 Training Pipeline Details
Architecture Specifications
Input: (batch, 18, 8, 8)
  ├─ 12 planes: piece positions (6 white + 6 black)
  ├─ 4 planes: castling rights
  ├─ 1 plane: turn indicator
  └─ 1 plane: en passant

Network:
  ├─ Conv2D (18 → 128 filters)
  ├─ 10x Residual Blocks (128 filters)
  ├─ Policy Head → 4096 moves
  └─ Value Head → [-1, 1] evaluation

Total Parameters: ~2-5M (depending on configuration)
Training Strategy
Self-Play Generation: Model plays against itself
Experience Collection: Store positions with policy/value targets
Batch Training: Train on collected data
Iteration: Repeat with improved model
Hyperparameters
Parameter	Default	Description
num_filters	128	Convolution filters
num_residual_blocks	10	Network depth
learning_rate	0.001	Initial LR
batch_size	256	Training batch size
temperature	1.0	Exploration temperature
🎮 Usage Examples
Example 1: Quick Training Test
python
# Small test run (finishes in minutes)
model = ChessNet(num_filters=64, num_residual_blocks=5)
trained_model, _ = train_chess_model(
    model=model,
    encoder=BoardEncoder(),
    num_iterations=3,
    games_per_iteration=20,
    epochs_per_iteration=5,
    device='cpu'
)
Example 2: UCI Engine Mode
bash
# Run as UCI engine (for GUIs like Arena, ChessBase)
python chess_uci_engine.py checkpoints/chess_model_v1.pt
Then in your chess GUI:

Add engine → Browse to chess_uci_engine.py
Play against it!
Example 3: Engine vs Engine
python
from chess_uci_engine import SimpleEngine
import chess

engine1 = SimpleEngine("checkpoints/model_v1.pt")
engine2 = SimpleEngine("checkpoints/model_v2.pt")

board = chess.Board()
while not board.is_game_over():
    if board.turn == chess.WHITE:
        move = engine1.get_move(board)
    else:
        move = engine2.get_move(board)
    board.push(move)
    print(board)

print(f"Result: {board.result()}")
📈 Performance Tracking
Monitor Training Progress
The trainer saves metrics during training:

python
# Load training history
import json
with open('checkpoints/training_history.json', 'r') as f:
    history = json.load(f)

# Plot losses
import matplotlib.pyplot as plt
plt.plot(history['epoch'], history['train_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
Evaluate on Lichess
Track your bot's performance:

Go to lichess.org/@/YOUR_BOT_USERNAME
View rating, games played, win rate
Analyze games to see where model succeeds/fails
🔧 Advanced Configuration
Customize Network Architecture
python
model = ChessNet(
    input_channels=18,          # Board encoding planes
    num_filters=256,            # More = stronger but slower
    num_residual_blocks=20      # Deeper = more capacity
)
Training with Existing Games
python
# Load PGN games and train on them (TODO: implement)
# This can bootstrap learning before self-play
Multi-GPU Training
python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
🐛 Troubleshooting
Issue: Out of Memory During Training
Solution: Reduce batch size or number of filters

python
train_chess_model(
    model=model,
    batch_size=64,  # Instead of 256
    # ...
)
Issue: Bot Not Accepting Challenges
Solution: Check Lichess API token has correct permissions

Go to Lichess → API Access Tokens
Ensure bot:play, challenge:read, challenge:write are enabled
Issue: Model Makes Illegal Moves
Solution: This shouldn't happen due to legal move masking, but if it does:

Check get_legal_move_mask() is working correctly
Ensure model predicts from top legal moves
Issue: Training Too Slow
Solution:

Use GPU: device='cuda'
Reduce num_residual_blocks
Reduce games_per_iteration
🎯 Next Steps & Improvements
Short Term
 Basic neural network
 Self-play generation
 UCI engine
 Lichess integration
 Opening book integration
 Endgame tablebase support
Medium Term
 Monte Carlo Tree Search (MCTS)
 Parallel self-play on multiple GPUs
 Train from master games database
 Web interface for local play
Long Term
 Distributed training across multiple machines
 Advanced exploration strategies
 Automatic curriculum learning
 Multi-variant support (Chess960, etc.)
📚 References
AlphaZero Paper
Lichess Bot API
UCI Protocol
python-chess Documentation
🤝 Contributing
This is a learning project! Feel free to:

Experiment with different architectures
Improve the training pipeline
Add new features
Share your results
📄 License
MIT License - Feel free to use and modify

🎉 Acknowledgments
Inspired by:

DeepMind's AlphaZero
Stockfish neural network efforts
The amazing python-chess library
Ready to train your chess AI? Start with the Quick Start guide above! 🚀

