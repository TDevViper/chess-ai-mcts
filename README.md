♟️ Self-Trained Chess AI (Neural Network + MCTS)

A research-oriented chess engine that learns to play chess through self-play reinforcement learning, using a policy–value neural network combined with Monte Carlo Tree Search (MCTS).

This project focuses on how chess AIs are trained, evaluated, and stabilized, not just raw playing strength.

🚀 Key Features

🧠 Policy + Value Neural Network (AlphaZero-style)

🌲 Monte Carlo Tree Search (MCTS) for move selection

🔁 Self-Play Reinforcement Learning

⚔️ Arena Evaluation (model vs model testing)

🧪 Draw / Mode-Collapse Detection & Mitigation

🔌 UCI-compatible engine (usable in chess GUIs)

⚡ GPU-accelerated training (CUDA supported)

📊 Current Training Status (Important Note)

This project has completed multiple self-play training iterations.

Observed behavior

Early iterations show clear improvement

Later iterations converge toward a draw-dominant equilibrium

Arena evaluation between distant checkpoints often results in draws

Example

selfplay_iter_18 vs selfplay_iter_10 → 20 / 20 draws


This indicates policy convergence, a known phenomenon in self-play RL,
not a bug or failure.

Breaking this equilibrium typically requires:

Larger neural networks

Much higher self-play volume

External data (e.g. master games / PGNs)

Stronger exploration or curriculum learning

🧠 Architecture Overview
Neural Network

Input: (18, 8, 8) board encoding

Backbone: Convolutional layers + residual blocks

Outputs:

Policy head: move probabilities (4096 possible moves)

Value head: position evaluation in range [-1, 1]

Inspired by AlphaZero-style policy/value learning, implemented fully from scratch.

📁 Project Structure
chess-ai-mcts/
├── src/
│   ├── core/
│   │   └── board_encoder.py        # Board → tensor encoding
│   ├── network/
│   │   └── chess_net.py            # Policy–value neural network
│   ├── engine/
│   │   ├── mcts.py                 # Monte Carlo Tree Search
│   │   └── chess_uci_engine.py     # UCI-compatible engine
│   └── training/
│       ├── self_play.py            # Self-play game generation
│       ├── trainer.py              # Training loop
│       └── anti_collapse_self_play.py
│
├── scripts/
│   ├── train_loop.py               # Main training loop
│   ├── arena_eval.py               # Model vs model evaluation
│   └── play_vs_stockfish.py
│
├── checkpoints/                    # Saved model checkpoints
└── requirements.txt

🛠️ Installation
Requirements

Python 3.9+

PyTorch

python-chess

NumPy

pip install -r requirements.txt


Verify setup:

python -c "import torch, chess; print('Setup OK')"

🔁 Training (Self-Play)

Run the full self-play + training loop:

python -m scripts.train_loop

What happens internally

Current model plays games against itself using MCTS

Positions, policies, and values are collected

Neural network is trained on generated data

New checkpoint is saved

Process repeats

⚔️ Arena Evaluation (Model vs Model)

Compare two trained checkpoints:

python -m scripts.arena_eval \
  --candidate checkpoints/selfplay_iter_18.pt \
  --baseline  checkpoints/selfplay_iter_10.pt

Arena rules

Randomized colors

Early-game exploration

Max move limit

Resign logic based on value head

Used to measure true improvement vs draw equilibrium.

🧪 Anti-Collapse Measures Implemented

To reduce draw spirals and training stagnation:

Opening diversity

Temperature scheduling

Early resignation thresholds

Draw value penalties

Repetition awareness

Reduced maximum game length

All measures are conservative, prioritizing training stability.

🎮 UCI Engine Usage

Run the engine in UCI mode (for GUIs like Arena, CuteChess, etc.):

python src/engine/chess_uci_engine.py checkpoints/selfplay_iter_18.pt


You can then add it as an engine in any UCI-compatible chess GUI.

⚠️ Limitations (Honest & Transparent)

❌ Not competitive with Stockfish or Leela

⚠️ Strength limited by compute and training volume

⚖️ Self-play equilibrium reached early

🧪 Lichess bot integration is experimental

This is a research & learning project, not a production chess engine.

🧠 What This Project Demonstrates

Strong understanding of reinforcement learning loops

Practical implementation of MCTS

Handling self-play instability

Debugging mode collapse

Building scalable ML training pipelines

Skills directly relevant to:

Game AI

Reinforcement learning research

ML / systems engineering roles

🔮 Future Improvements

Larger neural networks

External PGN bootstrapping

Parallel self-play

Curriculum learning

Opening books

Endgame tablebases

📜 License

MIT License — free to use, modify, and learn from.

🙌 Acknowledgements

Inspired by:

DeepMind’s AlphaZero

Stockfish NNUE ideas

python-chess library