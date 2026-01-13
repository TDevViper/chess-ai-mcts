import chess
import chess.engine
import torch
from src.engine.chess_uci_engine import SimpleEngine
from src.core.board_encoder import BoardEncoder

STOCKFISH_PATH = "/usr/bin/stockfish"  # mac: brew install stockfish
MODEL_PATH = "checkpoints/mcts_iter_5.pt"

GAMES = 20
SF_SKILL = 3

engine_nn = SimpleEngine(MODEL_PATH, device="cpu")
encoder = BoardEncoder()

sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
sf.configure({"Skill Level": SF_SKILL})

states, policies, values = [], [], []

for g in range(GAMES):
    board = chess.Board()
    game_states = []
    game_policies = []

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move, policy, _ = engine_nn.get_move_mcts_with_policy(
                board, sims=50, temperature=1.0
            )
            game_states.append(encoder.encode_board(board))
            game_policies.append(policy)
        else:
            move = sf.play(board, chess.engine.Limit(time=0.1)).move

        board.push(move)

    # ----- GAME VALUE -----
    result = board.result()
    if result == "1-0":
        z = 1.0
    elif result == "0-1":
        z = -1.0
    else:
        z = 0.0

    for i in range(len(game_states)):
        states.append(game_states[i])
        policies.append(game_policies[i])
        values.append(z)

    print(f"Game {g+1}/{GAMES} finished: {result}")

sf.quit()

torch.save(
    {
        "states": torch.stack(states),
        "policies": torch.stack(policies),
        "values": torch.tensor(values).unsqueeze(1),
    },
    "data/vs_stockfish.pt"
)

print("✅ Stockfish data saved")
