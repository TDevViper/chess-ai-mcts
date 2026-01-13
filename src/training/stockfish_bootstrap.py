import chess
import chess.engine
import torch
from src.core.board_encoder import BoardEncoder


# ==================================================
# SOFT STOCKFISH POLICY (multipv + softmax)
# ==================================================
def stockfish_policy(
    board,
    engine,
    encoder,
    multipv=5,
    temperature=1.0,
):
    info = engine.analyse(
        board,
        chess.engine.Limit(time=0.05),  # time-based, not depth-based
        multipv=multipv,
    )

    policy = torch.zeros(4096)
    scores = []

    for entry in info:
        if "pv" not in entry or len(entry["pv"]) == 0:
            scores.append(-1e9)
            continue

        score = entry["score"].pov(board.turn).score(mate_score=10000)
        scores.append(score)

    scores = torch.tensor(scores, dtype=torch.float32)
    probs = torch.softmax(scores / temperature, dim=0)

    for p, entry in zip(probs, info):
        if "pv" not in entry or len(entry["pv"]) == 0:
            continue

        move = entry["pv"][0]
        idx = encoder.encode_move(move, board)
        policy[idx] = p

    return policy


# ==================================================
# GENERATE STOCKFISH BOOTSTRAP DATA
# ==================================================
def generate_stockfish_data(
    stockfish_path,
    games=20,
    multipv=5,
    temperature=1.0,
    max_moves=200,
):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    # 🔥 KEY CHANGE: ELO-LIMITED STOCKFISH
    engine.configure({
        "Skill Level": 3,
        "UCI_LimitStrength": True,
        "UCI_Elo": 1320,
    })

    encoder = BoardEncoder()

    states = []
    policies = []
    values = []

    for g in range(games):
        board = chess.Board()
        game_states = []
        game_policies = []

        while not board.is_game_over() and board.fullmove_number < max_moves:
            game_states.append(board.copy())

            # ---- SOFT POLICY FROM WEAK STOCKFISH ----
            policy = stockfish_policy(
                board,
                engine,
                encoder,
                multipv=multipv,
                temperature=temperature,
            )
            game_policies.append(policy)

            # Play best move from policy
            move_idx = torch.argmax(policy).item()
            move = encoder.decode_move(move_idx, board)

            if move not in board.legal_moves:
                break

            board.push(move)

        # ---------- FINAL RESULT ----------
        if board.result() == "1-0":
            z = 1.0
        elif board.result() == "0-1":
            z = -1.0
        else:
            z = 0.0

        for b, p in zip(game_states, game_policies):
            states.append(encoder.encode_board(b))
            policies.append(p)
            values.append(z if b.turn == chess.WHITE else -z)

        print(f"🎓 Stockfish (Elo 1320) game {g + 1}/{games} done")

    engine.quit()

    return {
        "board_states": torch.stack(states),
        "policy_targets": torch.stack(policies),
        "value_targets": torch.tensor(values).unsqueeze(1),
    }
