# Engine Test Script
"""
Test Script for Chess Neural Network Engine
Test and analyze your trained model
"""

import chess
import chess.pgn
import torch
import argparse
from datetime import datetime
from pathlib import Path
import io

from src.engine.chess_uci_engine import SimpleEngine


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test chess neural network engine')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['interactive', 'self-play', 'vs-random', 'analyze'],
                       help='Test mode')
    parser.add_argument('--games', type=int, default=10,
                       help='Number of games for self-play/vs-random modes')
    parser.add_argument('--fen', type=str, default=None,
                       help='FEN position to analyze (for analyze mode)')
    parser.add_argument('--save-pgn', type=str, default=None,
                       help='Save games to PGN file')
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup compute device"""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    return device


def interactive_mode(engine):
    """
    Interactive mode - play against the engine
    """
    print("\n" + "="*60)
    print("Interactive Mode - Play Against the Engine")
    print("="*60)
    print("Commands:")
    print("  <move>  - Make a move (e.g., 'e2e4', 'Nf3')")
    print("  'show'  - Show current board")
    print("  'hint'  - Get engine suggestion")
    print("  'undo'  - Undo last move")
    print("  'quit'  - Exit")
    print("="*60 + "\n")
    
    board = chess.Board()
    print(board)
    print()
    
    while not board.is_game_over():
        # Human's turn
        if board.turn == chess.WHITE:
            print("Your turn (White):")
            user_input = input("> ").strip().lower()
            
            if user_input == 'quit':
                break
            elif user_input == 'show':
                print(board)
                continue
            elif user_input == 'hint':
                hint = engine.get_move(board)
                print(f"Engine suggests: {hint}")
                continue
            elif user_input == 'undo':
                if len(board.move_stack) >= 2:
                    board.pop()
                    board.pop()
                    print(board)
                else:
                    print("Cannot undo")
                continue
            
            # Try to parse move
            try:
                # Try UCI format first
                move = chess.Move.from_uci(user_input)
                if move not in board.legal_moves:
                    raise ValueError("Illegal move")
            except:
                # Try SAN format
                try:
                    move = board.parse_san(user_input)
                except:
                    print("Invalid move. Try again.")
                    continue
            
            board.push(move)
            print(board)
            print()
        
        # Engine's turn
        else:
            print("Engine thinking...")
            move = engine.get_move(board)
            print(f"Engine plays: {move}")
            board.push(move)
            print(board)
            print()
    
    print(f"\nGame Over!")
    print(f"Result: {board.result()}")
    if board.is_checkmate():
        winner = "White" if not board.turn else "Black"
        print(f"{winner} wins by checkmate!")
    elif board.is_stalemate():
        print("Draw by stalemate")
    elif board.is_insufficient_material():
        print("Draw by insufficient material")


def self_play_mode(engine, num_games=10, save_pgn=None):
    """
    Self-play mode - engine plays against itself
    """
    print("\n" + "="*60)
    print(f"Self-Play Mode - {num_games} Games")
    print("="*60 + "\n")
    
    results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0}
    games = []
    
    for game_num in range(num_games):
        board = chess.Board()
        move_count = 0
        max_moves = 200
        
        print(f"Playing game {game_num + 1}/{num_games}...", end=' ')
        
        while not board.is_game_over() and move_count < max_moves:
            move = engine.get_move(board)
            board.push(move)
            move_count += 1
        
        result = board.result()
        results[result] += 1
        games.append(board)
        
        print(f"Result: {result} ({move_count} moves)")
    
    # Print summary
    print("\n" + "="*60)
    print("Self-Play Summary")
    print("="*60)
    print(f"White wins: {results['1-0']}")
    print(f"Black wins: {results['0-1']}")
    print(f"Draws: {results['1/2-1/2']}")
    print(f"Average decisiveness: {(results['1-0'] + results['0-1']) / num_games:.1%}")
    print("="*60 + "\n")
    
    # Save to PGN if requested
    if save_pgn:
        save_games_to_pgn(games, save_pgn, "Self-Play")


def vs_random_mode(engine, num_games=10, save_pgn=None):
    """
    VS Random mode - engine plays against random moves
    """
    print("\n" + "="*60)
    print(f"VS Random Mode - {num_games} Games")
    print("="*60 + "\n")
    
    import random
    results = {'win': 0, 'loss': 0, 'draw': 0}
    games = []
    
    for game_num in range(num_games):
        board = chess.Board()
        move_count = 0
        max_moves = 200
        
        print(f"Playing game {game_num + 1}/{num_games}...", end=' ')
        
        while not board.is_game_over() and move_count < max_moves:
            if board.turn == chess.WHITE:
                # Engine plays as white
                move = engine.get_move(board)
            else:
                # Random opponent
                move = random.choice(list(board.legal_moves))
            
            board.push(move)
            move_count += 1
        
        result = board.result()
        if result == '1-0':
            results['win'] += 1
        elif result == '0-1':
            results['loss'] += 1
        else:
            results['draw'] += 1
        
        games.append(board)
        print(f"Result: {result} ({move_count} moves)")
    
    # Print summary
    win_rate = results['win'] / num_games
    print("\n" + "="*60)
    print("VS Random Summary")
    print("="*60)
    print(f"Wins: {results['win']}/{num_games}")
    print(f"Losses: {results['loss']}/{num_games}")
    print(f"Draws: {results['draw']}/{num_games}")
    print(f"Win Rate: {win_rate:.1%}")
    print("="*60 + "\n")
    
    # Save to PGN if requested
    if save_pgn:
        save_games_to_pgn(games, save_pgn, "vs Random")


def analyze_mode(engine, fen=None):
    """
    Analyze mode - analyze a specific position
    """
    print("\n" + "="*60)
    print("Analysis Mode")
    print("="*60 + "\n")
    
    if fen:
        board = chess.Board(fen)
    else:
        board = chess.Board()
    
    print("Position:")
    print(board)
    print(f"FEN: {board.fen()}")
    print()
    
    # Get top moves
    print("Analyzing position...")
    
    # Get model's evaluation
    board_tensor = engine.encoder.encode_board(board).to(engine.device)
    legal_mask = engine.encoder.get_legal_move_mask(board).to(engine.device)
    
    engine.model.eval()
    with torch.no_grad():
        policy_logits, value = engine.model(board_tensor.unsqueeze(0))
        policy_logits = policy_logits.masked_fill(legal_mask.unsqueeze(0) == 0, float('-inf'))
        policy_probs = torch.softmax(policy_logits, dim=1).squeeze()
    
    # Get top 5 moves
    legal_moves = list(board.legal_moves)
    move_probs = []
    for move in legal_moves:
        move_idx = engine.encoder.encode_move(move)
        prob = policy_probs[move_idx].item()
        move_probs.append((prob, move))
    
    move_probs.sort(reverse=True)
    
    print("Position Evaluation:")
    print(f"  Value: {value.item():.3f} ({'White' if value.item() > 0 else 'Black'} advantage)")
    print()
    print("Top Moves:")
    for i, (prob, move) in enumerate(move_probs[:5], 1):
        print(f"  {i}. {move.uci():6s} ({board.san(move):8s}) - {prob:.2%}")


def save_games_to_pgn(games, filename, event_name):
    """Save games to PGN file"""
    pgn_path = Path(filename)
    
    with open(pgn_path, 'w') as f:
        for i, board in enumerate(games, 1):
            game = chess.pgn.Game()
            game.headers["Event"] = event_name
            game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            game.headers["Round"] = str(i)
            game.headers["White"] = "ChessNeuralNet"
            game.headers["Black"] = "ChessNeuralNet" if event_name == "Self-Play" else "Random"
            game.headers["Result"] = board.result()
            
            node = game
            for move in board.move_stack:
                node = node.add_variation(move)
            
            print(game, file=f, end="\n\n")
    
    print(f"Games saved to {pgn_path}")


def main():
    """Main test function"""
    args = parse_args()
    
    # Setup
    device = setup_device(args.device)
    
    print("\n" + "="*60)
    print("Chess Neural Network Engine - Test Suite")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print("="*60)
    
    # Load engine
    print("\nLoading engine...")
    engine = SimpleEngine(args.model, device)
    print("Engine loaded successfully!\n")
    
    # Run selected mode
    if args.mode == 'interactive':
        interactive_mode(engine)
    elif args.mode == 'self-play':
        self_play_mode(engine, args.games, args.save_pgn)
    elif args.mode == 'vs-random':
        vs_random_mode(engine, args.games, args.save_pgn)
    elif args.mode == 'analyze':
        analyze_mode(engine, args.fen)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()