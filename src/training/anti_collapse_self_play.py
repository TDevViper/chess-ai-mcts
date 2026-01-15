"""
SAFE Anti-Collapse Self-Play Generator
Conservative fixes that won't break your training
"""

import chess
import torch
import random


class SafeAntiCollapseSelfPlayGenerator:
    """
    Self-play with SAFE anti-collapse measures.
    No dangerous LR changes, just smart diversity.
    """
    
    def __init__(self, engine, encoder, max_moves=90):
        self.engine = engine
        self.encoder = encoder
        self.max_moves = max_moves  # Reduced from 200
        
    def generate_games(self, num_games=40, sims=100):
        all_states, all_policies, all_value_targets = [], [], []
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
        
        for g in range(num_games):
            states, policies, value_targets, result = self.play_one_game(sims, game_num=g)
            
            all_states.extend(states)
            all_policies.extend(policies)
            all_value_targets.extend(value_targets)
            
            results[result] += 1
            print(f"🔥 Game {g + 1}/{num_games} finished: {result}")
        
        # Warning if too many draws
        draw_rate = results["1/2-1/2"] / num_games
        if draw_rate > 0.75:
            print(f"\n⚠️ WARNING: {draw_rate:.0%} draws detected")
        
        return {
            "board_states": torch.stack(all_states),
            "policy_targets": torch.stack(all_policies),
            "value_targets": torch.tensor(all_value_targets, dtype=torch.float32).unsqueeze(1),
            "results": results,
            "num_positions": len(all_states),
        }
    
    def play_one_game(self, sims, game_num=0):
        board = chess.Board()
        
        # ✅ SAFE FIX 1: Random opening diversity
        opening_moves = self._get_opening_for_game(game_num)
        for move_uci in opening_moves:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
            except:
                break
        
        board_states = []
        policy_targets = []
        turns = []
        
        recent_values = []
        RESIGN_THRESHOLD = -0.80  # ✅ Slightly more aggressive than -0.90
        RESIGN_COUNT = 4
        
        result = None
        move_count = len(board.move_stack)
        
        while not board.is_game_over() and move_count < self.max_moves:
            # ✅ SAFE FIX 2: Better temperature schedule
            if move_count < 20:
                temperature = 1.2  # Higher early exploration
            elif move_count < 50:
                temperature = 0.8
            else:
                temperature = 0.3
            
            move, mcts_policy, net_value = self.engine.get_move_mcts_with_policy(
                board, sims=sims, temperature=temperature
            )
            
            # Store data
            board_states.append(self.encoder.encode_board(board))
            policy_targets.append(mcts_policy)
            turns.append(board.turn)
            
            # Resign logic
            recent_values.append(net_value)
            if len(recent_values) > RESIGN_COUNT:
                recent_values.pop(0)
            
            if len(recent_values) == RESIGN_COUNT:
                avg_value = sum(recent_values) / RESIGN_COUNT
                if avg_value < RESIGN_THRESHOLD:
                    result = "0-1" if board.turn == chess.WHITE else "1-0"
                    print(f"⚠️ RESIGN | move={move_count} | avg={avg_value:.2f}")
                    break
            
            board.push(move)
            move_count += 1
        
        # Determine result
        if result is None:
            if board.is_game_over():
                result = board.result()
            else:
                result = "1/2-1/2"
        if not turns:
            return [], [], [], "1/2-1/2"

        # ✅ SAFE FIX 3: Modest draw penalty
        value_targets = self._compute_value_targets(result, turns)
        
        return board_states, policy_targets, value_targets, result
    
    def _get_opening_for_game(self, game_num):
        """Return opening moves for diversity"""
        openings = [
            [],  # Standard
            ["e2e4"],
            ["d2d4"],
            ["g1f3"],
            ["c2c4"],
            ["e2e4", "e7e5"],
            ["d2d4", "d7d5"],
            ["e2e4", "c7c5"],
            ["e2e4", "e7e6"],
            ["d2d4", "g8f6"],
        ]
        return openings[game_num % len(openings)]
    
    def _compute_value_targets(self, result, turns):
        """Compute value targets with MODEST draw penalty"""
        if result == "1-0":
            outcome_white = 1.0
            outcome_black = -1.0
        elif result == "0-1":
            outcome_white = -1.0
            outcome_black = 1.0
        else:
            # ✅ SAFE: Modest draw penalty
            outcome_white = -0.2  # Not too aggressive
            outcome_black = -0.2
        
        value_targets = []
        for turn in turns:
            if turn == chess.WHITE:
                value_targets.append(outcome_white)
            else:
                value_targets.append(outcome_black)
        
        return value_targets


# ✅ SAFE TRAINING SCRIPT
def safe_restart_training(base_model="checkpoints/selfplay_iter_1.pt"):
    """
    Safe restart with conservative anti-collapse measures.
    """
    print("="*60)
    print("🔧 SAFE ANTI-COLLAPSE RESTART")
    print("="*60)
    print("\n✅ Safe measures:")
    print("  - Learning rate: 3e-4 (conservative)")
    print("  - Value weight: 1.5 (not 3.0!)")
    print("  - Temperature: 1.2 early (not 1.5)")
    print("  - Draw penalty: -0.2 (modest)")
    print("  - Max moves: 90")
    print("  - Resign: -0.80")
    print("="*60 + "\n")
    
    # Import your actual project structure
    from src.engine.chess_uci_engine import SimpleEngine
    from src.core.board_encoder import BoardEncoder
    from src.training.trainer import ChessTrainer  # Must support value_weight!
    from src.network.chess_net import ChessNet
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = BoardEncoder()
    
    # Load engine
    print(f"Loading model: {base_model}")
    engine = SimpleEngine(base_model, device=device)
    
    # Generate games with safe anti-collapse
    print("\nGenerating self-play games...")
    generator = SafeAntiCollapseSelfPlayGenerator(engine, encoder, max_moves=90)
    data = generator.generate_games(num_games=40, sims=100)
    
    print(f"\n📊 Results:")
    print(f"  Positions: {data['num_positions']}")
    print(f"  Game results: {data['results']}")
    print(f"  Avg length: {data['num_positions'] / 40:.1f} moves/game")
    
    # Check health
    draw_rate = data['results']['1/2-1/2'] / 40
    decisive_rate = 1 - draw_rate
    
    if draw_rate > 0.85:
        print(f"\n❌ Still too many draws ({draw_rate:.0%})")
        print("   Consider:")
        print("   - Increase draw penalty to -0.3")
        print("   - Lower resign to -0.75")
        return None
    
    print(f"\n✅ Healthy diversity: {decisive_rate:.0%} decisive games")
    
    # Load model
    checkpoint = torch.load(base_model, map_location=device)
    model = ChessNet(
        input_channels=checkpoint["model_config"]["input_channels"],
        num_filters=checkpoint["model_config"]["num_filters"],
        num_residual_blocks=checkpoint["model_config"]["num_residual_blocks"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    # Train with SAFE settings
    print("\n🧠 Training...")
    trainer = ChessTrainer(
        model=model,
        device=device,
        learning_rate=3e-4,  # ✅ SAFE, not 0.002!
        value_weight=1.5     # ✅ Modest boost, not 3.0!
    )
    
    trainer.train(
        training_data=data,
        num_epochs=6,  # Reasonable
        batch_size=128
    )
    
    # Save
    output_path = "checkpoints/safe_restart_iter_1.pt"
    trainer.save_checkpoint(output_path, iteration=1)
    print(f"\n✅ Saved to {output_path}")
    
    return output_path


if __name__ == "__main__":
    print("🔧 SAFE Anti-Collapse Training")
    print("\nThis uses CONSERVATIVE fixes:")
    print("  ✅ No dangerous LR spikes")
    print("  ✅ Modest value weight")
    print("  ✅ Reasonable draw penalty")
    print("  ✅ Smart temperature schedule")
    print("  ✅ Opening diversity")
    print("\nStarting from: selfplay_iter_1.pt\n")
    
    model = safe_restart_training("checkpoints/selfplay_iter_1.pt")
    
    if model:
        print("\n✅ Safe restart complete!")
        print(f"Continue from: {model}")
    else:
        print("\n⚠️ Need stronger measures - contact for next steps")