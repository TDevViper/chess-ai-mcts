# Training Loop
"""
Training Loop for Chess Neural Network
Implements reinforcement learning through self-play
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from pathlib import Path
from typing import Dict
import json
import torch.nn.functional as F


class ChessDataset(Dataset):
    """Dataset for chess positions"""
    
    def __init__(self, board_states, policy_targets, value_targets):
        self.board_states = board_states
        self.policy_targets = policy_targets
        self.value_targets = value_targets
        
    def __len__(self):
        return len(self.board_states)
    
    def __getitem__(self, idx):
        return (
            self.board_states[idx],
            self.policy_targets[idx],
            self.value_targets[idx]
        )


class ChessTrainer:
    """
    Trainer for the chess neural network.
    Implements combined policy and value loss with regularization.
    """
    
    def __init__(self, model, device='cpu', learning_rate=0.01):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer,
        mode='min',
        factor=0.5,
        patience=5
        )

        
        
        # Loss weights
        self.policy_weight = 1.0
        self.value_weight = 1.0
        
        # Training history
        self.history = {
            'epoch': [],
            'train_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'learning_rate': []
        }
        
    def compute_loss(self, batch):
        """
        Compute combined loss for policy and value predictions.
        
        Args:
            batch: Tuple of (board_states, policy_targets, value_targets)
            
        Returns:
            total_loss, policy_loss, value_loss
        """
        board_states, policy_targets, value_targets = batch
        
        # Move to device
        board_states = board_states.to(self.device)
        policy_targets = policy_targets.to(self.device)
        value_targets = value_targets.to(self.device)
        
        # Forward pass
        policy_logits, value_pred = self.model(board_states)
        
        # Policy loss (cross-entropy with soft targets)
        policy_loss = -torch.sum(policy_targets * torch.log_softmax(policy_logits, dim=1), dim=1)
        policy_loss = policy_loss.mean()
        
        # Value loss (mean squared error)
        value_loss = F.mse_loss(value_pred, value_targets)
        
        # Combined loss
        total_loss = self.policy_weight * policy_loss + self.value_weight * value_loss
        
        return total_loss, policy_loss, value_loss
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # Compute loss
            total_loss, policy_loss, value_loss = self.compute_loss(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            epoch_loss += total_loss.item()
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            num_batches += 1
        
        # Average losses
        avg_loss = epoch_loss / num_batches
        avg_policy_loss = epoch_policy_loss / num_batches
        avg_value_loss = epoch_value_loss / num_batches
        
        return avg_loss, avg_policy_loss, avg_value_loss
    
    def train(self, training_data: Dict, num_epochs: int, batch_size: int = 256):
        """
        Train the model on generated self-play data.
        
        Args:
            training_data: Dictionary with board_states, policy_targets, value_targets
            num_epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Create dataset and dataloader
        dataset = ChessDataset(
            training_data['board_states'],
            training_data['policy_targets'],
            training_data['value_targets']
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device == 'cuda')
        )
        
        print(f"\nTraining for {num_epochs} epochs...")
        print(f"Dataset size: {len(dataset)} positions")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train for one epoch
            avg_loss, avg_policy_loss, avg_value_loss = self.train_epoch(dataloader)
            
            # Update learning rate
            self.scheduler.step(avg_loss)
            
            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(avg_loss)
            self.history['policy_loss'].append(avg_policy_loss)
            self.history['value_loss'].append(avg_value_loss)
            self.history['learning_rate'].append(current_lr)
            
            # Print progress
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Policy: {avg_policy_loss:.4f} | "
                  f"Value: {avg_value_loss:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {elapsed:.2f}s")
    
    def save_checkpoint(self, path: str, epoch: int):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from {path} (epoch {epoch})")
        return epoch
    
    def save_history(self, path: str):
        """Save training history to JSON"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


# Full training pipeline
def train_chess_model(
    model,
    encoder,
    num_iterations=10,
    games_per_iteration=100,
    epochs_per_iteration=10,
    batch_size=256,
    device='cpu'
):
    """
    Complete training pipeline with iterative self-play.
    
    Args:
        model: ChessNet model
        encoder: BoardEncoder
        num_iterations: Number of self-play iterations
        games_per_iteration: Games to generate per iteration
        epochs_per_iteration: Training epochs per iteration
        batch_size: Training batch size
        device: Training device
    """
    from chess_self_play import SelfPlayGenerator, ReplayBuffer
    
    # Initialize components
    generator = SelfPlayGenerator(model, encoder, device)
    trainer = ChessTrainer(model, device)
    replay_buffer = ReplayBuffer(max_size=100000)
    
    print("="*60)
    print("Chess Neural Network Training Pipeline")
    print("="*60)
    print(f"Iterations: {num_iterations}")
    print(f"Games per iteration: {games_per_iteration}")
    print(f"Epochs per iteration: {epochs_per_iteration}")
    print(f"Device: {device}")
    print("="*60)
    
    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")
        
        # Generate self-play games
        print("\n[1/3] Generating self-play games...")
        training_data = generator.generate_training_batch(
            num_games=games_per_iteration,
            temperature=1.0  # Exploration
        )
        
        print(f"\nGame results: {training_data['results']}")
        print(f"Positions generated: {training_data['num_positions']}")
        
        # Train on generated data
        print(f"\n[2/3] Training model...")
        trainer.train(training_data, epochs_per_iteration, batch_size)
        
        # Save checkpoint
        print(f"\n[3/3] Saving checkpoint...")
        checkpoint_path = f"checkpoints/chess_model_iter_{iteration+1}.pt"
        trainer.save_checkpoint(checkpoint_path, iteration + 1)
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    return model, trainer


# Example usage
if __name__ == "__main__":
    from src.network.chess_net import ChessNet
    from src.core.board_encoder import BoardEncoder

    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model = ChessNet(num_filters=128, num_residual_blocks=5)
    encoder = BoardEncoder()
    
    # Train model (small test run)
    trained_model, trainer = train_chess_model(
        model=model,
        encoder=encoder,
        num_iterations=2,
        games_per_iteration=10,
        epochs_per_iteration=5,
        batch_size=64,
        device=device
    )
    
    # Save final model
    trainer.save_checkpoint("checkpoints/chess_model_final.pt", epoch='final')
    trainer.save_history("checkpoints/training_history.json")