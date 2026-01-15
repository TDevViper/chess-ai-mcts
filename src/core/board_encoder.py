"""
Chess Board State Encoder
Converts chess positions into neural network input tensors
"""

import chess
import torch
import numpy as np

class BoardEncoder:
    """
    Encodes chess board state into a tensor representation for neural network input.
    Uses piece-centric bitboard planes (12 planes for pieces + additional metadata).
    """
    
    # Piece type mapping
    PIECE_TO_INDEX = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    
    def __init__(self):
        self.input_planes = 18  # 12 piece planes + 6 metadata planes
        
    def encode_board(self, board: chess.Board) -> torch.Tensor:
        """
        Encode a chess board into a tensor of shape (18, 8, 8).
        
        Planes:
        0-5:   White pieces (P, N, B, R, Q, K)
        6-11:  Black pieces (P, N, B, R, Q, K)
        12:    Turn indicator (1 if white to move, 0 otherwise)
        13:    White castling kingside
        14:    White castling queenside
        15:    Black castling kingside
        16:    Black castling queenside
        17:    En passant square
        
        Args:
            board: python-chess Board object
            
        Returns:
            Tensor of shape (18, 8, 8)
        """
        state = np.zeros((self.input_planes, 8, 8), dtype=np.float32)
        
        # Encode piece positions (planes 0-11)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                
                # Determine plane index
                piece_idx = self.PIECE_TO_INDEX[piece.piece_type]
                plane_idx = piece_idx if piece.color == chess.WHITE else piece_idx + 6
                
                state[plane_idx, rank, file] = 1.0
        
        # Turn indicator (plane 12)
        if board.turn == chess.WHITE:
            state[12, :, :] = 1.0
        
        # Castling rights (planes 13-16)
        if board.has_kingside_castling_rights(chess.WHITE):
            state[13, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            state[14, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            state[15, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            state[16, :, :] = 1.0
        
        # En passant square (plane 17)
        if board.ep_square is not None:
            rank, file = divmod(board.ep_square, 8)
            state[17, rank, file] = 1.0
        
        return torch.from_numpy(state)
    
    def encode_move(self, move: chess.Move) -> int:
        """
        Encode a move into a single integer for policy output.
        Uses from_square * 64 + to_square encoding (4096 possible moves).
        
        Args:
            move: python-chess Move object
            
        Returns:
            Integer move encoding
        """
        return move.from_square * 64 + move.to_square
    
    def decode_move(self, move_idx: int, board: chess.Board) -> chess.Move:
        """
        Decode a move index back into a chess.Move object.
        Handles promotions by choosing queen promotion by default.
        
        Args:
            move_idx: Integer move encoding
            board: Current board state
            
        Returns:
            chess.Move object
        """
        from_square = move_idx // 64
        to_square = move_idx % 64
        
        # Check if this is a pawn promotion
        piece = board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = to_square // 8
            if (piece.color == chess.WHITE and to_rank == 7) or \
               (piece.color == chess.BLACK and to_rank == 0):
                return chess.Move(from_square, to_square, promotion=chess.QUEEN)
        
        return chess.Move(from_square, to_square)
    
    def get_legal_move_mask(self, board: chess.Board) -> torch.Tensor:
        """
        Create a mask of legal moves for the current position.
        
        Args:
            board: Current board state
            
        Returns:
            Tensor of shape (4096,) with 1.0 for legal moves, 0.0 otherwise
        """
        mask = torch.zeros(4096, dtype=torch.float32)
        
        for move in board.legal_moves:
            move_idx = self.encode_move(move,board)
            mask[move_idx] = 1.0
        
        return mask
    def encode_move(self, move, board):
        """
        Encode a chess.Move into index [0, 4095]
        """
        from_sq = move.from_square
        to_sq = move.to_square
        return from_sq * 64 + to_sq




# Example usage
if __name__ == "__main__":
    encoder = BoardEncoder()
    
    # Test with starting position
    board = chess.Board()
    
    # Encode board
    state_tensor = encoder.encode_board(board)
    print(f"Board state shape: {state_tensor.shape}")
    print(f"Non-zero elements: {(state_tensor != 0).sum().item()}")
    
    # Get legal moves mask
    legal_mask = encoder.get_legal_move_mask(board)
    print(f"Number of legal moves: {legal_mask.sum().item()}")
    
    # Test move encoding/decoding
    test_move = chess.Move.from_uci("e2e4")
    encoded = encoder.encode_move(test_move)
    decoded = encoder.decode_move(encoded, board)
    print(f"\nMove encoding test:")
    print(f"Original: {test_move}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
