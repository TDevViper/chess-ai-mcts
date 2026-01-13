# Lichess Bot
"""
Lichess Bot Integration
Connects the neural network engine to Lichess.org to play real humans
"""

import chess
import requests
import json
import time
from typing import Optional, Dict
from src.engine.chess_uci_engine import SimpleEngine
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LichessBot:
    """
    Lichess bot that connects the neural network engine to Lichess.org.
    Uses the Lichess Bot API to accept challenges and play games.
    """
    
    def __init__(self, api_token: str, model_path: str, device='cpu'):
        """
        Initialize the Lichess bot.
        
        Args:
            api_token: Lichess API token (get from lichess.org/account/oauth/token)
            model_path: Path to trained model checkpoint
            device: Device for model inference
        """
        self.api_token = api_token
        self.base_url = "https://lichess.org"
        self.headers = {
            "Authorization": f"Bearer {api_token}"
        }
        
        # Initialize engine
        logger.info("Loading chess engine...")
        self.engine = SimpleEngine(model_path, device)
        logger.info("Engine loaded successfully")
        
        # Bot configuration
        self.accept_challenges = True
        self.playing_games = {}
        
    def get_account_info(self) -> Dict:
        """Get bot account information"""
        response = requests.get(
            f"{self.base_url}/api/account",
            headers=self.headers
        )
        return response.json()
    
    def upgrade_to_bot(self):
        """Upgrade account to bot account (one-time operation)"""
        response = requests.post(
            f"{self.base_url}/api/bot/account/upgrade",
            headers=self.headers
        )
        if response.status_code == 200:
            logger.info("Successfully upgraded to bot account")
        else:
            logger.error(f"Failed to upgrade: {response.text}")
    
    def stream_events(self):
        """
        Stream incoming events (challenges, game starts, etc.)
        This is the main event loop for the bot.
        """
        logger.info("Starting event stream...")
        
        response = requests.get(
            f"{self.base_url}/api/stream/event",
            headers=self.headers,
            stream=True,
            timeout=None
        )
        
        for line in response.iter_lines():
            if line:
                try:
                    event = json.loads(line.decode('utf-8'))
                    self.handle_event(event)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse event: {line}")
                except Exception as e:
                    logger.error(f"Error handling event: {e}")
    
    def handle_event(self, event: Dict):
        """Handle incoming events from Lichess"""
        event_type = event.get('type')
        
        if event_type == 'challenge':
            self.handle_challenge(event['challenge'])
        elif event_type == 'gameStart':
            self.handle_game_start(event['game'])
        elif event_type == 'gameFinish':
            self.handle_game_finish(event['game'])
        else:
            logger.debug(f"Unhandled event type: {event_type}")
    
    def handle_challenge(self, challenge: Dict):
        """Handle incoming challenge"""
        challenge_id = challenge['id']
        challenger = challenge['challenger']['name']
        variant = challenge.get('variant', {}).get('key', 'standard')
        time_control = challenge.get('timeControl', {})
        
        logger.info(f"Challenge received from {challenger}")
        logger.info(f"Variant: {variant}, Time control: {time_control}")
        
        # Accept only standard chess for now
        if self.accept_challenges and variant == 'standard':
            self.accept_challenge(challenge_id)
        else:
            self.decline_challenge(challenge_id)
    
    def accept_challenge(self, challenge_id: str):
        """Accept a challenge"""
        response = requests.post(
            f"{self.base_url}/api/challenge/{challenge_id}/accept",
            headers=self.headers
        )
        
        if response.status_code == 200:
            logger.info(f"Accepted challenge {challenge_id}")
        else:
            logger.error(f"Failed to accept challenge: {response.text}")
    
    def decline_challenge(self, challenge_id: str, reason: str = "generic"):
        """Decline a challenge"""
        response = requests.post(
            f"{self.base_url}/api/challenge/{challenge_id}/decline",
            headers=self.headers,
            json={"reason": reason}
        )
        
        if response.status_code == 200:
            logger.info(f"Declined challenge {challenge_id}")
    
    def handle_game_start(self, game: Dict):
        """Handle game start"""
        game_id = game['id']
        logger.info(f"Game started: {game_id}")
        
        # Start playing the game in a separate thread/process
        # For simplicity, we'll play it synchronously here
        self.play_game(game_id)
    
    def handle_game_finish(self, game: Dict):
        """Handle game finish"""
        game_id = game['id']
        logger.info(f"Game finished: {game_id}")
        
        if game_id in self.playing_games:
            del self.playing_games[game_id]
    
    def stream_game_state(self, game_id: str):
        """
        Stream game state updates for a specific game
        """
        response = requests.get(
            f"{self.base_url}/api/bot/game/stream/{game_id}",
            headers=self.headers,
            stream=True,
            timeout=None
        )
        
        for line in response.iter_lines():
            if line:
                try:
                    event = json.loads(line.decode('utf-8'))
                    yield event
                except json.JSONDecodeError:
                    continue
    
    def play_game(self, game_id: str):
        """
        Play a complete game
        """
        logger.info(f"Starting to play game {game_id}")
        
        board = chess.Board()
        self.playing_games[game_id] = True
        
        try:
            for event in self.stream_game_state(game_id):
                event_type = event.get('type')
                
                if event_type == 'gameFull':
                    # Initial game state
                    self.handle_game_full(game_id, event, board)
                    
                elif event_type == 'gameState':
                    # Game state update
                    self.handle_game_state(game_id, event, board)
                    
                elif event_type == 'chatLine':
                    # Chat message
                    pass
                    
        except Exception as e:
            logger.error(f"Error playing game {game_id}: {e}")
        finally:
            if game_id in self.playing_games:
                del self.playing_games[game_id]
    
    def handle_game_full(self, game_id: str, event: Dict, board: chess.Board):
        """Handle full game state"""
        state = event.get('state', {})
        initial_fen = event.get('initialFen', 'startpos')
        
        # Set up board
        if initial_fen != 'startpos':
            board.set_fen(initial_fen)
        
        # Get our color
        white_player = event.get('white', {})
        black_player = event.get('black', {})
        
        account = self.get_account_info()
        our_username = account['username']
        
        our_color = None
        if white_player.get('name') == our_username:
            our_color = chess.WHITE
        elif black_player.get('name') == our_username:
            our_color = chess.BLACK
        
        logger.info(f"Playing as {'White' if our_color == chess.WHITE else 'Black'}")
        
        # Apply moves
        moves = state.get('moves', '').split()
        for move_uci in moves:
            if move_uci:
                board.push(chess.Move.from_uci(move_uci))
        
        # Make move if it's our turn
        if board.turn == our_color and not board.is_game_over():
            self.make_move(game_id, board)
    
    def handle_game_state(self, game_id: str, event: Dict, board: chess.Board):
        """Handle game state update"""
        moves = event.get('moves', '').split()
        
        # Reset board and apply all moves
        board.reset()
        for move_uci in moves:
            if move_uci:
                board.push(chess.Move.from_uci(move_uci))
        
        # Check if it's our turn
        # (We need to know our color from the full game state)
        if not board.is_game_over():
            self.make_move(game_id, board)
    
    def make_move(self, game_id: str, board: chess.Board):
        """
        Calculate and make a move
        """
        logger.info(f"Thinking... (position: {board.fen()})")
        
        try:
            # Get move from engine
            move = self.engine.get_move(board)
            
            logger.info(f"Playing move: {move.uci()}")
            
            # Send move to Lichess
            response = requests.post(
                f"{self.base_url}/api/bot/game/{game_id}/move/{move.uci()}",
                headers=self.headers
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to make move: {response.text}")
                
        except Exception as e:
            logger.error(f"Error making move: {e}")
    
    def run(self):
        """Start the bot"""
        logger.info("="*60)
        logger.info("Chess Neural Network Bot - Lichess Integration")
        logger.info("="*60)
        
        # Get account info
        account = self.get_account_info()
        logger.info(f"Logged in as: {account['username']}")
        logger.info(f"Bot account: {account.get('title') == 'BOT'}")
        
        # Start event stream
        try:
            self.stream_events()
        except KeyboardInterrupt:
            logger.info("\nBot stopped by user")
        except Exception as e:
            logger.error(f"Bot crashed: {e}")


# Configuration and main entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Lichess Chess Bot')
    parser.add_argument('--token', required=True, help='Lichess API token')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Create and run bot
    bot = LichessBot(
        api_token=args.token,
        model_path=args.model,
        device=args.device
    )
    
    bot.run()