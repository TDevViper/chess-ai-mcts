"""
Visualize Chess AI Playing
Watch your neural network play chess with a web-based interface
"""

import chess
import torch
import json
import time
import argparse
from pathlib import Path
import webbrowser
import http.server
import socketserver
import threading

# Add parent directory to path to import modules
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.engine.chess_uci_engine import SimpleEngine



class ChessVisualizer:
    """
    Creates a web interface to watch your AI play chess
    """
    
    def __init__(self, model_path: str, device='cpu'):
        print("Loading chess engine...")
        self.engine = SimpleEngine(model_path, device)
        print("Engine loaded!")
        
    def play_game_with_visualization(self, opponent='self', max_moves=100):
        """
        Play a game and generate move-by-move data for visualization
        """
        board = chess.Board()
        game_data = {
            'moves': [],
            'positions': [],
            'evaluations': [],
            'thinking_times': []
        }
        
        print("\n" + "="*60)
        print("Playing game...")
        print("="*60 + "\n")
        
        move_count = 0
        
        while not board.is_game_over() and move_count < max_moves:
            print(f"Move {move_count + 1}: ", end='', flush=True)
            
            start_time = time.time()
            
            # Decide who plays
            if opponent == 'self':
                # AI vs AI
                move = self.engine.get_move(board)
                player = "White AI" if board.turn == chess.WHITE else "Black AI"
            elif opponent == 'random':
                if board.turn == chess.WHITE:
                    # AI plays white
                    move = self.engine.get_move(board)
                    player = "AI"
                else:
                    # Random plays black
                    import random
                    move = random.choice(list(board.legal_moves))
                    player = "Random"
            
            thinking_time = time.time() - start_time
            
            # Get evaluation
            board_tensor = self.engine.encoder.encode_board(board).to(self.engine.device)
            legal_mask = self.engine.encoder.get_legal_move_mask(board).to(self.engine.device)
            
            with torch.no_grad():
                policy_logits, value = self.engine.model(board_tensor.unsqueeze(0))
                evaluation = value.item()
            
            # Store move data
            game_data['moves'].append({
                'move': move.uci(),
                'san': board.san(move),
                'player': player,
                'move_number': move_count + 1,
                'turn': 'white' if board.turn == chess.WHITE else 'black'
            })
            game_data['positions'].append(board.fen())
            game_data['evaluations'].append(round(evaluation, 3))
            game_data['thinking_times'].append(round(thinking_time, 3))
            
            print(f"{player} plays {move.uci()} (eval: {evaluation:+.2f}, time: {thinking_time:.2f}s)")
            
            board.push(move)
            move_count += 1
        
        # Game over
        game_data['result'] = board.result()
        game_data['termination'] = self._get_termination_reason(board)
        
        print("\n" + "="*60)
        print(f"Game Over: {game_data['result']}")
        print(f"Reason: {game_data['termination']}")
        print(f"Total moves: {move_count}")
        print("="*60 + "\n")
        
        return game_data
    
    def _get_termination_reason(self, board):
        """Get the reason the game ended"""
        if board.is_checkmate():
            return "Checkmate"
        elif board.is_stalemate():
            return "Stalemate"
        elif board.is_insufficient_material():
            return "Insufficient material"
        elif board.can_claim_fifty_moves():
            return "Fifty-move rule"
        elif board.can_claim_threefold_repetition():
            return "Threefold repetition"
        else:
            return "Max moves reached"
    
    def generate_html(self, game_data, output_file='chess_visualization.html'):
        """Generate interactive HTML file with the game"""
        
        html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chess AI Visualization</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .game-layout {
            display: grid;
            grid-template-columns: 1fr 600px;
            gap: 30px;
        }
        .board-section {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .chessboard {
            display: grid;
            grid-template-columns: repeat(8, 70px);
            grid-template-rows: repeat(8, 70px);
            border: 3px solid #333;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        .square {
            width: 70px;
            height: 70px;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 45px;
            position: relative;
        }
        .square.light { background-color: #f0d9b5; }
        .square.dark { background-color: #b58863; }
        .square.highlight { background-color: #baca44 !important; }
        
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            font-weight: bold;
            transition: all 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .info-panel {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
        }
        .status-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .stats {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        .stat-label {
            color: #666;
            font-weight: 500;
        }
        .stat-value {
            font-weight: bold;
            color: #333;
        }
        
        .eval-bar {
            height: 30px;
            background: #333;
            border-radius: 5px;
            overflow: hidden;
            margin: 10px 0;
            position: relative;
        }
        .eval-fill {
            height: 100%;
            background: white;
            transition: width 0.3s;
        }
        .eval-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #888;
            font-weight: bold;
            font-size: 0.9em;
        }
        
        .moves-list {
            background: white;
            padding: 20px;
            border-radius: 10px;
            max-height: 400px;
            overflow-y: auto;
        }
        .move-item {
            padding: 10px;
            margin-bottom: 5px;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .move-item:hover {
            background: #f0f0f0;
        }
        .move-item.active {
            background: #667eea;
            color: white;
        }
        
        .result-banner {
            background: #4CAF50;
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 1.3em;
            font-weight: bold;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Chess Neural Network Visualization</h1>
        <p class="subtitle">Watch your AI play move by move</p>
        
        <div class="game-layout">
            <div class="board-section">
                <div id="chessboard" class="chessboard"></div>
                <div class="controls">
                    <button onclick="firstMove()">⏮️ First</button>
                    <button onclick="prevMove()">⏪ Previous</button>
                    <button onclick="nextMove()">⏩ Next</button>
                    <button onclick="lastMove()">⏭️ Last</button>
                    <button onclick="autoPlay()">▶️ Auto Play</button>
                </div>
            </div>
            
            <div class="info-panel">
                <div class="status-box">
                    <div style="font-size: 1.5em; margin-bottom: 10px;" id="statusText">
                        Move 0
                    </div>
                    <div id="turnText">White to move</div>
                </div>
                
                <div class="stats">
                    <div class="stat-row">
                        <span class="stat-label">Current Move:</span>
                        <span class="stat-value" id="currentMove">-</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Evaluation:</span>
                        <span class="stat-value" id="evaluation">0.00</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Thinking Time:</span>
                        <span class="stat-value" id="thinkTime">-</span>
                    </div>
                    
                    <div style="margin-top: 15px;">
                        <strong>Evaluation Bar:</strong>
                        <div class="eval-bar">
                            <div class="eval-fill" id="evalFill"></div>
                            <div class="eval-text" id="evalText">0.00</div>
                        </div>
                    </div>
                </div>
                
                <div class="moves-list">
                    <h3 style="margin-bottom: 15px;">📜 Move History</h3>
                    <div id="movesList"></div>
                </div>
                
                <div class="result-banner" id="resultBanner" style="display: none;"></div>
            </div>
        </div>
    </div>

    <script>
        const gameData = ''' + json.dumps(game_data) + ''';
        
        const pieces = {
            'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
            'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
        };
        
        let currentPosition = 0;
        let autoPlaying = false;
        
        function fenToBoard(fen) {
            const board = [];
            const rows = fen.split(' ')[0].split('/');
            
            for (const row of rows) {
                const boardRow = [];
                for (const char of row) {
                    if (char >= '1' && char <= '8') {
                        for (let i = 0; i < parseInt(char); i++) {
                            boardRow.push('');
                        }
                    } else {
                        boardRow.push(char);
                    }
                }
                board.push(boardRow);
            }
            return board;
        }
        
        function renderPosition(position) {
            const fen = position === 0 ? 
                'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' : 
                gameData.positions[position - 1];
            
            const board = fenToBoard(fen);
            const boardElement = document.getElementById('chessboard');
            boardElement.innerHTML = '';
            
            for (let row = 0; row < 8; row++) {
                for (let col = 0; col < 8; col++) {
                    const square = document.createElement('div');
                    square.className = 'square';
                    square.className += (row + col) % 2 === 0 ? ' light' : ' dark';
                    
                    // Highlight last move
                    if (position > 0) {
                        const lastMove = gameData.moves[position - 1];
                        const move = lastMove.move;
                        const from = [8 - parseInt(move[1]), move.charCodeAt(0) - 97];
                        const to = [8 - parseInt(move[3]), move.charCodeAt(2) - 97];
                        
                        if ((row === from[0] && col === from[1]) || 
                            (row === to[0] && col === to[1])) {
                            square.classList.add('highlight');
                        }
                    }
                    
                    const piece = board[row][col];
                    if (piece) {
                        square.textContent = pieces[piece];
                    }
                    
                    boardElement.appendChild(square);
                }
            }
            
            updateInfo(position);
        }
        
        function updateInfo(position) {
            if (position === 0) {
                document.getElementById('statusText').textContent = 'Starting Position';
                document.getElementById('turnText').textContent = 'White to move';
                document.getElementById('currentMove').textContent = '-';
                document.getElementById('evaluation').textContent = '0.00';
                document.getElementById('thinkTime').textContent = '-';
                updateEvalBar(0);
            } else {
                const move = gameData.moves[position - 1];
                const eval = gameData.evaluations[position - 1];
                const time = gameData.thinking_times[position - 1];
                
                document.getElementById('statusText').textContent = 
                    `Move ${move.move_number}`;
                document.getElementById('turnText').textContent = 
                    `${move.player} played ${move.san}`;
                document.getElementById('currentMove').textContent = move.san;
                document.getElementById('evaluation').textContent = eval.toFixed(2);
                document.getElementById('thinkTime').textContent = `${time}s`;
                updateEvalBar(eval);
            }
            
            // Update moves list
            const movesList = document.getElementById('movesList');
            movesList.innerHTML = '';
            gameData.moves.forEach((move, i) => {
                const moveItem = document.createElement('div');
                moveItem.className = 'move-item';
                if (i + 1 === position) moveItem.classList.add('active');
                moveItem.textContent = 
                    `${move.move_number}. ${move.player}: ${move.san} (${gameData.evaluations[i].toFixed(2)})`;
                moveItem.onclick = () => goToMove(i + 1);
                movesList.appendChild(moveItem);
            });
            
            // Show result if at end
            if (position === gameData.moves.length) {
                document.getElementById('resultBanner').style.display = 'block';
                document.getElementById('resultBanner').textContent = 
                    `🏆 Game Over: ${gameData.result} - ${gameData.termination}`;
            } else {
                document.getElementById('resultBanner').style.display = 'none';
            }
        }
        
        function updateEvalBar(eval) {
            const normalized = Math.max(-1, Math.min(1, eval));
            const percentage = ((normalized + 1) / 2) * 100;
            document.getElementById('evalFill').style.width = percentage + '%';
            document.getElementById('evalText').textContent = eval.toFixed(2);
        }
        
        function firstMove() {
            currentPosition = 0;
            renderPosition(currentPosition);
        }
        
        function prevMove() {
            if (currentPosition > 0) {
                currentPosition--;
                renderPosition(currentPosition);
            }
        }
        
        function nextMove() {
            if (currentPosition < gameData.moves.length) {
                currentPosition++;
                renderPosition(currentPosition);
            }
        }
        
        function lastMove() {
            currentPosition = gameData.moves.length;
            renderPosition(currentPosition);
        }
        
        function goToMove(position) {
            currentPosition = position;
            renderPosition(currentPosition);
        }
        
        function autoPlay() {
            if (autoPlaying) {
                autoPlaying = false;
                return;
            }
            
            autoPlaying = true;
            const playNext = () => {
                if (autoPlaying && currentPosition < gameData.moves.length) {
                    nextMove();
                    setTimeout(playNext, 1500);
                } else {
                    autoPlaying = false;
                }
            };
            playNext();
        }
        
        // Initialize
        renderPosition(0);
    </script>
</body>
</html>'''
        
        # Write HTML file
        output_path = Path(output_file)
        output_path.write_text(html_template)
        print(f"✅ Visualization saved to: {output_path.absolute()}")
        
        return str(output_path.absolute())


def main():
    parser = argparse.ArgumentParser(description='Visualize chess AI playing')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--opponent', default='self', choices=['self', 'random'],
                       help='Opponent type')
    parser.add_argument('--output', default='chess_visualization.html',
                       help='Output HTML file')
    parser.add_argument('--open', action='store_true',
                       help='Automatically open in browser')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Chess AI Visualizer")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Opponent: {args.opponent}")
    print("="*60)
    
    # Create visualizer
    viz = ChessVisualizer(args.model, args.device)
    
    # Play game
    game_data = viz.play_game_with_visualization(args.opponent)
    
    # Generate HTML
    html_path = viz.generate_html(game_data, args.output)
    
    print("\n✨ Done! Open the HTML file in your browser to watch the game.")
    
    # Open in browser if requested
    if args.open:
        print("Opening in browser...")
        webbrowser.open('file://' + html_path)


if __name__ == "__main__":
    main()