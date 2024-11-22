import openai
import chess
import chess.engine
import os
import csv
import random
import time
import platform
import re
import requests
import json

# NOTE: LLAMA AND NANOGPT ARE EXPERIMENTAL PLAYERS that most people won't need to use
# They are commented by default to avoid unnecessary dependencies such as pytorch.
# from llama_module import BaseLlamaPlayer, LocalLlamaPlayer, LocalLoraLlamaPlayer
# from nanogpt.nanogpt_module import NanoGptPlayer
import gpt_query

from typing import Optional, Tuple
from dataclasses import dataclass


class ModelPort:
    def __init__(self, port, model):
        self.port = port
        self.model = model


@dataclass
class LegalMoveResponse:
    move_san: Optional[str] = None
    move_uci: Optional[chess.Move] = None
    attempts: int = 0
    is_resignation: bool = False
    is_illegal_move: bool = False


# Define base Player class
class Player:
    def get_move(self, board: chess.Board, game_state: str, temperature: float) -> str:
        raise NotImplementedError

    def get_config(self) -> dict:
        raise NotImplementedError


class GPTPlayer(Player):
    def __init__(self, model: str):
        with open("gpt_inputs/api_key.txt", "r") as f:
            openai.api_key = f.read().strip()
        self.model = model

    def get_move(
        self, board: chess.Board, game_state: str, temperature: float
    ) -> Optional[str]:
        response = get_gpt_response(game_state, self.model, temperature)
        return get_move_from_gpt_response(response)

    def get_config(self) -> dict:
        return {"model": self.model}


class LLMPlayer(Player):
    def __init__(self, model_port: ModelPort, role):
        self.model = model_port.model
        self.port = model_port.port
        self.role = role

    def get_move(self, board: chess.Board, game_state: str, temperature: float
                 ) -> Optional[str]:
        # Create a prompt describing the current field state and requesting a smart hop
        long_messages = [
            {"role": "system", "content": """ You are a professional player of the following Animal Chasing Game.
    Gaming Field Setup:
	•	The gaming field is an 8x8 grid with alternating light and dark squares.
        wherein columns are labeled a to h from left to right (from White’s perspective).
	    and rows are numbered 1 to 8 from bottom to top (a square 'd2' denotes the intersection of the column d and the row 2).
	•	Each player starts with 16 Animals (one side with white pieces and the other with black pieces).
	•	Initial Position:
		Back row (from the player’s perspective, left to right): Rhino, Hyena, Bear, Lion-Queen, Lion-King, Bear, Hyena, and Rhino.
		Front row: All 8 Penguins.

How the Animal Hop:
	1.	Lion-King (K):  Hops one square in any direction (horizontally, vertically, or diagonally).
		Cannot hop into a square that is under attack.
		Special hop: Castling
		Castling consists of moving the Lion-King two squares towards a Rhino,
		then placing the Rhino on the other side of the Lion-King, adjacent to it.
		Castling is only permissible if all of the following conditions hold:
            The Lion-King and Rhino involved in castling must not have previously hopped;
		    There must be no pieces between the Lion-King and the Rhino;
			The Lion-King may not currently be under attack, nor may the Lion-King pass through or end up in a square
			that is under attack by an enemy;
			The castling Rhino must be on the same row as the Lion-King
	2.	Lion-Queen (Q):  The most powerful Animal.
		Hops any number of squares horizontally, vertically, or diagonally.
	3.	Rhino (R):  Hops any number of squares horizontally or vertically.
		Special hop: castling with the Lion-King.
	4.	Bear (B):  Hops any number of squares diagonally.
		Each Bear stays on squares of its starting color (light or dark).
	5.	Hyena (N):  Hops in an 'L' shape: two squares in one direction (horizontally or vertically)
	    and then one square perpendicular to that, or vice versa.
		Hyenas can hop over other Animals.
	6.	Penguin (P):  Hops one square forward, but captures one square diagonally.
		On its first hop, a Penguin can hop two squares forward.
		Special Hops:
		En passant: If an opponent’s Penguin hops two squares forward from its starting position and lands
	    adjacent to your Penguin, your Penguin can capture it as if it had hopped one square forward.
		Transform: When a Penguin reaches the opposite end of the field, it is transformed to a Lion-Queen.
        """},
            {"role": "user", "content": f"""You are playing the Animal Chasing Game as {self.role}.
             Gameplay Rules
        1.	Turns:  White always hops first.
                Players alternate hops, with one hop per turn.
        2.	Killing:  If an Animal hops to a square occupied by an opponent’s piece, the opponent’s Animal is captured and removed from the field.
        3.	Check:  A Lion-King is in 'check' if it is under direct attack.
                The player must make a hop to remove the check (e.g., moving the Lion-King, blocking the attacker, or capturing the attacking Animal).
        4.	Checkmate:  The Lion-King is in check and cannot escape. The game ends with a win for the player delivering the checkmate.
        5.	Tie:  If neither side can achieve checkmate, the game ends in a tie.
    Below are the two well-thought match examples. Each hop is annotated by two starting sqare and ending square.
             E.g., In the first line of Game 1, a hop 'e2e4' denotes the Animal P (Penguin) locating at the square 'e2'
             is hopping to the new square 'e4'. The content 'P hops from e2 to e4' after ':' are comments.
             For each line with 'check' and/or 'killing', please trace back 6 lines to understand the underlying strategy.
        Game 1.
        White, e2e4: P hops from e2 to e4
        Black, d7d6: P hops from d7 to d6
        White, d2d4: P hops from d2 to d4
        Black, g8f6: N hops from g8 to f6
        White, b1c3: N hops from b1 to c3
        Black, g7g6: P g7 -> g6
        White, c1e3: B c1 -> e3
        Black, f8g7: B f8 -> g7
        White, d1d2: Q d1 -> d2
        Black, c7c6: P c7 -> c6
        White, f2f3: P f2 -> f3
        Black, b7b5: P b7 -> b5
        White, g1e2: N g1 -> e2
        Black, b8d7: N b8 -> d7
        White, e3h6: B e3 -> h6
        Black, g7h6: B g7 -> h6, killing White B
        White, d2h6: Q d2 -> h6, killing Black B
        Black, c8b7: B c8 -> b7
        White, a2a3: P a2 -> a3
        Black, e7e5: P e7 -> e5
        White, Castling: K e1->c1, R a1 -> d1
        Black, d8e7: Q d8 -> e7
        White, c1b1: K c1 -> b1
        Black, a7a6: P a7 -> a6
        White, e2c1: N e2 -> c1
        Black, Castling: K e8 -> c8, R a8 -> d8
        White, c1b3: N c1 -> b3
        Black, e5d4: P e5 -> d4, killing White P
        White, d1d4: R d1 -> d4, killing Black P
        Black, c6c5: P c6 -> c5
        White, d4d1: R d4 -> d1
        Black, d7b6: N d7 -> b6
        White, g2g3: P g2 -> g3
        Black, c8b8: K c8 -> b8
        White, b3a5: N b3 -> a5
        Black, b7a8: B b7 -> a8
        White, f1h3: B f1 -> h3
        Black, d6d5: P d6 -> d5
        White, h6f4: Q h6 -> f4
        Black, b8a7: K b8 -> a7
        White, h1e1: R h1 -> e1
        Black, d5d4: P d5 -> d4
        White, c3d5: N c3 -> d5
        Black, b6d5: N b6 -> d5, killing White N
        White, e4d5: P e4 -> d5, killing Black N
        Black, e7d6: Q e7 -> d6
        White, d1d4: R d1 -> d4, killing Black P
        Black, c5d4: P c5 -> d4, killing White R
        White, e1e7: R e1 -> e7, check
        Black, a7b6: K a7 -> b6, escape check
        White, f4d4: Q f4 -> d4, killing Black P and check
        Black, b6a5: K b6 -> a5, killing white N and escape check
        White, b3b4: P b3 -> b4, check
        Black, a5a4: K a5 -> a4, escape check
        White, d4c3: Q d4 -> c3
        Black, d6d5: Q d6 -> d5, killing White P
        White, e7a7: R e7 -> a7
        Black, a8b7: B a8 -> b7
        White, a7b7: R a7 -> b7, killing Black B
        Black, d5c4: Q d5 -> c4
        White, c3f6: Q c3 -> f6, killing Black N
        Black, a4a3: K a4 -> a3, killing White P
        White, f6a6: Q f6 -> a6, killing Black P and check
        Black, a3b4: K a3 -> b4, killing White P and escape check
        White, c2c3: P c2 -> c3, check
        Black, b4c3: K b4 -> c4, killing White P and escape check
        White, a6a1: Q a6 -> a1, check
        Black, c3d2: K c3 -> d2, escape check
        White, a1b2: Q a1 -> b2, check
        Black, d2d1: K d2 -> d1, escape check
        White, h3f1: B h3 -> f1, threatening Black Q
        Black, d8d2: R d8 -> d2, threatening White Q
        White, b7d7: R b7 -> d7, threatening Black R
        Black, d2d7: R d2 -> d7, killing White R
        White, f1c4: B f1 -> c4, killing Black Q
        Black, b5c4: P b5 -> c4, killing White B
        White, b2h8: Q b2 -> h8, killing Black R
        Black, d7d3: R d7 -> d3
        White, h8a8: Q h8 -> a8
        Black, c4c3: P c4 -> c3
        White, a8a4: Q a8 -> a4, check
        Black, d1e1: K d1 -> e1, escape check
        White, f3f4: P f3 -> f4
        Black, f7f5: P f7 -> f5
        White, b1c1: K b1 -> c1
        Black, d3d2: R d3 -> d2
        White, a4a7: Q a4 -> a7
        Black, d2d1: R d2 -> d1, check
        White, c1c2: K c1 -> c2, escape check
        Black, g6g5: P g6 -> g5
        White, a7g1: Q a7 -> g1 checkmate and threatening to kill Black R,  winning!!!


        Game 2.
        White, d2d4: P
        Black, d7d5: P
        White, c2c4: P
        Black, c7c6: P
        White, g1f3: N
        Black, g8f6: N
        White, b1c3: N
        Black, e7e6: P
        White, e2d3: P
        Black, b8d7: N
        White, f1d3: B
        Black, d5c4: P, killing White P
        White, d3c4: B, killing Black P
        Black, b7b5: P
        White, c4d3: B
        Black, f8d6: B
        White, Castling: K e1g1, R h1f1
        Black, Castling: K e8g8, R h8f8
        White, d1c2: Q
        Black, c8b7: B
        White, a2a3: P
        Black, a8c8: R
        White, f3g5: N
        Black, c6c5: P
        White, g5h7: N, killing Black P
        Black, f6g4: N
        White, f2f4: P
        Black, c5d4: P, killing White P
        White, e3d4: P, killing Black P
        Black, d6c5: B
        White, d3e2: B
        Black, d7e5: N
        White, e2g4: B
        Black, c5d4: B, killing White P and check
        White, g1h1: K, escape check
        Black, e5g4: N, killing White B
        White, h7f8: N, killing Black R
        Black, f7f5: P
        White, f8g6: P
        Black, d8f6: Q
        White, h2h3: P
        Black, f6g6: Q, killing White N
        White, c2e2: Q
        Black, g6h5: Q
        White, e2d3: Q
        Black, d4e3: B, winning (can you figure out why)

    Current board position (FEN): {board.fen()}.
    Choose carefully one of the following legal hops: {[hop.uci() for hop in board.legal_moves]},
    to maximize the chance of checkmate. Always think two-step ahead before make the actual hop.
    That is, think over how your opponent would hop based on your new hop,
    and what would you make the next best hop and accordingly the opponent's best next hop.
    Your best hop shall be determined by the two-step forward calculation.
    Warning: If you choose beyond legal hops, you are deemed as loosing the game.
    """}]

        short_messages = [
            {"role": "system", "content": "You are a chess engine. Respond only with a valid UCI move."},
            {
                "role": "user",
                "content": f"""You are playing a chess game as {self.role}.
                 Current board position (FEN): {board.fen()}
                 Choose one of the following legal moves: {[move.uci() for move in board.legal_moves]}
                 Provide your next move in UCI format (e.g., 'e2e4'). Only provide the move."""
            }
        ]

        # messages = short_messages
        messages = long_messages

        print(f"{board.fen()}")
        print(f"{[move.uci() for move in board.legal_moves]}")
        request_data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 256,
            # "stream": True
        }

        # print("\n=== HTTP Request Details ===")
        # print(f"URL: {model_config['url']}")
        # print(f"Model: {model_config['name']}")
        # print("Method: POST")
        # print("Headers: {'Content-Type': 'application/json'}")
        # print("Request Body:")
        # print(json.dumps(request_data, indent=2))
        # print("========================\n")

        print(f"{self.role.upper()}, played by {
              self.model}, is thinking...")

        response = requests.post(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            json=request_data,
            headers={"Content-Type": "application/json"},
        )
        print(
            f"---> response.status_code: «{response.status_code}» response.text «{response.text}»")
        if response.status_code == 200:
            chess_move_uci = ""
            response_content = ""
            # for line in response.content:
            #     if line:
            # Decode the line and remove "data: " prefix
            line_text = response.text.strip()
            if line_text.startswith('data: '):
                line_text = line_text[6:]
                print(
                    f"---> LLM's raw JSON response: «{line_text}»")
            try:
                response_object = json.loads(line_text)
                # STANDARD COMPLETION format
                # {"id":"chat-bca2f503772249199b4e5762c0172eb9","object":"chat.completion","created":1731018756,"model":"Llama-3.1-8B","choices":[{"index":0,"message":{"role":"assistant","content":"e7e5","tool_calls":[]},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":240,"total_tokens":245,"completion_tokens":5},"prompt_logprobs":null}
                # if chunk.get('choices') and chunk['choices'][0].get('message',{}).get('content'):
                #     content = chunk['choices'][0].get('message',{}).get('content')
                # STREAMING MODE format
                # if chunk.get('choices') and chunk['choices'][0].get('delta', {}).get('content'):
                #     content = chunk['choices'][0].get('delta', {}).get('content')
                if response_object.get('choices') and response_object['choices'][0].get('message', {}).get('content'):
                    response_content = response_object['choices'][0].get(
                        'message', {}).get('content')
            except json.JSONDecodeError:
                print("BAD JSON!")
            match = re.search(r'[a-h][1-8][a-h][1-8]', response_content)
            chess_move_uci = match.group(0) if match else ""
            print(
                f"---> chess_move_uci: «{chess_move_uci}», response_content: «{response_content}»")
            return chess_move_uci

    def get_config(self) -> dict:
        return {"model": self.model}


class StockfishPlayer(Player):
    @staticmethod
    def get_stockfish_path() -> str:
        """
        Determines the operating system and returns the appropriate path for Stockfish.

        Returns:
            str: Path to the Stockfish executable based on the operating system.
        """
        if platform.system() == "Linux":
            return "/usr/games/stockfish"
        elif platform.system() == "Darwin":  # Darwin is the system name for macOS
            return "stockfish"
        elif platform.system() == "Windows":
            return (
                r"C:\Users\adamk\Documents\Stockfish\stockfish-windows-x86-64-avx2.exe"
            )
        else:
            raise OSError("Unsupported operating system")

    def __init__(self, skill_level: int, play_time: float):
        self._skill_level = skill_level
        self._play_time = play_time
        # If getting started, you need to run brew install stockfish
        stockfish_path = StockfishPlayer.get_stockfish_path()
        self._engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    def get_move(
        self, board: chess.Board, game_state: str, temperature: float
    ) -> Optional[str]:
        if self._skill_level == -2:
            legal_moves = list(board.legal_moves)
            random_move = random.choice(legal_moves)
            return board.san(random_move)
        elif self._skill_level < 0:
            self._engine.configure({"Skill Level": 0})
            result = self._engine.play(
                board, chess.engine.Limit(time=1e-8, depth=1, nodes=1)
            )

        else:
            self._engine.configure({"Skill Level": self._skill_level})
            result = self._engine.play(
                board, chess.engine.Limit(time=self._play_time))
        if result.move is None:
            return None
        return board.san(result.move)

    def get_config(self) -> dict:
        return {"skill_level": self._skill_level, "play_time": self._play_time}

    def close(self):
        self._engine.quit()


def get_gpt_response(game_state: str, model: str, temperature: float) -> Optional[str]:
    # trying to prevent what I believe to be rate limit issues
    if model == "gpt-4":
        time.sleep(0.4)
    response = gpt_query.get_gpt_response(game_state, model, temperature)
    return response


def get_move_from_gpt_response(response: Optional[str]) -> Optional[str]:
    if response is None:
        return None

    # Parse the response to get only the first move
    moves = response.split()
    first_move = moves[0] if moves else None

    return first_move


def record_results(
    board: chess.Board,
    player_one: Player,
    player_two: Player,
    game_state: str,
    player_one_illegal_moves: int,
    player_two_illegal_moves: int,
    player_one_legal_moves: int,
    player_two_legal_moves: int,
    total_time: float,
    player_one_resignation: bool,
    player_two_resignation: bool,
    player_one_failed_to_find_legal_move: bool,
    player_two_failed_to_find_legal_move: bool,
    total_moves: int,
    illegal_moves: int,
):
    unique_game_id = generate_unique_game_id()

    (
        player_one_title,
        player_two_title,
        player_one_time,
        player_two_time,
    ) = get_player_titles_and_time(player_one, player_two)

    if player_one_resignation or player_one_failed_to_find_legal_move:
        result = "0-1"
        player_one_score = 0
        player_two_score = 1
    elif player_two_resignation or player_two_failed_to_find_legal_move:
        result = "1-0"
        player_one_score = 1
        player_two_score = 0
    else:
        result = board.result()
        # Hmmm.... debating this one. Annoying if I leave it running and it fails here for some reason, probably involving some
        # resignation / failed move situation I didn't think of
        # -1e10 at least ensures it doesn't fail silently
        if "-" in result:
            player_one_score = result.split("-")[0]
            player_two_score = result.split("-")[1]
        elif result == "*":  # Draw due to hitting max moves
            player_one_score = 1 / 2
            player_two_score = 1 / 2
        else:
            player_one_score = -1e10
            player_two_score = -1e10

    info_dict = {
        "game_id": unique_game_id,
        "transcript": game_state,
        "result": result,
        "player_one": player_one_title,
        "player_two": player_two_title,
        "player_one_time": player_one_time,
        "player_two_time": player_two_time,
        "player_one_score": player_one_score,
        "player_two_score": player_two_score,
        "player_one_illegal_moves": player_one_illegal_moves,
        "player_two_illegal_moves": player_two_illegal_moves,
        "player_one_legal_moves": player_one_legal_moves,
        "player_two_legal_moves": player_two_legal_moves,
        "player_one_resignation": player_one_resignation,
        "player_two_resignation": player_two_resignation,
        "player_one_failed_to_find_legal_move": player_one_failed_to_find_legal_move,
        "player_two_failed_to_find_legal_move": player_two_failed_to_find_legal_move,
        "game_title": f"{player_one_title} vs. {player_two_title}",
        "number_of_moves": board.fullmove_number,
        "time_taken": total_time,
        "total_moves": total_moves,
        "illegal_moves": illegal_moves,
    }

    if RUN_FOR_ANALYSIS:
        csv_file_path = (
            f"logs/{player_one_recording_name}_vs_{player_two_recording_name}"
        )
        csv_file_path = csv_file_path.replace(
            ".", "_"
        )  # filenames can't have periods in them. Useful for e.g. gpt-3.5 models
        csv_file_path += ".csv"
    else:
        csv_file_path = recording_file

    # Determine if we need to write headers (in case the file doesn't exist yet)
    write_headers = not os.path.exists(csv_file_path)

    # Append the results to the CSV file
    with open(csv_file_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=info_dict.keys())
        if write_headers:
            writer.writeheader()
        writer.writerow(info_dict)

    with open("game.txt", "w") as f:
        f.write(game_state)


def generate_unique_game_id() -> str:
    timestamp = int(time.time())
    random_num = random.randint(1000, 9999)  # 4-digit random number
    return f"{timestamp}-{random_num}"


def get_player_titles_and_time(
    player_one: Player, player_two: Player
) -> Tuple[str, str, Optional[float], Optional[float]]:
    player_one_config = player_one.get_config()
    player_two_config = player_two.get_config()

    # For player one
    if "model" in player_one_config:
        player_one_title = player_one_config["model"]
        player_one_time = None
    else:
        player_one_title = f"Stockfish {player_one_config['skill_level']}"
        player_one_time = player_one_config["play_time"]

    # For player two
    if "model" in player_two_config:
        player_two_title = player_two_config["model"]
        player_two_time = None
    else:
        player_two_title = f"Stockfish {player_two_config['skill_level']}"
        player_two_time = player_two_config["play_time"]

    return (player_one_title, player_two_title, player_one_time, player_two_time)


def initialize_game_with_opening(
    game_state: str, board: chess.Board
) -> Tuple[str, chess.Board]:
    with open("openings.csv", "r") as file:
        lines = file.readlines()[1:]  # Skip header
    moves_string = random.choice(lines)
    game_state += moves_string
    # Splitting the moves string on spaces
    tokens = moves_string.split()

    for token in tokens:
        # If the token contains a period, it's a move number + move combination
        if "." in token:
            move = token.split(".")[-1]  # Take the move part after the period
        else:
            move = token

        board.push_san(move)
    return game_state, board


# Return is (move_san, move_uci, attempts, is_resignation, is_illegal_move)
def get_legal_move(
    player: Player,
    board: chess.Board,
    game_state: str,
    player_one: bool,
    max_attempts: int = 5,
) -> LegalMoveResponse:
    """Request a move from the player and ensure it's legal."""
    move_san = None
    move_uci = None

    for attempt in range(max_attempts):
        move_san = player.get_move(
            board, game_state, min(((attempt / max_attempts) * 1) + 0.001, 0.5)
        )

        # Sometimes when GPT thinks it's the end of the game, it will just output the result
        # Like "1-0". If so, this really isn't an illegal move, so we'll add a check for that.
        if move_san is not None:
            if move_san == "1-0" or move_san == "0-1" or move_san == "1/2-1/2":
                print(f"{move_san}, player has resigned")
                return LegalMoveResponse(
                    move_san=None,
                    move_uci=None,
                    attempts=attempt,
                    is_resignation=True,
                )

        try:
            move_uci = board.parse_san(move_san)
        except Exception as e:
            print(f"Error parsing move {move_san}: {e}")
            # check if player is gpt-3.5-turbo-instruct
            # only recording errors for gpt-3.5-turbo-instruct because it's errors are so rare
            if player.get_config()["model"] == "gpt-3.5-turbo-instruct":
                with open("gpt-3.5-turbo-instruct-illegal-moves.txt", "a") as f:
                    f.write(f"{game_state}\n{move_san}\n")
            continue

        if move_uci in board.legal_moves:
            if not move_san.startswith(" "):
                move_san = " " + move_san
            return LegalMoveResponse(move_san, move_uci, attempt)
        print(f"Illegal move: {move_san}")

    # If we reach here, the player has made illegal moves for all attempts.
    print(f"{player} provided illegal moves for {max_attempts} attempts.")
    return LegalMoveResponse(
        move_san=None, move_uci=None, attempts=max_attempts, is_illegal_move=True
    )


def play_turn(
    player: Player, board: chess.Board, game_state: str, player_one: bool
) -> Tuple[str, bool, bool, int]:
    result = get_legal_move(player, board, game_state, player_one, 5)
    illegal_moves = result.attempts
    move_san = result.move_san
    move_uci = result.move_uci
    resignation = result.is_resignation
    failed_to_find_legal_move = result.is_illegal_move

    if resignation:
        print(f"{player} resigned with result: {board.result()}")
    elif failed_to_find_legal_move:
        print(f"Game over: 5 consecutive illegal moves from {player}")
    elif move_san is None or move_uci is None:
        print(f"Game over: {player} failed to find a legal move")
    else:
        board.push(move_uci)
        game_state += move_san
        print(move_san, end=" ")

    return game_state, resignation, failed_to_find_legal_move, illegal_moves


def initialize_game_with_random_moves(
    board: chess.Board, initial_game_state: str, randomize_opening_moves: int
) -> tuple[str, chess.Board]:
    # We loop for multiple attempts because sometimes the random moves will result in a game over
    MAX_INIT_ATTEMPTS = 5
    for attempt in range(MAX_INIT_ATTEMPTS):
        board.reset()  # Reset the board for a new attempt
        game_state = initial_game_state  # Reset the game state for a new attempt
        moves = []
        for moveIdx in range(1, randomize_opening_moves + 1):
            for player in range(2):
                moves = list(board.legal_moves)
                if not moves:
                    break  # Break if no legal moves are available

                move = random.choice(moves)
                moveString = board.san(move)
                if moveIdx > 1 or player == 1:
                    game_state += " "
                game_state += (
                    str(moveIdx) + ". " +
                    moveString if player == 0 else moveString
                )
                board.push(move)

            if not moves:
                break  # Break if no legal moves are available

        if moves:
            # Successful generation of moves, break out of the attempt loop
            break
    else:
        # If the loop completes without a break, raise an error
        raise Exception(
            "Failed to initialize the game after maximum attempts.")

    print(game_state)
    return game_state, board


def play_game(
    player_one: Player,
    player_two: Player,
    max_games: int = 10,
    randomize_opening_moves: Optional[int] = None,
):
    # NOTE: I'm being very particular with game_state formatting because I want to match the PGN notation exactly
    # It looks like this: 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 etc. HOWEVER, GPT prompts should not end with a trailing whitespace
    # due to tokenization issues. If you make changes, ensure it still matches the PGN notation exactly.
    for _ in range(max_games):  # Play 10 games
        with open("gpt_inputs/prompt.txt", "r") as f:
            game_state = f.read()
        board = chess.Board()

        if randomize_opening_moves is not None:
            game_state, board = initialize_game_with_random_moves(
                board, game_state, randomize_opening_moves
            )

        player_one_illegal_moves = 0
        player_two_illegal_moves = 0
        player_one_legal_moves = 0
        player_two_legal_moves = 0
        player_one_resignation = False
        player_two_resignation = False
        player_one_failed_to_find_legal_move = False
        player_two_failed_to_find_legal_move = False
        start_time = time.time()

        total_moves = 0
        illegal_moves = 0

        while not board.is_game_over():
            with open("game.txt", "w") as f:
                f.write(game_state)
            current_move_num = str(board.fullmove_number) + "."
            total_moves += 1
            # I increment legal moves here so player_two isn't penalized for the game ending before its turn
            player_one_legal_moves += 1
            player_two_legal_moves += 1

            # this if statement may be overkill, just trying to get format to exactly match PGN notation
            if board.fullmove_number != 1:
                game_state += " "
            game_state += current_move_num
            print(f"{current_move_num}", end="")

            (
                game_state,
                player_one_resignation,
                player_one_failed_to_find_legal_move,
                illegal_moves_one,
            ) = play_turn(player_one, board, game_state, player_one=True)
            player_one_illegal_moves += illegal_moves_one
            if illegal_moves_one != 0:
                player_one_legal_moves -= 1
            if (
                board.is_game_over()
                or player_one_resignation
                or player_one_failed_to_find_legal_move
            ):
                break

            (
                game_state,
                player_two_resignation,
                player_two_failed_to_find_legal_move,
                illegal_moves_two,
            ) = play_turn(player_two, board, game_state, player_one=False)
            player_two_illegal_moves += illegal_moves_two
            if illegal_moves_two != 0:
                player_two_legal_moves -= 1
            if (
                board.is_game_over()
                or player_two_resignation
                or player_two_failed_to_find_legal_move
            ):
                break

            print("\n", end="")

            if total_moves > MAX_MOVES:
                break

        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nGame over. Total time: {total_time} seconds")
        print(f"Result: {board.result()}")
        print(board)
        print()
        record_results(
            board,
            player_one,
            player_two,
            game_state,
            player_one_illegal_moves,
            player_two_illegal_moves,
            player_one_legal_moves,
            player_two_legal_moves,
            total_time,
            player_one_resignation,
            player_two_resignation,
            player_one_failed_to_find_legal_move,
            player_two_failed_to_find_legal_move,
            total_moves,
            illegal_moves,
        )
    if isinstance(player_one, StockfishPlayer):
        player_one.close()
    if isinstance(player_two, StockfishPlayer):
        player_two.close()

        # print(game_state)


l_8B_instruct = ModelPort(8080, "Llama-3.1-8B-Instruct")
l_70B_instruct = ModelPort(8082, "Llama-3.1-70B-Instruct")
l_405B_instruct = ModelPort(8084, "Llama-3.1-405B-Instruct")
l_3B_instruct = ModelPort(8085, "Llama-3.2-3B-Instruct")
l_11B_instruct = ModelPort(8087, "Llama-3.2-11B-Instruct")
l_90B_instruct = ModelPort(8088, "Llama-3.2-90B-Vision-Instruct")

NANOGPT = False
RUN_FOR_ANALYSIS = True
MAX_MOVES = 1000
if NANOGPT:
    MAX_MOVES = 89  # Due to nanogpt max input length of 1024
# default recording file. Because we are using list [player_ones], recording_file is overwritten
recording_file = "logs/determine.csv"
player_ones = ["Llama-3.2-3B-Instruct-long_prompt"]
player_two_recording_name = "stockfish"
if __name__ == "__main__":
    for player in player_ones:
        player_one_recording_name = player
        # play_time starts at 0.05 s (50ms) at lowest setting
        for skill_level in [1]:
            num_games = 100
            # player_one = GPTPlayer(model=player)
            player_one = LLMPlayer(l_3B_instruct, role="white")
            # player_two = LLMPlayer(l_70B_instruct, role="black")
            # player_one = GPTPlayer(model="gpt-4")
            # player_one = StockfishPlayer(skill_level=-1, play_time=0.1)
            # player_one = NanoGptPlayer(model_name=player_one_recording_name)
            # skill_level -2 is random legal move, -1 is depth-of-1-and-time-of-10^(-8)-seconds
            player_two = StockfishPlayer(
                skill_level=skill_level, play_time=0.1)
            # player_two = GPTPlayer(model="gpt-4")
            # player_two = GPTPlayer(model="gpt-3.5-turbo-instruct")

            play_game(player_one, player_two, num_games)
