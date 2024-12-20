import copy
import json
import random
from typing import Dict, Generator, List, Optional, Set, Tuple

from board_divercite import BoardDivercite
from player_divercite import PlayerDivercite
from seahorse.game.game_layout.board import Piece
from seahorse.game.game_state import GameState
from seahorse.game.heavy_action import HeavyAction
from seahorse.game.light_action import LightAction
from seahorse.player.player import Player
from seahorse.utils.serializer import Serializable

class GameStateDivercite(GameState):
    """
    A class representing the state of an Divercite game.

    Attributes:
        score (list[float]): Scores of the state for each player.
        next_player (Player): Next player to play.
        players (list[Player]): List of players.
        rep (Representation): Representation of the game.
    """
    def __init__(self, scores: Dict, next_player: Player, players: List[Player], rep: BoardDivercite, step: int, 
                 players_pieces_left: dict[str: dict[str: int]],  *args, **kwargs) -> None:
        super().__init__(scores, next_player, players, rep)
        self.max_step = 40
        self.step = step
        self.players_pieces_left = {int(a):b for a,b in players_pieces_left.items()}

    def get_step(self) -> int:
        """
        Return the current step of the game.

        Returns:
            int: The current step of the game.
        """
        return self.step

    def is_done(self) -> bool:
        """
        Check if the game is finished.

        Returns:
            bool: True if the game is finished, False otherwise.
        """
        return self.step == self.max_step

    def get_neighbours(self, i: int, j: int) -> Dict[str,Tuple[str,Tuple[int,int]]]:
        return self.get_rep().get_neighbours(i, j)

    def in_board(self, index) -> bool:
        """
        Check if a given index is within the game board.

        Args:
            index: The index to check.

        Returns:
            bool: True if the index is within the game board, False otherwise.
        """
        return not BoardDivercite.FORBIDDEN_MASK[index[0]][index[1]]
    
    def piece_type_match(self, resource_or_city: str, pos: tuple) -> bool:
        """
        Check if a given piece can be placed on a given position (resource or city type must match).

        Args:
            res_city: The resource or city type.
            pos: The position to check.

        Returns:
            bool: True if the piece can be placed on the position, False otherwise.

        """
        return BoardDivercite.BOARD_MASK[pos[0]][pos[1]] == resource_or_city
    
    def get_player_id(self, pid) -> Player:
        """
        Get the player with the given ID.

        Args:
            pid: The ID of the player.

        Returns:
            Player: The player with the given ID.
        """
        for player in self.players:
            if player.get_id() == pid:
                return player
    
    def generate_possible_heavy_actions(self) -> Generator[HeavyAction, None, None]:
        """
        Generate possible actions.

        Returns:
            Generator[HeavyAction]: Generator of possible heavy actions.
        """
        current_rep = self.get_rep()
        b = current_rep.get_env()
        d = current_rep.get_dimensions()

        for piece, n_piece in self.players_pieces_left[self.next_player.get_id()].items():
            piece_color = piece[0]
            piece_res_city = piece[1]
            if n_piece > 0:
                for i in range(d[0]):
                    for j in range(d[1]):
                        if self.in_board((i, j)) and (i,j) not in b and self.piece_type_match(piece_res_city, (i, j)):
                            copy_b = copy.copy(b)
                            copy_b[(i, j)] = Piece(piece_type=piece_color+piece_res_city+self.next_player.piece_type, owner=self.next_player)
                            play_info = ((i,j), piece, self.next_player.get_id())
                            yield HeavyAction(
                                            self,
                                            GameStateDivercite(
                                                self.compute_scores(play_info),
                                                self.compute_next_player(),
                                                self.players,
                                                BoardDivercite(env=copy_b, dim=d),
                                                step=self.step + 1,
                                                players_pieces_left=self.compute_players_pieces_left(play_info),
                                            ),
                                        )
                            

    def generate_possible_light_actions(self) -> Generator[LightAction, None, None]:
        """
        Generate possible light actions for the current game state.

        Returns:
            Generator[LightAction]: Generator of possible light actions.

        """

        current_rep = self.get_rep()
        b = current_rep.get_env()
        d = current_rep.get_dimensions()
        for piece, n_piece in self.players_pieces_left[self.next_player.get_id()].items():
            piece_color = piece[0]
            piece_res_city = piece[1]
            if n_piece > 0:
                for i in range(d[0]):
                    for j in range(d[1]):
                        if self.in_board((i, j)) and (i,j) not in b and self.piece_type_match(piece_res_city, (i, j)):
                            data = {"piece": piece_color+piece_res_city, "position" : (i,j)}
                            yield LightAction(data)


    def apply_action(self, action: LightAction) -> GameState:
        """
        Apply an action to the game state.

        Args:
            action (LightAction): The action to apply.

        Returns:
            GameState: The new game state.
        """

        if not isinstance(action, LightAction):
            raise ValueError("The action must be a LightAction.")
        
        piece, position = action.data["piece"], action.data["position"]
        
        current_rep = self.get_rep()
        b = current_rep.get_env()
        d = current_rep.get_dimensions()
        copy_b = copy.copy(b)
        copy_b[position] = Piece(piece_type=piece+self.next_player.get_piece_type(), owner=self.next_player)
        new_board = BoardDivercite(env=copy_b, dim=d)
        play_info = (position, piece, self.next_player.get_id())

        return GameStateDivercite(
            self.compute_scores(play_info=play_info),
            self.compute_next_player(),
            self.players,
            new_board,
            step=self.step + 1,
            players_pieces_left=self.compute_players_pieces_left(play_info=play_info),
        )

    def convert_gui_data_to_action_data(self, gui_data: dict) -> dict:
        """
        Convert GUI data to action data.

        Args:
            gui_data (dict): The GUI data to convert.

        Returns:
            dict: The converted action data.
        """
        return {"piece": gui_data["piece"], "position": tuple(gui_data["position"])}

    def compute_players_pieces_left(self, play_info) -> dict[str: dict[str: int]]:
        """
        Compute the number of pieces left for each player.

        Args:
            id_add (int): The ID of the player to add the score for.

        Returns:
            dict[str: dict[str: int]]: A dictionary with player ID as the key and score as the value.
        """
        pos, piece, id_player = play_info
        players_pieces_left = copy.deepcopy(self.players_pieces_left)
        players_pieces_left[id_player][piece] -= 1
        return players_pieces_left
    
    def compute_scores(self, play_info: tuple) -> Dict[int, float]:
        """
        Compute the score of each player in a list.

        Args:
            id_add (int): The ID of the player to add the score for.

        Returns:
            dict[int, float]: A dictionary with player ID as the key and score as the value.
        """
        pos, piece, id_player = play_info
        color, res_city = piece[0], piece[1]
        scores = copy.copy(self.scores)
        if res_city == "C":
            if self.check_divercite(pos):
                scores[id_player] += 5
            else:
                scores[id_player] += len([n for n in self.get_neighbours(pos[0], pos[1]).values() 
                                          if isinstance(n[0], Piece) and n[0].get_type()[0] == color])
        else:            
            for n in self.get_neighbours(pos[0], pos[1]).values():
                if isinstance(n[0], Piece):
                    if self.check_divercite(n[1], color):
                        scores[n[0].get_owner_id()] -= int(n[0].get_type()[0] != color)
                        scores[n[0].get_owner_id()] += 5
                    else:
                        scores[n[0].get_owner_id()] += int(n[0].get_type()[0] == color)

        if self.step == self.max_step-1:
            # Last step, we prevent draws
            player1, player2 = self.players
            if scores[player1.get_id()] == scores[player2.get_id()]:
                
                env = copy.copy(self.get_rep().get_env())
                player = self.get_player_id(id_player)
                env[pos] = Piece(piece_type=color+res_city+player.piece_type, owner=player)
                new_board = BoardDivercite(env=env, dim=self.get_rep().get_dimensions())
                return self.remove_draw(scores, new_board)
        
        return scores
    
    def remove_draw(self, scores: dict, board: BoardDivercite) -> Dict[int, float]:
        """
        Remove the draw between two players.

        Args:
            scores (dict): The scores of the players.
            env (dict): The environment of the game.

        Returns:
            dict: The new scores of the players.
        """
        
        d = board.get_dimensions()
        env = board.get_env()
        
        def count_divercite(player_id: int) -> int:
            return sum([self.check_divercite((i,j), board=board) for i in range(d[0]) for j in range(d[1]) 
                        if self.in_board((i,j)) and env.get((i,j)) and env.get((i,j)).get_type()[1] == 'C' and env[(i,j)].get_owner_id() == player_id])
            
        
        def count_nstack(player_id, n) -> int:
            return sum([sum([p[0].get_type()[0] == env[(i,j)].get_type()[0] for p in board.get_neighbours(i,j).values() if isinstance(p[0], Piece)]) == n 
                        for i in range(d[0]) for j in range(d[1]) if self.in_board((i,j)) and env.get((i,j)) and env.get((i,j)).get_type()[1] == 'C' and env[(i,j)].get_owner_id() == player_id])
        
        player1, player2 = self.players
        
        player1_div = count_divercite(player1.get_id())
        player2_div = count_divercite(player2.get_id())
        
        scores[player1.get_id()] += player1_div > player2_div
        scores[player2.get_id()] += player2_div > player1_div 
                
        stack = 4
        while scores[player1.get_id()] == scores[player2.get_id()]:
            
            player1_stack = count_nstack(player1.get_id(), stack)
            player2_stack = count_nstack(player2.get_id(), stack)
            scores[player1.get_id()] += player1_stack > player2_stack
            scores[player2.get_id()] += player2_stack > player1_stack
            
            if stack == 2:
                winner = player1.get_id() # random.choice([player1.get_id(), player2.get_id()])
                scores[winner] += 1
                break
            stack -= 1
        
        return scores

    
    def check_divercite(self, pos, piece_color = None, board: BoardDivercite = None) -> bool:
        """
        Check if a given position has won a divercite.

        Args:
            pos: The position to check.

        Returns:
            bool: True if the position has won a divercite, False otherwise.
        """
        neighbors = self.get_neighbours(pos[0], pos[1]) if not board else board.get_neighbours(pos[0], pos[1])
        return len(set([n[0].get_type()[0] for n in neighbors.values() if isinstance(n[0], Piece)]).union(set([piece_color]) if piece_color else {})) == 4
    
    
    def __str__(self) -> str:
        if not self.is_done():
            return super().__str__()
        return "The game is finished!"

    def to_json(self) -> str:
        return { i:j for i,j in self.__dict__.items() if i!="_possible_light_actions" and i!="_possible_heavy_actions"}

    @classmethod
    def from_json(cls,data:str,*,next_player:Optional[PlayerDivercite]=None) -> Serializable:
        d = json.loads(data)
        return cls(**{**d,"scores":{int(k):v for k,v in d["scores"].items()},"players":[PlayerDivercite.from_json(json.dumps(x)) if not isinstance(x,str) else next_player for x in d["players"]],"next_player":next_player,"rep":BoardDivercite.from_json(json.dumps(d["rep"]))})

