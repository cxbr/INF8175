from __future__ import annotations
import json
from typing import Dict, List, Tuple
from colorama import Fore, Style
from seahorse.game.game_layout.board import Board, Piece
from seahorse.utils.serializer import Serializable

class BoardDivercite(Board):
    """
    A class representing an Divercite board.

    Attributes:
        env (dict[Tuple[int], Piece]): The environment dictionary composed of pieces.
        dimensions (list[int]): The dimensions of the board.
    """

    #EMPTY_POS=3
    FORBIDDEN_POS=0
    CITY_POS=1
    RESOURCE_POS=2


    FORBIDDEN_MASK = [
     [True,  True,  True,  True,  False, True,  True,  True,  True],
     [True,  True,  True,  False, False, False, True,  True,  True],
     [True,  True,  False, False, False, False, False, True,  True],
     [True,  False, False, False, False, False, False, False, True],
     [False, False, False, False, False, False, False, False, False],
     [True,  False, False, False, False, False, False, False, True],
     [True,  True,  False, False, False, False, False, True,  True],
     [True,  True,  True,  False, False, False, True,  True,  True],
     [True,  True,  True,  True,  False, True,  True,  True,  True]
    ]

    BOARD_MASK = [
            [ 0,   0,   0,   0,  'R',  0,   0,   0,   0],
            [ 0,   0,   0,  'R', 'C', 'R',  0,   0,   0],
            [ 0,   0,  'R', 'C', 'R', 'C', 'R',  0,   0],
            [ 0,  'R', 'C', 'R', 'C', 'R', 'C', 'R',  0],
            ['R', 'C', 'R', 'C', 'R', 'C', 'R', 'C', 'R'],
            [ 0,  'R', 'C', 'R', 'C', 'R', 'C', 'R',  0],
            [ 0,   0,  'R', 'C', 'R', 'C', 'R',  0,   0],
            [ 0,   0,   0,  'R', 'C', 'R',  0,   0,   0],
            [ 0,   0,   0,   0,  'R',  0,   0,   0,   0]
]


    def __init__(self, env: dict[tuple[int], Piece], dim: list[int]) -> None:
        super().__init__(env, dim)

    def __str__(self):
        grid_data = self.get_grid()
        rotated_grid = self.rotate_grid_45(grid_data)
        board_string = "\n"
        max_len = max(len(row) for row in rotated_grid)
        for i, row in enumerate(rotated_grid):
            if all(cell == ' ' for cell in row):
                continue
            padded_row = [' '] * ((max_len - len(row)) // 2) + row + [' '] * ((max_len - len(row)) // 2)
            if i%2 == 1:
                padded_row = [''] + padded_row
            for cell in padded_row:
                if isinstance(cell, tuple):
                    char, color = cell
                    if color == 'R':
                        board_string += Fore.RED + char + Style.RESET_ALL + " "
                    elif color == 'G':
                        board_string += Fore.GREEN + char + Style.RESET_ALL + " "
                    elif color == 'Y':
                        board_string += Fore.YELLOW + char + Style.RESET_ALL + " "
                    elif color == 'B':
                        board_string += Fore.BLUE + char + Style.RESET_ALL + " "
                    elif color == 'Black':
                        board_string += Fore.BLACK + char + Style.RESET_ALL + " "
                else:
                    board_string += cell + "  "
            board_string += "\n"
        return board_string
    
    # def __str__(self):
    #     grid_data = self.get_grid()
    #     board_string = ""
    #     for i in range(self.dimensions[0]):
    #         # if i % 2 == 1:
    #         #     board_string += " "  # Add an extra space for odd rows
    #         for j in range(self.dimensions[1]):
    #             cell = grid_data[i][j]
    #             if isinstance(cell, tuple):
    #                 char, color = cell
    #                 if color == 'R':
    #                     board_string += Fore.RED + char + Style.RESET_ALL + " "
    #                 elif color == 'G':
    #                     board_string += Fore.GREEN + char + Style.RESET_ALL + " "
    #                 elif color == 'Y':
    #                     board_string += Fore.YELLOW + char + Style.RESET_ALL + " "
    #                 elif color == 'B':
    #                     board_string += Fore.BLUE + char + Style.RESET_ALL + " "
    #                 elif color == 'Black':
    #                     board_string += Fore.BLACK + char + Style.RESET_ALL + " "
    #             else:
    #                 board_string += cell + "  "
    #         board_string += "\n"
    #     return board_string


    
    def get_neighbours(self, i:int ,j: int) -> Dict[str,Tuple[str|Piece,Tuple[int,int]]]:
        """ returns a dictionnary of the neighbours of the cell (i,j) with the following format:
            
        (neighbour_name: (neighbour_type, (i,j)))


        Args:
            i (int): line indice
            j (int): column indice

        Returns:
            Dict[str,Tuple[str,Tuple[int,int]]]: dictionnary of the neighbours of the cell (i,j)
        """
        neighbours = {"top_right":(i-1, j), "top_left":(i,j-1), "bot_left":(i, j+1), "bot_right":(i+1,j)}
        for k,v in neighbours.items():
            if v not in self.env.keys():
                if v[0] < 0 or v[1] < 0 or v[0] >= self.dimensions[0] or v[1] >= self.dimensions[1]:
                    neighbours[k] = ("OUTSIDE", neighbours[k])
                else:
                    if BoardDivercite.FORBIDDEN_MASK[v[0]][v[1]]:
                        neighbours[k] = ("OUTSIDE",neighbours[k])
                    else:
                        neighbours[k] = ("EMPTY",neighbours[k])
            else:
                neighbours[k] = (self.env[neighbours[k]],neighbours[k])
        return neighbours

    def get_grid(self) -> List[List[int]]:
        """
        Return a nice representation of the board.

        Returns:
            str: The nice representation of the board.
        """
        grid_data = [
            [0, 0, 0, 0, 2, 0, 0, 0, 0],
            [0, 0, 0, 2, 1, 2, 0, 0, 0],
            [0, 0, 2, 1, 2, 1, 2, 0, 0],
            [0, 2, 1, 2, 1, 2, 1, 2, 0],
            [2, 1, 2, 1, 2, 1, 2, 1, 2],
            [0, 2, 1, 2, 1, 2, 1, 2, 0],
            [0, 0, 2, 1, 2, 1, 2, 0, 0],
            [0, 0, 0, 2, 1, 2, 0, 0, 0],
            [0, 0, 0, 0, 2, 0, 0, 0, 0],
        ]
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                if (i,j) in self.env:
                    piece_type = self.env[(i,j)].get_type()
                    piece_color, piece_res_city = piece_type[0], piece_type[1]
                    if piece_res_city == "C" and piece_type[2] == 'W':
                        char = "🅆"
                    elif piece_res_city == "C" and piece_type[2] == 'B':
                        char = "🄱"
                    else:
                        char = "◆ "
                    grid_data[i][j] = (char, piece_color)
                elif BoardDivercite.BOARD_MASK[i][j] == 'C':
                    grid_data[i][j] = ("▢ ", "Black")
                elif BoardDivercite.BOARD_MASK[i][j] == 'R':
                    grid_data[i][j] = ("◇ ", "Black")
                else:
                    grid_data[i][j] = " "
         
        return grid_data
    
    def rotate_grid_45(self, grid_data: List[List[tuple|str]]) -> List[List[tuple|str]]:
        """
        Rotate the grid by 45 degrees.

        Args:
            grid_data (List[List[int]]): The grid to rotate.

        Returns:
            List[List[int]]: The rotated grid.
        """
        rot_grid = []
        n = len(grid_data)
        for i in range(n):
            row = ['']*(n//2 +1) if i%2 == 0 else ['']*(n//2)
            rot_grid.append(row)
        
        for i in range(0, n):
            for j in range(0, n//2 + 1):
                if i%2 == 0:
                    rot_grid[i][j] = grid_data[i//2+n//2-j][j+i//2]
                else:
                    if j != 4:
                        rot_grid[i][j] = grid_data[i//2+n//2-j][j+1+i//2]

        return rot_grid
    
    def to_json(self) -> dict:
        """
        Converts the board to a JSON object.

        Returns:
            dict: The JSON representation of the board.
        """
        # TODO: migrate below into js code
        #board = [[None for _ in range(self.dimensions[1])] for _ in range(self.dimensions[0])]
        #for key, value in self.env.items():
        #    board[key[0]][key[1]] = value.piece_type if value is not None else None
        #return {"board": board}
        return {"env":{str(x):y for x,y in self.env.items()},"dim":self.dimensions}

    @classmethod
    def from_json(cls, data) -> Serializable:
        d = json.loads(data)
        dd = json.loads(data)
        for x,y in d["env"].items():
            # TODO eval is unsafe
            del dd["env"][x]
            dd["env"][eval(x)] = Piece.from_json(json.dumps(y))
        return cls(**dd)
