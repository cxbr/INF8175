from __future__ import annotations

import json

from seahorse.game.action import Action
from seahorse.game.game_layout.board import Piece
from seahorse.player.player import Player
from seahorse.utils.serializer import Serializable


class PlayerDivercite(Player):
    """
    A player class for the Divercite game.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, name: str = "bob", *args, **kwargs) -> None:
        """
        Initializes a new instance of the PlayerDivercite class.

        Args:
            piece_type (str): The type of the player's game piece.
            name (str, optional): The name of the player. Defaults to "bob".
        """
        super().__init__(name,*args,**kwargs)
        self.piece_type = piece_type # White or Black
    
    def get_piece_type(self) -> str:
        """
        Gets the type of the player's game piece.

        Returns:
            str: The type of the player's game piece.
        """
        return self.piece_type

    def to_json(self) -> str:
        return {i:j for i,j in self.__dict__.items() if not i.startswith("_")}

    @classmethod
    def from_json(cls, data) -> Serializable:
        return PlayerDivercite(**json.loads(data))
