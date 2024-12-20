import numpy as np
from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.game.game_layout.board import Piece


class MyPlayer(PlayerDivercite):
    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        super().__init__(piece_type, name)
        self.__max_depth = 2
        self.__transposition_table = {}
        self.__evaluation_cache = {}
        self.__divercite_cache = {}
        self.__blocking_cache = {}
        self.__placement_cache = {}
        self.__final_move_cache = {}
        self.__winning_cache = {}
        self.__zobrist_table = self.__initialize_zobrist_table()

    def __initialize_zobrist_table(self):
        """Use NumPy for fast Zobrist table initialization."""
        board_size = 81
        rng = np.random.default_rng(seed=42)
        return rng.integers(0, 2**64, size=(board_size, 256), dtype=np.uint64)

    def compute_action(self, current_state: GameState, remaining_time: int = int(1e9), **kwargs) -> Action:
        possible_actions = list(
            current_state.generate_possible_heavy_actions())
        if not possible_actions:
            return None

        is_first_player = self.get_id() == current_state.get_next_player().get_id()

        env_size = len(current_state.get_rep().get_env()) // 2
        self.__max_depth = 2 if env_size < 10 else (4 if env_size < 17 else 9)

        current_hash = self.__compute_state_hash(current_state)

        best_action = None
        for depth in range(1, self.__max_depth + 1):
            best_action = self.__search_best_action(
                current_state, current_hash, possible_actions, depth, is_first_player)
        return best_action

    def __search_best_action(self, current_state, current_hash, possible_actions, depth, is_first_player):
        """Search for the best action using minimax with alpha-beta pruning."""
        pruned_actions = self.__prune_actions(
            possible_actions, current_hash, threshold=0.5)

        best_score = float('-inf')
        best_action = None

        for action in pruned_actions:
            next_state = action.get_next_game_state()
            next_hash = self.__compute_state_hash(next_state)

            score = self.__minimax(next_state, next_hash, depth - 1,
                                   False, float('-inf'), float('inf'), is_first_player)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def __minimax(self, state: GameStateDivercite, state_hash: int, depth: int, maximizing_player: bool, alpha: float, beta: float, is_first_player: bool) -> float:
        if state_hash in self.__transposition_table:
            stored_entry = self.__transposition_table[state_hash]
            if stored_entry['depth'] >= depth:
                return stored_entry['score']

        if depth == 0 or state.is_done():
            eval_score = self.__evaluate_state(
                state, state_hash, is_first_player)
            self.__transposition_table[state_hash] = {
                'score': eval_score, 'depth': depth}
            return eval_score

        actions = list(state.generate_possible_heavy_actions())
        if not actions:
            eval_score = self.__evaluate_state(
                state, state_hash, is_first_player)
            self.__transposition_table[state_hash] = {
                'score': eval_score, 'depth': depth}
            return eval_score

        actions.sort(
            key=lambda action: self.__evaluate_state(action.get_next_game_state(
            ), self.__compute_state_hash(action.get_next_game_state()), is_first_player),
            reverse=maximizing_player,
        )

        if maximizing_player:
            max_eval = float('-inf')
            for action in actions:
                next_state = action.get_next_game_state()
                next_hash = self.__compute_state_hash(next_state)

                eval_score = self.__minimax(
                    next_state, next_hash, depth - 1, False, alpha, beta, is_first_player)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            self.__transposition_table[state_hash] = {
                'score': max_eval, 'depth': depth}
            return max_eval
        else:
            min_eval = float('inf')
            for action in actions:
                next_state = action.get_next_game_state()
                next_hash = self.__compute_state_hash(next_state)

                eval_score = self.__minimax(
                    next_state, next_hash, depth - 1, True, alpha, beta, is_first_player)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            self.__transposition_table[state_hash] = {
                'score': min_eval, 'depth': depth}
            return min_eval

    def __prune_actions(self, actions, state_hash, threshold):
        """Prune actions based on a heuristic threshold."""
        return [action for action in actions if self.__evaluate_action(action, state_hash) > threshold]

    def __evaluate_action(self, action, state_hash):
        """A heuristic evaluation of the action's potential."""
        next_state = action.get_next_game_state()
        next_hash = self.__compute_state_hash(next_state)
        return self.__evaluate_state(next_state, next_hash, is_first_player=True)

    def __compute_state_hash(self, state: GameStateDivercite) -> int:
        """Compute Zobrist hash for the game state."""
        board = state.get_rep().get_env()
        hash_value = np.uint64(0)
        for (i, j), piece in board.items():
            if piece:
                position = i * 8 + j
                piece_type = ord(piece.get_type()[0])
                hash_value ^= self.__zobrist_table[position, piece_type]
        return hash_value

    def __evaluate_state(self, state: GameStateDivercite, state_hash: int, is_first_player: bool) -> float:
        """Improved heuristic evaluation with caching."""
        if state_hash in self.__evaluation_cache:
            return self.__evaluation_cache[state_hash]

        player_score = state.scores[self.get_id()]
        opponent_id = next(player.get_id()
                           for player in state.players if player.get_id() != self.get_id())
        opponent_score = state.scores[opponent_id]

        player_divercite = self.__cached_divercite_potential(
            state, state_hash, self.get_id())
        opponent_divercite = self.__cached_divercite_potential(
            state, state_hash, opponent_id)

        blocking = self.__cached_blocking_potential(
            state, state_hash, opponent_id)
        resource_placement = self.__cached_resource_placement_potential(
            state, state_hash)
        final_move_control = self.__cached_final_move_control(
            state, state_hash, opponent_id)
        winning_patterns = self.__cached_winning_patterns(state, state_hash)

        weights = np.array([4.0, 1.5, 0.5, 1.0, 1.0, 1.0]) if is_first_player else np.array(
            [2.5, 1.0, 0.5, 0.8, 1.0, 1.0])
        features = np.array([player_score - opponent_score,
                             player_divercite - opponent_divercite,
                             blocking,
                             resource_placement,
                             final_move_control,
                             winning_patterns])

        eval_score = np.dot(weights, features)

        self.__evaluation_cache[state_hash] = eval_score
        return eval_score

    def __cached_divercite_potential(self, state, state_hash, player_id):
        """Cache the divercite potential evaluation."""
        if state_hash in self.__divercite_cache:
            return self.__divercite_cache[state_hash]
        result = self.__count_divercite_potential(state, player_id)
        self.__divercite_cache[state_hash] = result
        return result

    def __cached_blocking_potential(self, state, state_hash, opponent_id):
        """Cache the blocking potential evaluation."""
        if state_hash in self.__blocking_cache:
            return self.__blocking_cache[state_hash]
        result = self.__evaluate_blocking_potential(state, opponent_id)
        self.__blocking_cache[state_hash] = result
        return result

    def __cached_resource_placement_potential(self, state, state_hash):
        """Cache the resource placement evaluation."""
        if state_hash in self.__placement_cache:
            return self.__placement_cache[state_hash]
        result = self.__evaluate_resource_placement_potential(state)
        self.__placement_cache[state_hash] = result
        return result

    def __cached_final_move_control(self, state, state_hash, opponent_id):
        """Cache the final move control evaluation."""
        if state_hash in self.__final_move_cache:
            return self.__final_move_cache[state_hash]
        result = self.__evaluate_final_move_control(state, opponent_id)
        self.__final_move_cache[state_hash] = result
        return result

    def __cached_winning_patterns(self, state, state_hash):
        """Cache the winning patterns evaluation."""
        if state_hash in self.__winning_cache:
            return self.__winning_cache[state_hash]
        result = self.__evaluate_winning_patterns(state)
        self.__winning_cache[state_hash] = result
        return result

    def __count_divercite_potential(self, state: GameStateDivercite, player_id: int) -> int:
        """Count potential for diversities."""
        return sum(
            4 - len({n[0].get_type()[0] for n in state.get_neighbours(i,
                    j).values() if isinstance(n[0], Piece)})
            for (i, j), piece in state.get_rep().get_env().items()
            if piece.get_owner_id() == player_id and piece.get_type()[1] == 'C'
        )

    def __evaluate_blocking_potential(self, state: GameStateDivercite, opponent_id: int) -> int:
        """Evaluate potential to block opponent diversities."""
        return sum(
            4 - len({n[0].get_type()[0] for n in state.get_neighbours(i,
                    j).values() if isinstance(n[0], Piece)})
            for (i, j), piece in state.get_rep().get_env().items()
            if piece.get_owner_id() == opponent_id and piece.get_type()[1] == 'C'
        )

    def __evaluate_resource_placement_potential(self, state: GameStateDivercite) -> float:
        """Evaluate the placement potential of resources."""
        return sum(
            (4 - len({n[0].get_type()[0] for n in state.get_neighbours(i,
             j).values() if isinstance(n[0], Piece)})) * 0.5
            for (i, j), piece in state.get_rep().get_env().items()
            if piece.get_type()[1] == 'C'
        )

    def __evaluate_final_move_control(self, state: GameStateDivercite, opponent_id: int) -> float:
        """Evaluate the control of the final move."""
        empty_slots = [(i, j) for (i, j), piece in state.get_rep(
        ).get_env().items() if piece is None]
        if len(empty_slots) != 1:
            return 0
        final_slot = empty_slots[0]
        surrounding_resources = state.get_neighbours(*final_slot)
        return -5 if any(len({n[0].get_type()[0] for n in surrounding_resources.values() if isinstance(n[0], Piece)}) == 3
                         for neighbor_piece, _ in surrounding_resources.values() if neighbor_piece and neighbor_piece.get_owner_id() == opponent_id) else 0

    def __evaluate_winning_patterns(self, state: GameStateDivercite) -> float:
        """Evaluate patterns that lead to winning."""
        return sum(
            5 if len({n[0].get_type()[0] for n in state.get_neighbours(
                i, j).values() if isinstance(n[0], Piece)}) == 4 else 3
            for (i, j), piece in state.get_rep().get_env().items()
            if piece.get_owner_id() == self.get_id() and piece.get_type()[1] == 'C'
        )
