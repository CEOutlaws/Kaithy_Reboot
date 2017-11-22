import numpy as np
import gym
from gym import spaces
from gym import error
from gym.utils import seeding
from six import StringIO
import sys
import os
import six
import random

from .util import gomoku_util
from .util import make_beginner_policy

# Rules from Wikipedia: Gomoku is an abstract strategy board game, Gobang or Five in a Row, it is traditionally played with Go pieces (black and white stones) on a go board with 19x19 or (15x15)
# The winner is the first player to get an unbroken row of five stones horizontally, vertically, or diagonally. (so-calle five-in-a row)
# Black plays first if white did not win in the previous game, and players alternate in placing a stone of their color on an empty intersection.


class GomokuState(object):
    '''
    Similar to Go game, Gomoku state consists of a current player and a board.
    Actions are exposed as integers in [0, num_actions), which is to place stone on empty intersection
    '''

    def __init__(self, board, color):
        '''
        Args:
            board: current board
            color: color of current player
        '''
        assert color in ['black', 'white'], 'Invalid player color'
        self.board, self.color = board, color

    def act(self, action):
        '''
        Executes an action for the current player

        Returns:
            a new GomokuState with the new board and the player switched
        '''
        return GomokuState(self.board.play(action, self.color), gomoku_util.other_color(self.color))

    def encode(self):
        '''Return: np array
            np.array(board_size, board_size, 3): state observation of the board
        '''
        obs_w_w_3 = np.zeros(
            (self.board.size, self.board.size, 3), dtype=np.int32)
        board_state_iter = np.nditer(
            self.board.board_state, flags=['multi_index'])
        while not board_state_iter.finished:
            obs_w_w_3[board_state_iter.multi_index][board_state_iter[0]] = 1
            obs_w_w_3[board_state_iter.multi_index][0] = gomoku_util.color_dict[self.color] - 1

            board_state_iter.iternext()

        return obs_w_w_3

    def __repr__(self):
        '''stream of board shape output'''
        # To Do: Output shape * * * o o
        return 'To play: {}\n{}'.format(six.u(self.color), self.board.__repr__())

# Sampling without replacement Wrapper
# sample() method will only sample from valid spaces


class DiscreteWrapper2d(spaces.Discrete):
    '''
    Attribute:
        invalid_mask:
            fill space is a space that no longer store any more value
            1 indicate fill space
            0 indicate empty space
    '''

    def __init__(self, width):
        self.n = width * width
        self.invalid_mask = np.zeros(self.n, dtype=np.int32)

    def sample(self):
        '''Only sample from the remaining valid action
        '''
        try:
            return np.random.choice(np.argwhere(self.invalid_mask == 0).flatten())
        except(ValueError):
            return print("No valid action available")

    def remove(self, s):
        '''Fill space s
        '''
        if s is None:
            return
        self.invalid_mask[s] = 1


# Environment
class GomokuEnv(gym.Env):
    '''
    GomokuEnv environment. Play against a fixed opponent.
    '''
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, player_color, opponent, board_size, random_reset=False):
        """
        Args:
            player_color: Stone color for the agent. Either 'black' or 'white'
            opponent: Name of the opponent policy, e.g. random, beginner, medium, expert
            board_size: board_size of the board to use
        """
        # Below attribute is used for randome_reset
        self.action_list = range(board_size * board_size)
        self.random_reset = random_reset

        self.board_size = board_size
        self.player_color = player_color

        self._seed()

        # opponentopponent_policy
        self.opponent_policy = None
        self.opponent = opponent

        # Observation space on board
        # board_size * board_size
        shape = (self.board_size, self.board_size, 3)
        self.observation_space = spaces.Box(np.zeros(shape), np.ones(shape))
        # One action for each board position
        self.action_space = DiscreteWrapper2d(self.board_size)

        # Keep track of the moves
        # self.moves = []

        # Empty State
        self.state = None

        # reset the board during initialization
        # self._reset()

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**32
        return [seed1, seed2]

    def _reset(self, custom_opponent_policy=None):
        self.state = GomokuState(
            Board(self.board_size), gomoku_util.BLACK)  # Black Plays First

        # reset action_space
        self.action_space = DiscreteWrapper2d(self.board_size)

        if self.random_reset:
            # self.action_space.invalid_mask =
            num_black_actions = random.randint(
                0, (len(self.action_list) - 1) / 3)

            black_actions = random.sample(self.action_list, num_black_actions)
            white_actions = random.sample(
                [piece for piece in self.action_list if piece not in black_actions], num_black_actions)

            for action in white_actions:
                color = 'white'
                self.state = GomokuState(self.state.board.play(action, color),
                                         gomoku_util.other_color(color))
                self.action_space.remove(action)
            for action in black_actions:
                color = 'black'
                self.state = GomokuState(self.state.board.play(action, color),
                                         gomoku_util.other_color(color))
                self.action_space.remove(action)

        # (re-initialize) the opponent,
        self._reset_opponent(self.state.board, custom_opponent_policy)
        # self.moves = []

        # Let the opponent play if it's not the agent's turn, there is no resign in Gomoku
        if self.state.color != self.player_color:
            opponent_action = self._exec_opponent_play(
                self.state, None, None)
            self.state = self.state.act(opponent_action)
            self.action_space.remove(opponent_action)
            # self.moves.append(self.state.board.last_coord)

        # We should be back to the agent color
        assert self.state.color == self.player_color

        self.done = self.state.board.is_terminal()
        return self.state.encode()

    def _close(self):
        self.opponent_policy = None
        self.state = None

    def _render(self, mode="human", close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(repr(self.state) + '\n')
        return outfile

    def _step(self, action):
        '''
        Args:
            action:
                value: 0 -> num_actions
                type: int
        Return:
            observation:
                board encoding
                type: 2D array, black X -> 1, white O -> 2
            reward:
                reward of the game,
                type: float
                value:
                    1: win
                    -1: lose or player's action is invalid
                    0: draw or nothing or opponent's action is invalid
            done:
                type: boolean
                value:
                    True: game is finish or invalid move is taken
                    False: vice versa
            info: state dict
        Raise:
            Illegal Move action, basically the position on board is not empty

        Args:
            action: function
        Do:
            Reset env and attach defined opponent policy to env
        '''
        # A trick to use step as general method of enviroment class
        if (callable(action)):
            return self._reset(custom_opponent_policy=action)

        assert self.state.color == self.player_color  # it's the player's turn
        # If already terminal, then don't do anything
        if self.done:
            return self.state.encode(), 0., True, {'state': self.state}

        # check if it's illegal move
        # if the space is fill
        if self.action_space.invalid_mask[action]:
            return self.state.encode(), -1., True, {'state': self.state}

        # Player play
        prev_state = self.state
        self.state = self.state.act(action)
        # self.moves.append(self.state.board.last_coord)
        # remove current action from action_space
        self.action_space.remove(action)

        # Opponent play
        if not self.state.board.is_terminal():
            opponent_action = self._exec_opponent_play(
                self.state, prev_state, action)
            # check if it's illegal move
            # if the space is fill
            if self.action_space.invalid_mask[opponent_action]:
                return self.state.encode(), 0., True, {'state': self.state}

            self.state = self.state.act(opponent_action)
            # self.moves.append(self.state.board.last_coord)
            # remove opponent action from action_space
            self.action_space.remove(opponent_action)
            # After opponent play, we should be back to the original color
            assert self.state.color == self.player_color

        # Reward: if nonterminal, there is no 5 in a row, then the reward is 0
        if not self.state.board.is_terminal():
            self.done = False
            return self.state.encode(), 0., False, {'state': self.state}

        # We're in a terminal state. Reward is 1 if won, -1 if lost
        assert self.state.board.is_terminal(), 'The game is terminal'
        self.done = True

        # Check Fianl wins
        exist, win_color = gomoku_util.check_five_in_row(
            self.state.board.board_state)  # 'empty', 'black', 'white'
        reward = 0.
        if win_color == "empty":  # draw
            reward = 0.
        else:
            # check if player_color is the win_color
            player_wins = (self.player_color == win_color)
            reward = 1. if player_wins else -1.
        return self.state.encode(), reward, True, {'state': self.state}

    def _exec_opponent_play(self, curr_state, prev_state, prev_action):
        '''There is no resign in gomoku'''
        assert curr_state.color != self.player_color
        return self.opponent_policy(
            curr_state, prev_state, prev_action)

    @property
    def _state(self):
        return self.state

    # @property
    # def _moves(self):
    #     return self.moves

    def _reset_opponent(self, board, custom_opponent_policy=None):
        if self.opponent == 'beginner':
            self.opponent_policy = make_beginner_policy(self.np_random)
        elif self.opponent == 'player':
            self.opponent_policy = custom_opponent_policy
        else:
            raise error.Error(
                'Unrecognized opponent policy {}'.format(self.opponent))


class Board(object):
    '''
    Basic Implementation of a Go Board, natural action are int [0,board_size**2)
    '''

    def __init__(self, board_size):
        self.size = board_size
        # initialize board states to empty
        self.board_state = np.array([[gomoku_util.color_dict['empty']]
                                     * board_size for i in range(board_size)], dtype=np.int32)
        self.move = 0                 # how many move has been made
        self.last_coord = (-1, -1)     # last action coord
        self.last_action = None       # last action made

    def coord_to_action(self, i, j):
        ''' convert coordinate i, j to action a in [0, board_size**2)
        '''
        return i * self.size + j  # action index

    def action_to_coord(self, a):
        return (a // self.size, a % self.size)

    def get_legal_move(self):
        ''' Get all the next legal move, namely empty space that you can place your 'color' stone
            Return: Coordinate of all the empty space, [(x1, y1), (x2, y2), ...]
        '''
        legal_move = []
        for i in range(self.size):
            for j in range(self.size):
                if (self.board_state[i][j] == 0):
                    legal_move.append((i, j))
        return legal_move

    def get_legal_action(self):
        ''' Get all the next legal action, namely empty space that you can place your 'color' stone
            Return: Coordinate of all the empty space, [(x1, y1), (x2, y2), ...]
        '''
        legal_action = []
        for i in range(self.size):
            for j in range(self.size):
                if (self.board_state[i][j] == 0):
                    legal_action.append(self.coord_to_action(i, j))
        return legal_action

    def play(self, action, color):
        '''
            Args: input action, current player color
            Return: new copy of board object
        '''
        coord = self.action_to_coord(action)

        # Duplicate board instance
        result_board = Board(self.size)
        result_board.board_state = np.copy(self.board_state)
        result_board.move = self.move

        result_board.board_state[coord[0]][coord[1]
                                           ] = gomoku_util.color_dict[color]
        result_board.move += 1  # move counter add 1
        result_board.last_coord = coord  # save last coordinate
        result_board.last_action = action
        return result_board

    def is_terminal(self):
        exist, color = gomoku_util.check_five_in_row(self.board_state)
        is_full = gomoku_util.check_board_full(self.board_state)
        if (is_full):  # if the board if full of stones and no extra empty spaces, game is finished
            return True
        else:
            return exist

    def __repr__(self):
        ''' representation of the board class
            print out board_state
        '''
        out = ""
        size = len(self.board_state)

        letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:size]
        numbers = list(range(1, 100))[:size]

        label_move = "Move: " + str(self.move) + "\n"
        label_letters = "     " + " ".join(letters) + "\n"
        label_boundry = "   " + "+-" + "".join(["-"] * (2 * size)) + "+" + "\n"

        # construct the board output
        out += (label_move + label_letters + label_boundry)

        for i in range(size - 1, -1, -1):
            line = ""
            line += (str("%2d" % (i + 1)) + " |" + " ")
            for j in range(size):
                # check if it's the last move
                line += gomoku_util.color_shape[self.board_state[i][j]]
                if (i, j) == self.last_coord:
                    line += ")"
                else:
                    line += " "
            line += ("|" + "\n")
            out += line
        out += (label_boundry + label_letters)
        return out
