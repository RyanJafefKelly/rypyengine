import numpy as np
from collections import namedtuple
from itertools import count
import re

# TODO: smarter number values
piece = {'P': 100, 'N': 300, 'B': 300, 'R': 500, 'Q': 900, 'K': 60000}
#TODO: piece-square table from sunfish... add opening + end game
pst = {
    'P': [0,   0,   0,   0,   0,   0,   0,   0,
          78,  83,  86,  73, 102,  82,  85,  90,
          7,  29,  21,  44,  40,  31,  44,   7,
          -17,  16,  -2,  15,  14,   0,  15, -13,
          -26,   3,  10,   9,   6,   1,   0, -23,
          -22,   9,   5, -11, -10,  -2,   3, -19,
          -31,   8,  -7, -37, -36, -14,   3, -31,
          0,   0,   0,   0,   0,   0,   0,   0],
    'N': [ -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69],
    'B': [ -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10],
    'R': [  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32],
    'Q': [  6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42],
    'K': [  4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18],
}

# Pad tables and join piece and pst dictionaries
# board represented as 10x12 mailbox board


for k, table in pst.items():
        # TODO: got to be a better + faster method
        tmp_table = np.array([])
        bounds_end = [0] * 20
        tmp_table = np.append(tmp_table, bounds_end)
        for i in range(8):
                row = [0] + table[(i*8):(i*8)+8] + [0]
                tmp_table = np.append(tmp_table, row)
        tmp_table = np.append(tmp_table, bounds_end)
        pst[k] = tmp_table.flatten()
        # print('test', test)

## GLOBAL CONSTANTS
A8, H8, A1, H1 = 21, 28, 91, 98  # white at "bottom"
# lowercase - black, UPPER - white, letters - explanatory, \n - empty
init_board = ['FF'] * 20 + ['FF'] + ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'] + \
             ['FF'] + ['FF'] + 8*['p'] + ['FF']  + \
             4*(['FF'] + 8*['o'] + ['FF']) + \
             ['FF'] + 8*['P'] + ['FF'] + \
             ['FF'] + ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'] + ['FF'] + \
             20*['FF']

# Lists of possible moves for each      piece type.
up, down, left, right = -10, 10, -1, 1 # numbers referring to change idx
directions = {
    'P': (up, up+up, up+left, up+right),
    'R': (up, down, left, right),
    'N': (up+up+left, up+up+right, right+right+up, right+right+down,
          down+down+right, down+down+left, left+left+down, left+left+up),
    'B': (up+left, up+right, down+left, down+right),
    'K': (up+left, up, up+right, left, right, down+left, down, down+right),
    'Q': (up+left, up, up+right, left, right, down+left, down, down+right),
    'FF': ()
}

# TODO: curr copied sunfish
# Mate value must be greater than 8*queen + 2*(rook+knight+bishop)
# King value is set to twice this value such that if the opponent is
# 8 queens up, but we got the king, we still exceed MATE_VALUE.
# When a MATE is detected, we'll set the score to MATE_UPPER - plies to get there
# E.g. Mate in 3 will be MATE_UPPER - 6
MATE_LOWER = piece['K'] - 10*piece['Q']
MATE_UPPER = piece['K'] + 10*piece['Q']

# TODO: Include below constants?
# Table size
# TABLE_SIZE = 1e7
# Constants tuning search
# QS_LIMIT = 219
# EVAL_ROUGHNESS = 13
# DRAW_TEST = True

# Chess Logic
# class Position
class Position(namedtuple('Position',
'board score castling opp_castling en_passant')):
    """[summary]

    Args:
        board ([type]): [description]
        score ([type]): [description]
        castling ([type]): [description]
        opp_castling ([type]): [description]
        en_passant ([type]): [description]
        king_passant ([type]): [description]
    """

    def generate_moves(self):
        moves = []
        for i, p in enumerate(self.board):
            if not p.isupper():
                continue
            if p is 'FF':
                continue
            for d in directions[p]:
                for j in count(i+d, d):
                    q = self.board[j]
                    # Stay inside the board, and off friendly pieces
                    if q.isupper() or q == 'FF':
                        break
                    # Pawn move, double move and capture
                    if p == 'P' and d in (up, up+up) and q != 'o':
                        break
                    if p == 'P' and d == up+up and (i < A1 + up or self.board[i+up] != 'o'):
                        break
                    if p == 'P' and d in (up+left, up+right) and q == 'o':
                        break
                    moves.append((i, j))
                    # Stop at pieces move w/o/ sliding and captures
                    if p in 'PNK' or q.islower():
                        break
                    # Castling

        return moves
                    
    def rotate(self):
        pass

    def null_move(self):
        pass

    def move(self, move):
        x, y = move
        p, q = self.board[x], self.board[y]
        print('x', x, 'y', y)
        print('p', p, 'q', q)
        
        def put(board, x, p):
            return board[:x] + [p] + board[x+1:]  # insert
        
        board = self.board
        # TODO?   Copy variables and reset ep and kp
        castling, opp_castling, en_passant = self.castling, self.opp_castling, 0
        score = self.score + self.value(move)

        # Actual move logic
        board = put(board, y, board[x])
        board = put(board, x, 'o')

        # Castling rights, we move the rook or capture the opponent's
        if x == A1:
            castling = (False, opp_castling[1])
        if x == H1:
            castling = (opp_castling[0], False)
        if y == A8:
            opp_castling = (castling[0], False)
        if y == H8:
            opp_castling = (False, castling[1])
        # TODO: Castling logic
        if p == 'K':
            opp_castling = (False, False)
        # TODO: Pawn Promotion
        return Position(board, score, castling, opp_castling, en_passant)

    def value(self, move):
        return 0


# Search logic
# class Searcher
class Searcher:
    def __init__(self):
        pass

    def bound(self, pos, gamma, depth, root=True):
        pass

        def moves():
            pass

    def search(self, pos, history=()):
        pass



# User interface

def print_board(board):
    print('board', board)
    # board is 120 char
    def chunks(lst, n):
        """stackoverflow...split list evenly sized chunks

        Args:
            lst ([type]): [description]
            n ([type]): [description]
        """
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    reduced_board = board[20:100] # remove first and last 2  sentinel rows 
    chess_symbol_dict = {
        'FF': '', # sentinel val
        'R': '♜ ',
        'N': '♞ ',
        'B': '♝ ',
        'Q': '♛ ',
        'K': '♚ ',
        'P': '♟︎ ',
        'r': '♖ ',
        'n': '♘ ',
        'b': '♗ ',
        'q': '♕ ',
        'k': '♔ ',
        'p': '♙ ',
        'o': '. ' # blank on board
    }
    for ii, row in enumerate(list(chunks(reduced_board, 10))):
        # incredible unicode includes chess pieces
        print(8-ii, end='  ')
        for el in row:  # double for loop but really doesn't matter for this...
            print(chess_symbol_dict[el], sep=' ', end='')
        print('')
    print('   a b c d e f g h')
    return


def parse_to_index(c):
    file, rank = ord(c[0]) - ord('a'), int(c[1]) - 1
    idx = A1 + file - 10 * rank
    return idx

def main():

    # print_board(init_board)
    moves = [Position(board=init_board, score=0, castling=[True, True],
    opp_castling=[True, True], en_passant=0)]
    searcher = Searcher()
    while True:
        print_board(moves[-1].board)
        if moves[-1].score <= -MATE_LOWER:
            print("You lose!")

            break

        # query until enter a pseudo-legal move
        move_list = moves[-1].generate_moves()
        move = None
        while move not in move_list:
            match = re.match('([a-h][1-8])'*2, input('Your move: '))
            if match:
                move = parse_to_index(match.group(1)), parse_to_index(match.group(2))
                if move not in move_list:
                    print("Not a valid move")
            else:
                print("Please enter a move like c2c3")

        moves.append(moves[-1].move(move))
        print_board(moves[-1].board)

        # after move rotate board and print to see effect of move

        # Search for a move

        # rotate back to display

        break
    pass

if __name__ == '__main__':
    main()

