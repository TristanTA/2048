import numpy as np
import random


class Session:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=np.int64)
        self.score = 0
        self.alive = True

        self.add_random_tile()
        self.add_random_tile()

    def add_random_tile(self):
        empty = np.argwhere(self.board == 0)
        if empty.size == 0:
            self.alive = False
            return

        y, x = empty[random.randint(0, len(empty) - 1)]
        self.board[y, x] = 4 if random.random() < 0.1 else 2

    def slide_and_merge_line(self, line):
        nonzero = line[line != 0]

        merged = []
        score_gain = 0
        skip = False

        for i in range(len(nonzero)):
            if skip:
                skip = False
                continue

            if i + 1 < len(nonzero) and nonzero[i] == nonzero[i + 1]:
                new_val = nonzero[i] * 2
                merged.append(new_val)
                score_gain += new_val
                skip = True
            else:
                merged.append(nonzero[i])

        merged = np.array(merged, dtype=np.int64)
        # Pad with zeros
        if len(merged) < 4:
            merged = np.pad(merged, (0, 4 - len(merged)), 'constant')

        return merged, score_gain

    def move_left(self):
        new_board = np.zeros((4, 4), dtype=np.int64)
        total_gain = 0

        for y in range(4):
            merged_row, gain = self.slide_and_merge_line(self.board[y])
            new_board[y] = merged_row
            total_gain += gain

        self.board = new_board
        self.score += total_gain

    def move_right(self):
        new_board = np.zeros((4, 4), dtype=np.int64)
        total_gain = 0

        for y in range(4):
            reversed_row = self.board[y][::-1]
            merged_row, gain = self.slide_and_merge_line(reversed_row)
            new_board[y] = merged_row[::-1]
            total_gain += gain

        self.board = new_board
        self.score += total_gain

    def move_up(self):
        new_board = np.zeros((4, 4), dtype=np.int64)
        total_gain = 0

        for x in range(4):
            col = self.board[:, x]
            merged_col, gain = self.slide_and_merge_line(col)
            new_board[:, x] = merged_col
            total_gain += gain

        self.board = new_board
        self.score += total_gain

    def move_down(self):
        new_board = np.zeros((4, 4), dtype=np.int64)
        total_gain = 0

        for x in range(4):
            col = self.board[::-1, x]
            merged_col, gain = self.slide_and_merge_line(col)
            new_board[:, x] = merged_col[::-1]
            total_gain += gain

        self.board = new_board
        self.score += total_gain

    def step(self, move):
        if not self.alive:
            return

        prev_board = self.board.copy()

        if move == 0:
            self.move_up()
        elif move == 1:
            self.move_left()
        elif move == 2:
            self.move_down()
        elif move == 3:
            self.move_right()

        moved = False
        if not np.array_equal(prev_board, self.board):
            self.add_random_tile()
            moved = True

        if self.check_game_over():
            self.alive = False
        return moved

    def check_game_over(self):
        if np.any(self.board == 0):
            return False

        if np.any(self.board[:, :-1] == self.board[:, 1:]):
            return False

        if np.any(self.board[:-1, :] == self.board[1:, :]):
            return False

        return True