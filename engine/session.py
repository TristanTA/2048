import random
from collections import Counter

from engine.square import Square

class Session:
    def __init__(self):
        self.x_grid = 4
        self.y_grid = 4
        self.values = None
        self.alive = True
        self.score = 0
        self.add_square()
        self.add_square()

    def add_square(self):
        if not self.values: 
            square = Square()
            self.values = []
            square.position = random.choice(range(self.x_grid * self.y_grid))
            square.x, square.y = self.get_coordinates(square.position)
            self.values.append(square)
        else:
            open_pos = self.get_open_position()
            square = Square()
            square.position = open_pos
            square.x, square.y = self.get_coordinates(open_pos)
            self.values.append(square)

    def get_open_position(self):
        if self.alive:
            occupied_squares = []
            for square in self.values:
                position = self.get_position(square=square) 
                occupied_squares.append(position)
            if len(occupied_squares) >= self.x_grid * self.y_grid:
                print("No open positions available.")
                print("Game Over!")
                self.alive = False
                return -1
            all_squares = []
            for i in range(self.x_grid * self.y_grid):
                all_squares.append(i)
            for pos in set(occupied_squares):
                if pos in all_squares:
                    all_squares.remove(pos)
            else:
                open_position = random.choice(all_squares)
                return open_position

    def get_position(self, square: Square) -> int:
        position = (
            square.x + square.y * self.x_grid
        )
        return position
    
    def get_all_positions(self):
        positions = []
        for square in self.values:
            position = self.get_position(square)
            positions.append(position)
        return positions

    def get_coordinates(self, position):
        x = position % self.x_grid
        y = position // self.x_grid
        return x, y
    
    def display_grid(self):
        board = {}
        for sq in self.values:
            pos = sq.y * self.x_grid + sq.x
            board[pos] = sq.value

        grid = []
        for i in range(self.x_grid * self.y_grid):
            if i in board:
                grid.append(board[i])
            else:
                grid.append("-")

        for row_start in range(0, len(grid), self.x_grid):
            row = grid[row_start : row_start + self.x_grid]
            print(row)

    def step(self, move):
        if not self.alive:
            return
        if int(move) in [0, 1, 2, 3]:
            if move == 0: # Up
                self.move_squares_up()
            elif move == 1: # Left
                self.move_squares_left()
            elif move == 2: # Down
                self.move_squares_down()
            elif move == 3: # Right
                self.move_squares_right()
            self.combine_squares()
            self.add_square()
            print(f"Score: {self.get_score()}")
            if self.check_game_over():
                self.alive = False
        else:
            print("Invalid move. Use 0 (Up), 1 (Left), 2 (Down), or 3 (Right).")

    def get_square_at(self, x, y):
        for sq in self.values:
            if sq.x == x and sq.y == y:
                return sq
        return None

    def move_squares_up(self):
        self.values.sort(key=lambda s: s.y)
        for square in self.values:
            while True:
                next_y = square.y - 1
                if next_y < 0:
                    break

                blocker = self.get_square_at(square.x, next_y)

                if blocker is None:
                    square.y = next_y

                else:
                    if blocker.value == square.value:
                        square.y = next_y
                    break

    def move_squares_left(self):
        self.values.sort(key=lambda s: s.x)

        for square in self.values:
            while True:
                next_x = square.x - 1
                if next_x < 0:
                    break

                blocker = self.get_square_at(next_x, square.y)

                if blocker is None:
                    square.x = next_x
                else:
                    if blocker.value == square.value:
                        square.x = next_x
                    break

    def move_squares_down(self):
        self.values.sort(key=lambda s: s.y, reverse=True)

        for square in self.values:
            while True:
                next_y = square.y + 1
                if next_y >= self.y_grid:
                    break

                blocker = self.get_square_at(square.x, next_y)

                if blocker is None:
                    square.y = next_y
                else:
                    if blocker.value == square.value:
                        square.y = next_y
                    break

    def move_squares_right(self):
        self.values.sort(key=lambda s: s.x, reverse=True)

        for square in self.values:
            while True:
                next_x = square.x + 1
                if next_x >= self.x_grid:
                    break

                blocker = self.get_square_at(next_x, square.y)

                if blocker is None:
                    square.x = next_x
                else:
                    if blocker.value == square.value:
                        square.x = next_x 
                    break
    
    def combine_squares(self):
        position_map = {}
        for sq in self.values:
            pos = (sq.x, sq.y)
            if pos not in position_map:
                position_map[pos] = []
            position_map[pos].append(sq)

        new_values = []

        for pos, squares in position_map.items():
            if len(squares) == 1:
                # keep tile unchanged
                new_values.append(squares[0])

            elif len(squares) == 2:
                sq1, sq2 = squares
                if sq1.value == sq2.value:
                    sq1.value *= 2
                    new_values.append(sq1)
                else:
                    new_values.extend(squares)

            else:
                squares[0].value *= 2
                new_values.append(squares[0])
                new_values.extend(squares[1:])

        self.values = new_values

    def check_game_over(self):
        if len(self.values) < self.x_grid * self.y_grid:
            return False

        for sq in self.values:
            right = self.get_square_at(sq.x + 1, sq.y)
            if right and right.value == sq.value:
                return False

        for sq in self.values:
            down = self.get_square_at(sq.x, sq.y + 1)
            if down and down.value == sq.value:
                return False

        return True

    def get_score(self):
        self.score = 0
        for sq in self.values:
            self.score += sq.value