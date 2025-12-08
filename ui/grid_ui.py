import pygame

from engine.session import Session
from ui.colors import gold, charcoal

class GridUI:
    def __init__(self, session: Session, scale = 150):
        self.scale = scale
        self.x_grid = session.x_grid
        self.y_grid = session.y_grid
        self.w = self.x_grid * scale
        self.h = self.y_grid * scale
        self.session = session

        pygame.init()
        self.window = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()

    def draw(self):
        self.window.fill(gold)
        grid_centers = self.get_grids()
        values = self.get_values(session=self.session)
        for i in range(self.x_grid * self.y_grid):
            x, y = grid_centers[i]
            rect = pygame.Rect(
                x - self.scale // 2 + 5,
                y - self.scale // 2 + 5,
                self.scale - 10,
                self.scale - 10,
            )
            pygame.draw.rect(self.window, charcoal, rect, border_radius=10)
            if i in values:
                font = pygame.font.Font(None, 74)
                text = font.render(str(values[i]), True, gold)
                text_rect = text.get_rect(center=(x, y))
                self.window.blit(text, text_rect)
        pygame.display.update()

    def get_grids(self):
        grid_centers = {}
        for i in range(self.x_grid * self.y_grid):
            x = (i % self.x_grid) * self.scale + self.scale // 2
            y = (i // self.x_grid) * self.scale + self.scale // 2
            grid_centers[i] = (x, y)
        return grid_centers
    
    def get_values(self, session:Session):
        values = {}
        for square in session.values:
            position = session.get_position(square)
            values[position] = square.value
        return values
    
    def get_input(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        return 0
                    if event.key == pygame.K_DOWN:
                        return 2
                    if event.key == pygame.K_LEFT:
                        return 1
                    if event.key == pygame.K_RIGHT:
                        return 3
            pygame.time.wait(10)