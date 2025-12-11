import pygame
from engine.session import Session
from ui.colors import gold, charcoal

class GridUI:
    def __init__(self, session: Session, scale=150):
        self.scale = scale
        self.session = session

        self.w = 4 * scale
        self.h = 4 * scale

        pygame.init()
        self.window = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.Font(None, 72)

    def draw(self):
        self.window.fill(gold)

        for y in range(4):
            for x in range(4):
                value = self.session.board[y, x]

                px = x * self.scale
                py = y * self.scale

                rect = pygame.Rect(
                    px + 5, py + 5,
                    self.scale - 10, self.scale - 10
                )
                pygame.draw.rect(self.window, charcoal, rect, border_radius=10)

                # Draw tile number
                if value != 0:
                    text = self.font.render(str(value), True, gold)
                    text_rect = text.get_rect(
                        center=(px + self.scale // 2, py + self.scale // 2)
                    )
                    self.window.blit(text, text_rect)

        pygame.display.update()

    def get_input(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        return 0
                    if event.key == pygame.K_LEFT:
                        return 1
                    if event.key == pygame.K_DOWN:
                        return 2
                    if event.key == pygame.K_RIGHT:
                        return 3

            pygame.time.wait(10)