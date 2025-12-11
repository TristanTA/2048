import torch
import time
import pygame

from model.transformer import Transformer2048
from engine.session import Session
from ui.grid_ui import GridUI

def watch():
    model = Transformer2048()
    model.load_state_dict(torch.load("model_2048.pth"))
    model.eval()

    session = Session()
    grid = GridUI(session)
    while session.alive:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        grid.draw()

        state = model.embedder.get_value_id_tensor(session)

        with torch.no_grad():
            q_values = model(state)
            action = torch.argmax(q_values).item()

        session.step(action)
        print(action, session.score)

        time.sleep(0.1)


if __name__ == "__main__":
    watch()