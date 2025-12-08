

from engine.session import Session
from ui.grid_ui import GridUI

def main():
    session = Session()
    grid = GridUI(session=session)
    while session.alive:
        grid.draw()
        move = grid.get_input()
        session.step(move=move)

if __name__ == "__main__":
    main()