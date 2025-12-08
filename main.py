

from engine.session import Session

def main():
    session = Session()
    while session.alive:
        session.display_grid()
        user_input = int(input("Move: "))
        session.step(move = user_input)

if __name__ == "__main__":
    main()