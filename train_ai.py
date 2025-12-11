import torch
import torch.nn as nn
import torch.optim as optim
import random
import time

from engine.session import Session
from model.transformer import Transformer2048
from model.replay_buffer import ReplayBuffer


def train():
    model = Transformer2048()
    target_model = Transformer2048()
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    buffer = ReplayBuffer()

    episodes = 20000
    batch_size = 64
    gamma = 0.50
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    average_episode = []

    for episode in range(episodes):
        start_time = time.time()
        session = Session()
        total_reward = 0

        while session.alive:
            state = model.embedder.get_value_id_tensor(session)   # [1,16]

            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                q_values = model(state)
                action = torch.argmax(q_values).item()

            moved = session.step(action)

            reward = session.score
            total_reward += reward
            if moved is False:
                total_reward -= 500

            next_state = model.embedder.get_value_id_tensor(session)

            buffer.push(state.squeeze(0), action, reward,
                        next_state.squeeze(0), not session.alive)

            if len(buffer) > batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                q_pred = model(states)
                q_pred = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q = target_model(next_states).max(1)[0]
                    q_target = rewards + gamma * next_q * (1 - dones)

                loss = criterion(q_pred, q_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % 50 == 0:
            target_model.load_state_dict(model.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        end_time = time.time()
        elapsed = end_time - start_time

        print(f"Episode {episode} | Score: {total_reward} | Eps: {epsilon:.3f} | Time: {elapsed:.2f}s")
        average_episode.append(elapsed)
        if episode % 10 == 0:
            remaining_episodes = episodes - (episode + 1)
            estimated_time = sum(average_episode) / len(average_episode) * remaining_episodes
            print(f"Estimated remaining time: {estimated_time/60:.2f} minutes")

    torch.save(model.state_dict(), "model_2048.pth")


if __name__ == "__main__":
    train()