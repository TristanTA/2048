import torch
import random

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = []
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.stack(states),
                torch.tensor(actions),
                torch.tensor(rewards, dtype=torch.float),
                torch.stack(next_states),
                torch.tensor(dones, dtype=torch.float))

    def __len__(self):
        return len(self.buffer)