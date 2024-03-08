import torch
import numpy as np
from .SumTree import SumTree

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from collections import deque
import random
import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = []
        self.action = []
        self.next_state = []
        self.reward = []
        self.done = []

        self.experience_deque = deque(maxlen=max_size)

        self.tree = SumTree(max_size)
        self.max_priority = 1.0
        self.beta = 0.4
        self.epsilon_d = 1e-6

        self.device = DEVICE

    def add(self, state, action, reward, next_state, done):
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        }

        self.state.append(state)
        self.action.append(action)
        self.next_state.append(next_state)
        self.reward.append(reward)
        self.done.append(done)

        self.experience_deque.append(experience)

        self.tree.set(self.ptr, self.max_priority)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        indices = self.tree.sample(batch_size)
        weights = (self.tree.levels[-1][indices] ** -self.beta)
        weights /= weights.max()
        self.beta = min(self.beta + 2e-7, 1)

        # Convert the lists of NumPy arrays to a single NumPy array
        state_batch = np.array([self.state[i] for i in indices])
        action_batch = np.array([self.action[i] for i in indices])
        reward_batch = np.array([self.reward[i] for i in indices])
        next_state_batch = np.array([self.next_state[i] for i in indices])
        done_batch = np.array([self.done[i] for i in indices])

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, indices, weights

        '''  return (
            torch.FloatTensor(state_batch).to(self.device),
            torch.FloatTensor(action_batch).to(self.device),
            torch.FloatTensor(reward_batch).to(self.device),
            torch.FloatTensor(next_state_batch).to(self.device),
            torch.FloatTensor(done_batch).to(self.device),
            indices,
            torch.FloatTensor(weights).to(self.device).reshape(-1, 1)
         )
         '''

    def update_priority(self, indices, priorities):
        self.max_priority = max(priorities.max(), self.max_priority)
        self.tree.batch_set(indices, priorities)

