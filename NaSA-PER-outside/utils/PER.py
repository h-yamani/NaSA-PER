import torch
import numpy as np
from collections import deque
import random
from .SumTree import SumTree

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PER:
    def __init__(self, max_capacity=int(1e6)):
        self.max_capacity = max_capacity
        self.buffer = deque([], maxlen=self.max_capacity)
        self.ptr = 0
        self.size = 0
        self.tree = SumTree(max_capacity)
        self.max_priority = 1.0
        self.beta = 0.4
        self.epsilon_d = 1e-6

        self.device = DEVICE

    def add(self, **experience):
        state = experience["state"]
        action = experience["action"]
        reward = experience["reward"]
        next_state = experience["next_state"]
        done = experience["done"]
        novelty_surprise = experience["novelty_surprise"]
        self.buffer.append([state, action, reward, next_state, done, novelty_surprise])
        self.tree.set(self.ptr, self.max_priority)
        self.ptr = (self.ptr + 1) % self.max_capacity
        self.size = min(self.size + 1, self.max_capacity)

    def sample(self, batch_size):
        ind = self.tree.sample(batch_size)

        weights = self.tree.levels[-1][ind] ** -self.beta
        weights /= weights.max()

        self.beta = min(self.beta + 2e-7, 1)

        # Assuming you have already defined self.buffer and ind
        #batch_size = len(ind)  # Get the batch size

        # Initialize lists to store the selected experiences
        selected_states = []
        selected_actions = []
        selected_rewards = []
        selected_next_states = []
        selected_dones = []
        selected_novelty_surprise = []

        # Loop through the indices and fetch the corresponding experiences
        for i in ind:
            experience = self.buffer[i]  # Get the experience tuple at index i

            # Unpack the experience tuple into individual components
            state, action, reward, next_state, done, novelty_surprise = experience

            # Append the components to the corresponding lists
            selected_states.append(state)
            selected_actions.append(action)
            selected_rewards.append(reward)
            selected_next_states.append(next_state)
            selected_dones.append(done)
            selected_novelty_surprise.append(novelty_surprise)

        # Convert the lists to NumPy arrays if needed
        states = np.array(selected_states)
        actions = np.array(selected_actions)
        rewards = np.array(selected_rewards)
        next_states = np.array(selected_next_states)
        dones = np.array(selected_dones)
        novelty_surprises = np.array(selected_novelty_surprise)

        # Now, you have the selected experiences in separate arrays/listss

        # Randomly sample experiences from buffer of size batch_size
        # experience_batch = random.sample(self.buffer, batch_size)

        # Destructure batch experiences into tuples of _
        # eg. tuples of states, tuples of actions...
        #states, actions, rewards, next_states, dones = zip(*experience_batch)

        return states, actions, rewards, next_states, dones, novelty_surprises, ind, weights

    def update_priority(self, ind, priority):
        self.max_priority = max(priority.max(), self.max_priority)
        self.tree.batch_set(ind, priority)

    def clear(self):
        self.buffer = deque([], maxlen=self.max_capacity)