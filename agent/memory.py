import numpy as np


class PPOMemory:
    def __init__(self, batch_size):
        self.features = []
        self.masks = []
        self.actions = []
        self.rewards = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_features = len(self.features)
        batch_start = np.arange(0, n_features, self.batch_size)
        indices = np.arange(n_features, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.features), \
            np.array(self.masks), \
            np.array(self.actions), \
            np.array(self.rewards), \
            batches

    def __len__(self):
        return len(self.features)

    def store_memory(self, features, mask, action, reward):
        self.features += [features.tolist()]
        self.masks += [mask.tolist()]
        self.actions += [action.tolist()]
        self.rewards += [reward]

    def get_memory(self):
        return self.features, self.masks, self.actions, self.rewards

    def clear_memory(self):
        self.features = []
        self.masks = []
        self.actions = []
        self.rewards = []
