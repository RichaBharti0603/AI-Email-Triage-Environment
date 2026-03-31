import random
from env.dataset import load_email_dataset
from env.state import build_state
from env.reward import compute_reward
from config import CONFIG


class EmailEnv:
    def __init__(self):
        self.data = load_email_dataset(CONFIG["dataset_size"])
        self.index = 0
        random.seed(CONFIG["seed"])

    def reset(self):
        self.index = 0
        return build_state(self.data[self.index])

    def step(self, action):
        current = self.data[self.index]

        reward = compute_reward(
            action,
            current,
            CONFIG["reward_weights"]
        )

        self.index += 1
        done = self.index >= len(self.data)

        next_state = None if done else build_state(self.data[self.index])

        info = {
            "ground_truth": current
        }

        return next_state, reward, done, info