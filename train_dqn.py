import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from environment import EmailTriageEnv

# Simple MLP for DQN
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = QNetwork(state_dim, action_dim).to('cpu') # Force CPU
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def train_step(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.model(next_state_t)).item()
            
            state_t = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state_t)
            target_f[0][action] = target
            
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state_t), target_f)
            loss.backward()
            self.optimizer.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Mock State Encoder (since the real one uses DistilBART)
def get_mock_state(observation):
    # Convert text to a fixed-size vector for demo
    # In a real system, use an embedding model
    return np.random.rand(16).astype(np.float32)

def train_agent(episodes=50):
    env = EmailTriageEnv()
    state_dim = 16 # Mock embedding size
    action_dim = 3 # Category actions (Urgent, Normal, Spam)
    agent = DQNAgent(state_dim, action_dim)
    
    print(f"Starting Training ({episodes} episodes on CPU)...")
    
    for e in range(episodes):
        obs, _ = env.reset()
        state = get_mock_state(obs)
        total_reward = 0
        
        for time in range(len(env.email_database)):
            action_idx = agent.act(state)
            
            # Map discrete action to multi-objective dict for env
            action_map = {0: 'urgent', 1: 'normal', 2: 'spam'}
            action_dict = {
                'category': action_map[action_idx],
                'priority': 'Medium', # Fixed for basic training
                'department': 'Tech'   # Fixed for basic training
            }
            
            next_obs, reward, done, _, _ = env.step(action_dict)
            next_state = get_mock_state(next_obs) if not done else state
            agent.memory.append((state, action_idx, reward, next_state, done))
            state = next_state
            total_reward += reward
            
            if done:
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break
        
        agent.train_step()
    
    print("Training finished. (Decoupled experiment complete)")

if __name__ == "__main__":
    train_agent()
