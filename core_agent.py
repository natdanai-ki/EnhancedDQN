import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque

# ==========================================
# PART 1: Structures for Enhanced Agent (PER + Entropy)
# ==========================================

class SumTree:
    """Tree structure for Prioritized Experience Replay"""
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0: self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree): return idx
        if s <= self.tree[left]: return self._retrieve(left, s)
        else: return self._retrieve(right, s - self.tree[left])

    def total(self): return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity: self.write = 0
        if self.n_entries < self.capacity: self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 0.01

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0: max_p = 1.0
        self.tree.add(max_p, transition)

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            
            # Fallback for empty data
            if isinstance(data, int) and data == 0:
                 idx = 0 
                 while True:
                     (idx, p, data) = self.tree.get(random.uniform(0, self.tree.total()))
                     if not (isinstance(data, int) and data == 0): break
            
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weights /= is_weights.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), 
                np.array(next_states), np.array(dones, dtype=np.float32), idxs, np.array(is_weights, dtype=np.float32))

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (np.abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)

# ==========================================
# PART 2: Networks
# ==========================================

class DuelingQNetwork(nn.Module):
    """Network for Enhanced & Dueling Agents"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DuelingQNetwork, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

class SimpleQNetwork(nn.Module):
    """Network for DQN & DDQN Agents"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(SimpleQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    def forward(self, x): return self.net(x)

# ==========================================
# PART 3: Agents
# ==========================================

class EnhancedDQNAgent:
    """The Proposed Agent (Double + Dueling + PER + Entropy)"""
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, buffer_size=100000, batch_size=64, tau=1e-3, entropy_coef=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.entropy_coef = entropy_coef
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.qnetwork_local = DuelingQNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(buffer_size)
        self.beta = 0.4
        self.beta_increment = 0.00001

    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad(): action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps: return np.argmax(action_values.cpu().data.numpy())
        else: return random.choice(np.arange(self.action_size))

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def learn(self):
        if self.memory.tree.n_entries < self.batch_size: return 0.0
        states, actions, rewards, next_states, dones, idxs, weights = self.memory.sample(self.batch_size, self.beta)
        
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().unsqueeze(1).to(self.device)
        weights = torch.from_numpy(weights).float().unsqueeze(1).to(self.device)

        # Double DQN Logic
        best_actions = self.qnetwork_local(next_states).argmax(1).unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, best_actions)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # PER Update
        td_errors = torch.abs(Q_targets - Q_expected).detach().cpu().numpy()
        self.memory.update_priorities(idxs, td_errors)

        # Entropy
        probs = torch.softmax(self.qnetwork_local(states), dim=1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=1).mean()

        loss = (weights * (Q_expected - Q_targets) ** 2).mean() - (self.entropy_coef * entropy)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft Update
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        return loss.item()

    # --- Train Loop for CLI ---
    def train(self, env_name, episodes=2000, seed=0):
        import gymnasium as gym
        try: from envs_split_ac import SplitACEnv
        except ImportError: pass

        if env_name in ["SplitAC-v1", "SmartHVAC-v1"]: env = SplitACEnv()
        else: env = gym.make(env_name)

        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        env.reset(seed=seed)
        
        rewards, losses = [], []
        eps = 1.0
        eps_end = 0.01
        eps_decay = 0.995

        for i_episode in range(1, episodes+1):
            state, _ = env.reset()
            score = 0
            ep_losses = []
            while True:
                action = self.act(state, eps)
                next_state, reward, done, truncated, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                loss = self.learn()
                state = next_state
                score += reward
                if loss != 0: ep_losses.append(loss)
                if done or truncated: break
            
            eps = max(eps_end, eps * eps_decay)
            rewards.append(score)
            losses.append(np.mean(ep_losses) if ep_losses else 0)
            if i_episode % 100 == 0:
                print(f"Episode {i_episode}/{episodes} | Avg Reward: {np.mean(rewards[-100:]):.2f}")
        return rewards, losses

# ==========================================
# PART 4: Standard Agents (Dueling, DQN, DDQN) for Baselines
# ==========================================

class BaseDQNAgent:
    """Base Agent for DQN, DDQN, Dueling (Standard Replay Buffer, No Entropy)"""
    def __init__(self, state_size, action_size, dueling=False, double=False, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.double = double
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.99
        self.batch_size = 64
        self.tau = 1e-3
        
        random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)

        if dueling:
            self.qnetwork_local = DuelingQNetwork(state_size, action_size).to(self.device)
            self.qnetwork_target = DuelingQNetwork(state_size, action_size).to(self.device)
        else:
            self.qnetwork_local = SimpleQNetwork(state_size, action_size).to(self.device)
            self.qnetwork_target = SimpleQNetwork(state_size, action_size).to(self.device)
        
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=1e-3)
        self.memory = deque(maxlen=100000)

    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad(): action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps: return np.argmax(action_values.cpu().data.numpy())
        else: return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.batch_size: return self.learn()
        return 0.0

    def learn(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

        # Target Calculation
        if self.double:
            best_actions = self.qnetwork_local(next_states).argmax(1).unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).gather(1, best_actions)
        else:
            Q_targets_next = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1)
            
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft Update
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        return loss.item()

    # Generic Train Loop for Baselines
    def train(self, env_name, episodes=2000, seed=0):
        import gymnasium as gym
        env = gym.make(env_name)
        env.reset(seed=seed)
        rewards, losses = [], []
        eps = 1.0
        eps_end = 0.01
        eps_decay = 0.995

        for i in range(1, episodes+1):
            state, _ = env.reset()
            score = 0
            ep_losses = []
            while True:
                action = self.act(state, eps)
                next_state, reward, done, truncated, _ = env.step(action)
                loss = self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if loss != 0: ep_losses.append(loss)
                if done or truncated: break
            
            eps = max(eps_end, eps * eps_decay)
            rewards.append(score)
            losses.append(np.mean(ep_losses) if ep_losses else 0)
            if i % 100 == 0: print(f"Episode {i}/{episodes} | Avg Reward: {np.mean(rewards[-100:]):.2f}")
        return rewards, losses

# Wrapper Classes for easy instantiation
class DuelingDQNAgent(BaseDQNAgent):
    def __init__(self, state_size, action_size, seed=0):
        super().__init__(state_size, action_size, dueling=True, double=False, seed=seed)

class DQNAgent(BaseDQNAgent):
    def __init__(self, state_size, action_size, seed=0):
        super().__init__(state_size, action_size, dueling=False, double=False, seed=seed)

class DDQNAgent(BaseDQNAgent):
    def __init__(self, state_size, action_size, seed=0):
        super().__init__(state_size, action_size, dueling=False, double=True, seed=seed)