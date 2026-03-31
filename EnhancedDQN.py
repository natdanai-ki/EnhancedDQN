import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class BayesianDuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, 256)
        self.drop2 = nn.Dropout(p=dropout_rate)

        self.value = nn.Linear(256, 1)
        self.adv = nn.Linear(256, action_dim)

    def forward(self, x: torch.Tensor, train_mode: bool = True) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        if train_mode:
            x = self.drop1(x)
        x = torch.relu(self.fc2(x))
        if train_mode:
            x = self.drop2(x)

        v = self.value(x)
        a = self.adv(x)
        return v + (a - a.mean(dim=1, keepdim=True))


@dataclass
class AgentConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 128
    replay_size: int = 100_000
    per_alpha: float = 0.6
    per_beta: float = 0.4
    per_eps: float = 1e-6
    softmax_temp: float = 0.5
    n_step: int = 3
    dropout_rate: float = 0.1


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-6):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.memory: List[Optional[Tuple[np.ndarray, int, float, np.ndarray, bool]]] = [None] * self.capacity
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool], priority: Optional[float] = None) -> None:
        if priority is None:
            max_prio = float(self.priorities[: self.size].max()) if self.size > 0 else 1.0
            priority = max(max_prio, 1.0)

        self.memory[self.pos] = transition
        self.priorities[self.pos] = float(priority)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float):
        if self.size < batch_size:
            return None

        prios = self.priorities[: self.size]
        scaled = np.power(np.maximum(prios, self.eps), self.alpha)
        probs = scaled / scaled.sum()

        indices = np.random.choice(self.size, size=batch_size, replace=False, p=probs)
        batch = [self.memory[idx] for idx in indices]
        assert all(item is not None for item in batch)

        weights = np.power(self.size * probs[indices], -beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, td_errors: np.ndarray) -> None:
        td_errors = np.abs(np.asarray(td_errors, dtype=np.float32)) + self.eps
        for idx, err in zip(indices, td_errors):
            self.priorities[int(idx)] = float(err)


class EnhancedDQNAgent:
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig = AgentConfig()):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = int(action_dim)
        self.cfg = config

        self.q_net = BayesianDuelingQNetwork(state_dim, action_dim, dropout_rate=self.cfg.dropout_rate).to(self.device)
        self.target_net = BayesianDuelingQNetwork(state_dim, action_dim, dropout_rate=self.cfg.dropout_rate).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.cfg.lr)
        self.memory = PrioritizedReplayBuffer(
            capacity=self.cfg.replay_size,
            alpha=self.cfg.per_alpha,
            eps=self.cfg.per_eps,
        )
        self.global_step = 0
        self.n_step_buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=self.cfg.n_step)

    def _softmax_action(self, q_values: torch.Tensor) -> int:
        probs = torch.softmax(q_values / self.cfg.softmax_temp, dim=1)
        action = torch.multinomial(probs, num_samples=1)
        return int(action.item())

    def select_action(self, state: np.ndarray, train: bool = True) -> int:
        self.global_step += 1
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(s, train_mode=train)
            if train:
                return self._softmax_action(q)
            return int(torch.argmax(q, dim=1).item())

    def _get_n_step_transition(self):
        reward = 0.0
        done = False
        next_state = self.n_step_buffer[-1][3]

        for i, (_, _, r, ns, d) in enumerate(self.n_step_buffer):
            reward += (self.cfg.gamma ** i) * float(r)
            next_state = ns
            if d:
                done = True
                break

        state, action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
        return state, action, reward, next_state, done

    def store_transition(self, s, a, r, ns, d):
        transition = (
            np.asarray(s, dtype=np.float32),
            int(a),
            float(r),
            np.asarray(ns, dtype=np.float32),
            bool(d),
        )
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) >= self.cfg.n_step:
            self.memory.add(self._get_n_step_transition())

        if d:
            while len(self.n_step_buffer) > 1:
                self.n_step_buffer.popleft()
                self.memory.add(self._get_n_step_transition())
            self.n_step_buffer.clear()

    def update(self):
        batch = self.memory.sample(self.cfg.batch_size, beta=self.cfg.per_beta)
        if batch is None:
            return None

        states, actions, rewards, next_states, dones, indices, weights = batch

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        gamma_n = self.cfg.gamma ** self.cfg.n_step

        with torch.no_grad():
            next_q_online = self.q_net(next_states, train_mode=False)
            next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
            next_q_target = self.target_net(next_states, train_mode=False).gather(1, next_actions)
            target_q = rewards + gamma_n * next_q_target * (1.0 - dones)

        current_q = self.q_net(states, train_mode=True).gather(1, actions)
        td_error = target_q - current_q
        loss = torch.mean(weights * td_error.pow(2))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.memory.update_priorities(indices, td_error.detach().squeeze(1).cpu().numpy())
        return float(loss.item())

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
