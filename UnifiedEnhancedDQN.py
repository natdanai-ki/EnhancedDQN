import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import pandas as pd
from collections import deque
import os

# =========================================================
# 1. Bayesian Neural Network (MC Dropout)
# กลไก: ใช้ Dropout ระหว่างรันเพื่อประมาณค่าความไม่แน่นอน (Uncertainty)
# ประโยชน์ใน HVAC: ช่วยให้ AI ตัดสินใจอย่างระมัดระวังเมื่อเซนเซอร์แกว่งจากฝุ่น PM10
# =========================================================
class BayesianQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, dropout_rate=0.1):
        super(BayesianQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.dropout = nn.Dropout(p=dropout_rate) 
        self.fc2 = nn.Linear(256, 256)
        
        # Dueling Architecture: แยก Value (สภาวะห้อง) และ Advantage (ความคุ้มค่าของ Action)
        self.value_stream = nn.Linear(256, 1)
        self.advantage_stream = nn.Linear(256, action_dim)

    def forward(self, state, train_mode=True):
        # เปิด Dropout ตลอดเวลา (Monte Carlo Dropout) เพื่อสร้าง Uncertainty
        x = torch.relu(self.fc1(state))
        if train_mode: x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        if train_mode: x = self.dropout(x)
        
        val = self.value_stream(x)
        adv = self.advantage_stream(x)
        # สูตร Dueling: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return val + (adv - adv.mean(dim=1, keepdim=True))

# =========================================================
# 2. Prioritized Experience Replay (PER)
# กลไก: เก็บประสบการณ์ที่มี Error สูงไว้เรียนรู้ซ้ำ
# ประโยชน์ใน HVAC: เน้นเรียนรู้จากช่วงที่แอร์กินไฟผิดปกติเนื่องจากกรองเริ่มตัน
# =========================================================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size: return None
        
        prios = self.priorities[:len(self.buffer)]
        probs = (prios + 1e-6) ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

# =========================================================
# 3. Enhanced DQN Agent
# รวบรวมเทคนิค: Bayesian, PER, Multi-step, และ Entropy
# =========================================================
class EnhancedDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        
        # Networks Setup
        self.q_net = BayesianQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = BayesianQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=3e-4)
        self.memory = PrioritizedReplayBuffer(100000)
        
        # Hyperparameters
        self.gamma = 0.99
        self.n_step = 3            # Tech 3: Multi-step returns (มองการณ์ไกล 3 ก้าว)
        self.entropy_alpha = 0.01  # Tech 4: Entropy Regularization (รักษาความหลากหลายของ Action)
        self.batch_size = 128
        self.n_step_buffer = deque(maxlen=self.n_step)

    def select_action(self, state):
        # ปรับมิติข้อมูลให้รองรับ Batch [1, state_dim]
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # ใช้ Bayesian Uncertainty ในการช่วยสำรวจ
            q_values = self.q_net(state, train_mode=True)
            # ใช้ Softmax (Gibbs) Policy เพื่อรองรับ Entropy
            probs = torch.softmax(q_values / 0.5, dim=-1) # Temp=0.5 เพื่อความแม่นยำ
            action = torch.multinomial(probs, 1).item()
        return action

    def get_n_step_info(self, buffer):
        # คำนวณผลตอบแทนสะสมย้อนหลัง n ก้าว
        reward = 0
        for i, transition in enumerate(buffer):
            reward += (self.gamma ** i) * transition[2]
        return buffer[0][0], buffer[0][1], reward, buffer[-1][3], buffer[-1][4]

    def update(self):
        sample_data = self.memory.sample(self.batch_size)
        if sample_data is None: return 0
        
        samples, indices, weights = sample_data
        states, actions, rewards, next_states, dones = zip(*samples)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # Double DQN target
        with torch.no_grad():
            next_actions = self.q_net(next_states, train_mode=False).argmax(1, keepdim=True)
            next_q = self.target_net(next_states, train_mode=False).gather(1, next_actions)
            # ใช้ gamma^n สำหรับ Multi-step
            target_q = rewards + (self.gamma ** self.n_step) * next_q * (1 - dones)

        current_q = self.q_net(states, train_mode=True).gather(1, actions)
        
        # คำนวณ Entropy เพื่อนำไปลดใน Loss (ยิ่ง Entropy สูง AI ยิ่งกระจายการเลือก Action)
        logits = self.q_net(states, train_mode=True)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits + 1e-10, dim=-1)
        entropy = -(probs * log_probs).sum(dim=1).mean()

        # Weighted MSE Loss สำหรับ PER
        td_error = torch.abs(target_q - current_q).detach()
        loss = (weights * (target_q - current_q).pow(2)).mean()
        
        # Loss รวม = TD Loss - (alpha * Entropy)
        total_loss = loss - (self.entropy_alpha * entropy)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0) # ป้องกัน Gradient Exploding
        self.optimizer.step()

        # อัปเดตความสำคัญใน Buffer
        self.memory.update_priorities(indices, td_error.cpu().numpy().flatten() + 1e-5)
        return total_loss.item()

def train(seed=42, episodes=2000):
    # เลือกรุ่น LunarLander อัตโนมัติ
    env_name = "LunarLander-v3"
    try:
        env = gym.make(env_name)
    except:
        print(f"⚠️ ไม่พบ {env_name} ในระบบ กำลังใช้ v2 แทน...")
        env_name = "LunarLander-v2"
        env = gym.make(env_name)

    print(f"✅ เริ่มการทดสอบบน: {env_name} (Seed: {seed})")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    agent = EnhancedDQNAgent(env.observation_space.shape[0], env.action_space.n)
    history = []

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=seed+ep)
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # เก็บเข้า n-step buffer
            agent.n_step_buffer.append((state, action, reward, next_state, done))
            if len(agent.n_step_buffer) == agent.n_step:
                s, a, r, ns, d = agent.get_n_step_info(agent.n_step_buffer)
                agent.memory.push(s, a, r, ns, d)
            
            state = next_state
            total_reward += reward
            agent.update()
            
        # เคลียร์ buffer ที่ค้างอยู่ตอนจบ Episode (สำคัญมากสำหรับความต่อเนื่องของข้อมูล)
        while len(agent.n_step_buffer) > 0:
            s, a, r, ns, d = agent.get_n_step_info(agent.n_step_buffer)
            agent.memory.push(s, a, r, ns, d)
            agent.n_step_buffer.popleft()
            
        if ep % 50 == 0:
            agent.target_net.load_state_dict(agent.q_net.state_dict())
            print(f"Seed {seed} | Episode {ep:4d} | Reward: {total_reward:8.2f}")
            
        history.append({"episode": ep, "reward": total_reward})
    
    env.close()
    return pd.DataFrame(history)

if __name__ == "__main__":
    # รายการ Seed 5 ค่า เพื่อเปรียบเทียบกับ Baseline
    seeds = [0, 1, 2, 3, 4]
    total_episodes = 2000
    
    print(f"🚀 เริ่มต้นการเรียนรู้ Enhanced DQN ทั้งหมด {len(seeds)} Seeds (Episodes: {total_episodes} ต่อ Seed)")
    
    for current_seed in seeds:
        print(f"\n--- กำลังประมวลผล Seed: {current_seed} ---")
        df_results = train(seed=current_seed, episodes=total_episodes)
        
        if not df_results.empty:
            # บันทึกไฟล์แยกตาม Seed เพื่อใช้พล็อตกราฟเปรียบเทียบ (Confidence Interval)
            output_file = f"enhanced_seed_{current_seed}.csv"
            df_results.to_csv(output_file, index=False)
            print(f"✅ บันทึกผลลัพธ์ Seed {current_seed} สำเร็จ: '{output_file}'")

    print("\n🏁 การทดสอบครบทั้ง 5 Seeds เรียบร้อยแล้ว! พร้อมสำหรับการนำไฟล์ไปทำ Learning Curve ต่อครับ")