import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import time


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.ptr = 0
        self.size = 0

    def add(self, priority, experience):
        index = self.ptr + self.capacity - 1
        self.data[self.ptr] = experience
        self.update(index, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, index, priority):
        change = priority - self.tree[index]
        self.tree[index] = priority
        self._propagate(index, change)

    def _propagate(self, index, change):
        parent = (index - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get_leaf(self, value):
        parent = 0
        while 2 * parent + 1 < len(self.tree):
            left = 2 * parent + 1
            right = left + 1
            if value <= self.tree[left]:
                parent = left
            else:
                value -= self.tree[left]
                parent = right
        index = parent
        data_index = index - self.capacity + 1
        return index, self.tree[index], self.data[data_index]

    def total_priority(self):
        return self.tree[0]

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_beta = 1.0
        self.epsilon = 1e-5

    def add(self, error, experience):
        priority = (np.abs(error) + self.epsilon) ** self.alpha
        self.tree.add(priority, experience)

    def sample(self, batch_size):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total_priority() / batch_size
        
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            value = np.random.uniform(a, b)
            index, priority, experience = self.tree.get_leaf(value)
            batch.append(experience)
            idxs.append(index)
            priorities.append(priority)

        probabilities = np.array(priorities) / self.tree.total_priority()
        importance_weights = (len(self.tree.data) * probabilities) ** (-self.beta)
        importance_weights /= importance_weights.max()

        # Increase beta
        self.increase_beta()
        
        return batch, idxs, importance_weights

    def update_priorities(self, idxs, errors):
        for i, error in zip(idxs, errors):
            priority = (np.abs(error) + self.epsilon) ** self.alpha
            self.tree.update(i, priority)
    
    def increase_beta(self):
        self.beta = min(self.max_beta, self.beta + self.beta_increment)

class DQNAgent:
    def __init__(self, environment, state_size, action_size=3, alpha=0.001,
                 gamma=0.99, epsilon=0.3, epsilon_decay=0.99, epsilon_min=0.01,
                 replay_buffer_size=10000, batch_size=64, per_alpha=0.6, per_beta=0.4,
                 replay_freq=8, n_step=3, total_episodes=50, hidden_size=256):
        
        self.env = environment
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size  # New parameter for network size
        self.total_episodes = total_episodes
        self.current_episode = 0
        self.episode_start_time = None  # Track episode start time
        
        # Training step counters
        self.steps_done = 0
        self.replay_freq = replay_freq
        
        # Tracking rewards for averaging
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        
        # Gamma settings
        self.gamma_min = 0.0  # Start from zero
        self.gamma_max = 0.998  # Slightly higher than before
        self.gamma = self.gamma_min  # Start at minimum
        
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural Networks and Optimizer
        self.model = self.build_model().to(self.device)
        self.target_model = self.build_model().to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.criterion = nn.MSELoss(reduction='none')
        
        # Training metrics
        self.loss_history = []
        self.train_step_count = 0
        self.target_update_interval = 1000
        
        # Experience Replay
        self.replay_buffer = PrioritizedReplayBuffer(
            replay_buffer_size, alpha=per_alpha, beta=per_beta
        )
        
        # N-step Returns
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Price history
        self.recent_prices_24h = deque(maxlen=24)
        self.recent_prices_4h = deque(maxlen=4)
        
        # Action space
        self.action_bins = np.linspace(-1, 1, self.action_size)

    def build_model(self):
        """
        Builds the neural network for Q-value estimation.
        Uses self.hidden_size to determine layer width.
        """
        return nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_size)
        )

    def update_target_model(self):
        """Hard update of target network weights."""
        self.target_model.load_state_dict(self.model.state_dict())

    def custom_reward(self, state, continuous_action, base_reward, next_state):
        """
        Enhanced reward function that balances buying and selling decisions based on price and storage levels.
        Includes dynamic price thresholds and storage-based urgency factors.
        """
        reward = base_reward
        storage_level, current_price, current_hour, current_day = state
        next_storage_level, next_price, next_hour, next_day = next_state

        # Storage management rewards - scaled by price
        if storage_level <= 10 and continuous_action < 0:  # Selling with low storage
            # Bigger penalty for selling at low prices (worse decision)
            penalty = 20 * (1 + abs(current_price) / 50)  # Scale with price magnitude
            reward -= penalty
        elif storage_level >= 190 and continuous_action > 0:  # Buying with high storage
            # Bigger penalty for buying at high prices (worse decision)
            penalty = 20 * (1 + abs(current_price) / 50)  # Scale with price magnitude
            reward -= penalty
        
        # End of day storage management
        if current_hour >= 23:
            if storage_level > 170:
                reward -= 40  # Penalty for too high storage
            elif storage_level < 50:
                reward -= 40  # Penalty for too low storage
            elif 80 <= storage_level <= 150:
                reward += 40  # Reward for optimal storage range
        
        # Price-based trading rewards
        if len(self.recent_prices_24h) > 23:
            last_24h_avg_price = sum(self.recent_prices_24h) / len(self.recent_prices_24h)
            price_diff_24h = current_price - last_24h_avg_price
            
            # Dynamic price threshold based on storage level
            storage_factor = max(0, min(1, (storage_level - 30) / 90))  # 1 at 120, 0 at 30
            max_acceptable_premium = 15 * (1 - storage_factor)  # Up to 15€ premium when storage very low
            
            # Selling reward when price is high
            if price_diff_24h > 0 and continuous_action < 0:  # Current price above average
                sell_reward = price_diff_24h * 6.0  # Base multiplier
                if storage_level > 100:  # Extra reward if we have enough storage
                    sell_reward *= 2.0
                if storage_level > 150:  # Even more reward if storage is very high
                    sell_reward *= 1.5
                reward += sell_reward
            
            # Buying reward when price is low or storage is critical
            elif continuous_action > 0:  # Buying action
                if price_diff_24h < 0:  # Price is below average
                    # Standard buy reward for good prices
                    buy_reward = abs(price_diff_24h) * 6.0  # Base multiplier
                    if storage_level < 150:  # Extra reward if we have room
                        buy_reward *= 2.0
                    if storage_level < 50:  # Even more reward if storage is very low
                        buy_reward *= 1.5
                    reward += buy_reward
                elif storage_level < 80:  # Storage is getting low
                    # Calculate how "reasonable" the price is
                    price_premium = price_diff_24h  # How much above average
                    if price_premium <= max_acceptable_premium:
                        # Convert to positive reward if price premium is acceptable
                        urgency_factor = 1 - (storage_level / 80)  # 1 when empty, 0 at 80
                        buy_reward = (max_acceptable_premium - price_premium) * urgency_factor * 3.0
                        reward += buy_reward

        # Short-term price trends
        if len(self.recent_prices_4h) > 3:
            last_4h_avg_price = sum(self.recent_prices_4h) / len(self.recent_prices_4h)
            price_diff_4h = current_price - last_4h_avg_price
            
            # Short-term selling reward
            if price_diff_4h > 0 and continuous_action < 0:  # Price spike, good time to sell
                sell_reward = price_diff_4h * 3.0
                if storage_level > 120:  # Extra reward if we have good storage
                    sell_reward *= 1.5
                reward += sell_reward
            
            # Short-term buying reward with storage consideration
            elif continuous_action > 0:  # Buying action
                if price_diff_4h < 0:  # Price is below short-term average
                    buy_reward = abs(price_diff_4h) * 3.0
                    if storage_level < 80:  # Extra reward if storage is low
                        buy_reward *= 1.5
                    reward += buy_reward
                elif storage_level < 60:  # Very low storage
                    # Similar logic for short-term, but more aggressive
                    storage_urgency = 1 - (storage_level / 60)
                    acceptable_premium = 10 * storage_urgency
                    if price_diff_4h <= acceptable_premium:
                        buy_reward = (acceptable_premium - price_diff_4h) * storage_urgency * 2.0
                        reward += buy_reward

        return reward

    def get_n_step_info(self):
        """Return n-step reward, next_state, and done."""
        reward = 0
        next_state = None
        done = False
        
        for i, (_, _, r, next_s, d) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** i) * r
            if d:
                done = True
                next_state = next_s
                break
            elif i == self.n_step - 1:
                next_state = next_s
                
        return reward, next_state, done

    def remember(self, state, action, base_reward, next_state, done):
        # Start timing the first remember call of each episode
        if self.episode_start_time is None:
            self.episode_start_time = time.time()
            self.current_episode_reward = 0.0  # Reset episode reward
        
        # Update current episode reward
        self.current_episode_reward += base_reward
        
        action_index = np.argmin(np.abs(self.action_bins - action))
        self.recent_prices_24h.append(state[1])
        self.recent_prices_4h.append(state[1])
        
        # Calculate adjusted reward
        adjusted_reward = self.custom_reward(state, self.action_bins[action_index], base_reward, next_state)
        
        # Store experience in n-step buffer
        self.n_step_buffer.append((state, action_index, adjusted_reward, next_state, done))
        
        # If n-step buffer is ready
        if len(self.n_step_buffer) == self.n_step:
            n_step_reward, n_step_next_state, n_step_done = self.get_n_step_info()
            first_state, first_action, _, _, _ = self.n_step_buffer[0]
            
            # Calculate TD error for prioritization
            state_tensor = torch.FloatTensor(first_state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(n_step_next_state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                current_q = self.model(state_tensor)[0][first_action]
                next_q = self.target_model(next_state_tensor).max(1)[0][0]
                td_error = abs(n_step_reward + (1 - n_step_done) * (self.gamma ** self.n_step) * next_q - current_q)
            
            # Add to replay buffer with TD error
            self.replay_buffer.add(
                td_error.item(),
                (first_state, first_action, n_step_reward, n_step_next_state, n_step_done)
            )

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            action_index = np.random.randint(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            action_index = q_values.argmax(dim=1).item()
        return self.action_bins[action_index]
    
    def soft_update(self, tau=0.01):
        for target_param, current_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * current_param.data + (1.0 - tau) * target_param.data)
    
    def reduce_epsilon(self):
        """Update epsilon and increment episode counter."""
        # Calculate episode duration if we have a start time
        if self.episode_start_time is not None:
            episode_duration = time.time() - self.episode_start_time
        else:
            episode_duration = 0
        
        # Store the episode reward
        self.episode_rewards.append(self.current_episode_reward)
        
        # Calculate average reward of last 10 episodes
        last_10_avg = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.current_episode += 1  # Increment episode counter
        
        # Update gamma and print complete status
        self.update_gamma(episode_duration, last_10_avg)
        
        # Reset start time and episode reward for next episode
        self.episode_start_time = time.time()
        self.current_episode_reward = 0.0

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """
        Load model with improved error handling for size mismatches.
        """
        try:
            # First try loading with weights_only=True
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            
            # Check if the loaded model has different layer sizes
            first_layer_shape = state_dict['0.weight'].shape[0]
            if first_layer_shape != self.hidden_size:
                print(f"Warning: Loaded model has hidden size {first_layer_shape}, but current model is {self.hidden_size}")
                print("Reinitializing model to match loaded weights...")
                self.hidden_size = first_layer_shape
                self.model = self.build_model().to(self.device)
                self.target_model = self.build_model().to(self.device)
            
            self.model.load_state_dict(state_dict)
            self.update_target_model()
        except RuntimeError as e:
            print(f"Warning: Error loading model: {e}")
            print("Attempting to load with compatibility mode...")
            try:
                state_dict = torch.load(path, map_location=self.device)
                first_layer_shape = state_dict['0.weight'].shape[0]
                if first_layer_shape != self.hidden_size:
                    print(f"Warning: Loaded model has hidden size {first_layer_shape}, but current model is {self.hidden_size}")
                    print("Reinitializing model to match loaded weights...")
                    self.hidden_size = first_layer_shape
                    self.model = self.build_model().to(self.device)
                    self.target_model = self.build_model().to(self.device)
                self.model.load_state_dict(state_dict)
                self.update_target_model()
            except Exception as e2:
                print(f"Fatal error loading model: {e2}")
                raise

    def update_gamma(self, episode_duration=0, last_10_avg=0):
        """Update gamma using a power curve for very quick initial growth that gradually slows."""
        if self.current_episode >= self.total_episodes:
            self.gamma = self.gamma_max
            return
        
        # Use power function for even faster initial growth
        progress = (self.current_episode / self.total_episodes) ** 0.4
        
        # Scale to our gamma range
        self.gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * progress
        
        # Ensure we don't exceed gamma_max
        self.gamma = min(self.gamma_max, self.gamma)
        
        # Log changes every episode with complete metrics
        print(f"Episode {self.current_episode}/{self.total_episodes} | "
              f"Last 10 Avg Reward: {last_10_avg:.2f} | "
              f"Epsilon: {self.epsilon:.4f} | "
              f"Gamma: {self.gamma:.4f} | "
              f"Episode Time: {episode_duration:.2f}s")

    def replay(self):
        """Sample a batch from the replay buffer and train."""
        if len(self.replay_buffer.tree.data) < self.batch_size:
            return

        batch, idxs, importance_weights = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)
        importance_weights = torch.tensor(np.array(importance_weights), dtype=torch.float32, device=self.device)

        # Current Q-values
        q_values = self.model(states).gather(1, actions)

        # Target Q-values using CURRENT gamma value
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1, keepdim=True)[0]
            targets = rewards.unsqueeze(1) + (self.gamma ** self.n_step) * next_q_values * (1 - dones.unsqueeze(1))

        # Update priorities
        errors = torch.abs(targets - q_values).detach().cpu().numpy()
        self.replay_buffer.update_priorities(idxs, errors)

        # Compute loss with importance sampling weights
        loss = (importance_weights.unsqueeze(1) * self.criterion(q_values, targets)).mean()
        self.loss_history.append(loss.item())

        # Print current loss
        if self.train_step_count % 100 == 0:  # Print loss every 100 steps
            print(f"    Step {self.train_step_count} - Loss: {loss.item():.4f}")

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step_count += 1
        
        # Soft update target network more frequently
        if self.train_step_count % 100 == 0:  # Increased frequency
            self.soft_update(tau=0.1)  # Increased tau for faster updates

