import numpy as np

class QAgent:
    def __init__(self, environment, alpha=0.1, gamma=0.99, epsilon=0.3,
                 training=True, epsilon_decay=0.995, epsilon_min=0.01,
                 pretrained_q_table=None):
        self.env = environment
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_space = np.array([-1, 0, 1])  # sell, hold, buy

        # Discretize storage (0–170 MWh) into 12 bins
        self.storage_bins = np.linspace(0, 170, 12)

        # Discretize prices up to 2500 €/MWh
        self.price_bins = [0, 10, 20, 30, 50, 70, 100, 250,
                           500, 1000, 1500, 2000, 2500]
        self.n_hours = 24

        # Initialize Q-table
        if pretrained_q_table is not None:
            self.Q = np.load(pretrained_q_table)
        else:
            # Q-table shape = [storage_bins, price_bins, hours, actions]
            self.Q = np.zeros((len(self.storage_bins),
                               len(self.price_bins) - 1,
                               self.n_hours,
                               len(self.action_space)))

        # Track the best policy
        self.best_reward = -np.inf
        self.action_history = []

    def _discretize_storage(self, s):
        return np.digitize([s], self.storage_bins)[0] - 1

    def _discretize_price(self, p):
        idx = np.digitize([p], self.price_bins)[0] - 1
        return np.clip(idx, 0, len(self.price_bins) - 2)

    def _discretize_hour(self, h):
        return int(h - 1)

    def _discretize_state(self, state):
        """
        state = [storage_level, price, hour, day]
        Convert to discrete index (day is ignored).
        """
        s, p, h, _ = state
        s_bin = self._discretize_storage(s)
        p_bin = self._discretize_price(p)
        h_bin = self._discretize_hour(h)
        return (s_bin, p_bin, h_bin)

    def choose_action_idx(self, s_bin, p_bin, h_bin):
        """
        Epsilon-greedy policy for choosing the action index.
        """
        if self.training and np.random.rand() < self.epsilon:
            return np.random.randint(len(self.action_space))
        else:
            q_values = self.Q[s_bin, p_bin, h_bin, :]
            return np.argmax(q_values)

    def act(self, state):
        """
        Return the actual action (sell=-1, hold=0, buy=1).
        """
        s_bin, p_bin, h_bin = self._discretize_state(state)
        action_idx = self.choose_action_idx(s_bin, p_bin, h_bin)
        return self.action_space[action_idx]

    def update(self, state, action, reward, next_state, total_episode_reward=None):
        """
        Revised update with cross-day price window.
        """
        s_bin, p_bin, h_bin = self._discretize_state(state)
        ns_bin, np_bin, nh_bin = self._discretize_state(next_state)

        action_idx = np.where(self.action_space == action)[0][0]
        current_Q = self.Q[s_bin, p_bin, h_bin, action_idx]

        # Basic info
        storage_level, current_price, current_hour, current_day = state
        next_storage, next_price, next_hour, next_day = next_state

        # ----------------------------------------------------
        # 1) PENALIZE FORCED BUYS
        # ----------------------------------------------------
        forced_penalty = 0.0
        real_action = float(np.clip(action, -1, 1))

        hours_left = 24 - current_hour
        shortfall = self.env.daily_energy_demand - storage_level
        max_possible_buy = hours_left * self.env.max_power_rate

        if shortfall > max_possible_buy:
            needed_now = shortfall - max_possible_buy
            forced_fraction = min(1.0, needed_now / self.env.max_power_rate)
            if real_action < forced_fraction:
                forced_penalty -= 50.0

        if real_action < 0:
            sell_mwh = -real_action * self.env.max_power_rate
            potential_storage = storage_level - sell_mwh
            potential_shortfall = self.env.daily_energy_demand - potential_storage
            hours_left_after = hours_left - 1
            max_buy_after = hours_left_after * self.env.max_power_rate
            if potential_shortfall > max_buy_after:
                forced_penalty -= 20.0

        reward += forced_penalty

        # ----------------------------------------------------
        # 2) IMPROVED PRICE FORECASTING (cross-day window)
        # ----------------------------------------------------
        extended_window = 8
        current_day_int = int(current_day)
        current_hour_int = int(current_hour)

        # Calculate absolute timestep
        absolute_hour = (current_day_int - 1) * 24 + (current_hour_int - 1)
        start_absolute = absolute_hour - (extended_window - 1)
        start_absolute = max(start_absolute, 0)

        past_prices = []
        for timestep in range(start_absolute, absolute_hour + 1):
            day_idx = timestep // 24
            hour_idx = timestep % 24
            if day_idx >= len(self.env.price_values):
                continue
            past_prices.append(self.env.price_values[day_idx][hour_idx])

        past_prices = np.array(past_prices)

        if len(past_prices) > 0:
            avg_recent = np.mean(past_prices)
            if action > 0 and current_price < 0.8 * avg_recent:
                reward += 10.0
            if action < 0 and current_price > 1.2 * avg_recent:
                reward += 10.0

        # ----------------------------------------------------
        # 3) ENCOURAGE LARGER STORAGE
        # ----------------------------------------------------
        if int(current_hour) == 12 and storage_level >= 70.0:
            reward += 20.0

        if int(current_hour) == 20 and storage_level >= 100.0:
            reward += 30.0

        if (int(next_hour) == 1) and (int(next_day) == int(current_day) + 1):
            if storage_level >= 120.0:
                reward += 50.0

        # ----------------------------------------------------
        # Q-LEARNING UPDATE
        # ----------------------------------------------------
        best_next_action_idx = np.argmax(self.Q[ns_bin, np_bin, nh_bin, :])
        best_next_Q = self.Q[ns_bin, np_bin, nh_bin, best_next_action_idx]
        td_target = reward + self.gamma * best_next_Q
        td_error = td_target - current_Q
        self.Q[s_bin, p_bin, h_bin, action_idx] += self.alpha * td_error

        if self.training and total_episode_reward is not None:
            if total_episode_reward > self.best_reward:
                self.best_reward = total_episode_reward
                np.save('best_q_table.npy', self.Q)
                print(f"New best policy saved with reward: {self.best_reward}")

    def reduce_epsilon(self, episode, threshold=0):
        if episode > threshold:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def set_evaluation_mode(self):
        self.training = False
        self.epsilon = 0.0