import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.lines import Line2D
########## Most of the functions we used and plotss

def run_training(env, agent, episodes=50, model_save_path=None):
    """
    Trains the agent and returns the list of episode rewards.
    """
    episode_rewards = []
    total_start_time = time.time()  # Track total training time

    for ep in range(episodes):
        ep_start_time = time.time()  # Start time for the current episode

        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            # Only replay every replay_freq steps
            agent.steps_done += 1
            if agent.steps_done % agent.replay_freq == 0 and len(agent.replay_buffer.tree.data) >= agent.batch_size:
                agent.replay()

            state = next_state
            total_reward += reward

        agent.reduce_epsilon()
        episode_rewards.append(total_reward)

        ep_end_time = time.time()
        ep_duration = ep_end_time - ep_start_time

        if (ep + 1) % 10 == 0:
            avg_last_10 = np.mean(episode_rewards[-10:])
            print(
                f"Episode {ep+1}/{episodes} | "
                f"Last 10 Avg Reward: {avg_last_10:.2f} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Gamma: {agent.gamma:.4f} | "
                f"Beta: {agent.replay_buffer.beta:.4f} | "  # Added beta tracking
                f"Episode Time: {ep_duration:.2f}s"
            )

        # Save model every 50 episodes if a path is provided
        if (ep + 1) % 50 == 0 and model_save_path:
            agent.save_model(f"{model_save_path}_ep{ep+1}.pth")
            print(f"Model saved after episode {ep+1}.")

    # Save final model
    if model_save_path:
        agent.save_model(model_save_path)
        print(f"Final model saved to {model_save_path}")

    total_time = time.time() - total_start_time
    print(f"Total training time for {episodes} episodes: {total_time:.2f}s")

    return episode_rewards

def validate_agent(agent, env):
    """
    Runs validation with the trained agent and returns the total reward and actions.
    """
    state = env.reset()
    total_reward = 0.0
    done = False
    actions = []
    rewards = []

    while not done:
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0  # Disable exploration during validation
        action = agent.choose_action(state)
        agent.epsilon = old_epsilon

        next_state, reward, done = env.step(action)
        total_reward += reward
        actions.append(action)
        rewards.append(reward)
        state = next_state

    print(f"[VALIDATION] Total reward: {total_reward:.2f}")
    return total_reward, actions, rewards

def hyperparam_search(train_data, alpha_candidates, gamma_candidates, episodes=50, plot=True):
    """
    Performs a grid search over alpha and gamma and returns a summary of results.
    """
    env = DataCenterEnv(train_data)

    from Tfiles.agent3_DQN import DQNAgent  # or adjust as needed

    results = {}
    for alpha in alpha_candidates:
        for gamma in gamma_candidates:
            print(f"\n[SEARCH] Training with alpha={alpha}, gamma={gamma}...")

            agent = DQNAgent(
                environment=env,
                state_size=4,
                action_size=3,
                alpha=alpha,
                gamma=gamma,
                epsilon=0.3,
                epsilon_decay=0.995,
                epsilon_min=0.01,
                replay_buffer_size=10000,
                batch_size=64,
                total_episodes=episodes,
                replay_freq=4,
                n_step=3
            )
            
            rewards = run_training(env, agent, episodes=episodes)
            avg_reward = np.mean(rewards[-10:])

            results[(alpha, gamma)] = avg_reward
            print(f" --> Avg of Last 10 Episodes: {avg_reward:.2f}")

    if plot:
        x_labels = [f"α={a}, γ={g}" for (a, g) in results.keys()]
        y_values = list(results.values())

        plt.figure(figsize=(8, 6))
        plt.bar(x_labels, y_values, color='blue')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Avg Reward (Last 10 Eps)")
        plt.title("Hyperparameter Search Results")
        plt.tight_layout()
        plt.show()

    return results

# --------------------------------------------------------------------------------
# NEW: Function to run a "test" episode and collect full histories for plotting
# --------------------------------------------------------------------------------
def test_and_collect_data(env, agent):
    """
    Runs one full episode with the trained agent (no exploration) and collects
    state, action, and reward histories for plotting.
    Returns (state_history, actions, rewards).
    """
    state = env.reset()
    done = False
    state_history = []
    actions = []
    reward_history = []
    total_reward = 0.0
    # Disable exploration
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    while not done:
        state_history.append(state)
        action = agent.choose_action(state)
        actions.append(action)

        next_state, reward, done = env.step(action)
        total_reward += reward
        reward_history.append(reward)
        state = next_state

    # Restore epsilon
    agent.epsilon = old_epsilon
    print(f"[VALIDATION] Total reward: {total_reward:.2f}")
    return np.array(state_history), np.array(actions), np.array(reward_history)

# --------------------------------------------------------------------------------
# Example of replicating your Q-learning style plots (for first 5 days = 120 timesteps)
# Adjust the indexes for storage & price in the state if needed
# --------------------------------------------------------------------------------
def plot_dqn_results(state_history, actions, reward_history, timesteps=120):
    """
    Generates four plots similar to your Q-learning snippet:
      1) Storage Level
      2) Prices
      3) Rewards
      4) Actions vs. Prices (color-coded)
    Assumes state_history[:, 0] = Storage, state_history[:, 1] = Price
    Adjust accordingly if your state representation is different.
    """
    # If your environment has a different state structure, adjust these indices:
    storage_levels = state_history[:, 0]
    prices = state_history[:, 1]

    # 1) Plot: Storage Level (First 5 Days)
    plt.figure(figsize=(12, 6))
    plt.plot(storage_levels[:timesteps], label='Storage (MWh)', color='tab:blue')
    for day in range(1, 6):  # 5 days
        plt.axvline(x=day * 24, color='grey', linestyle='--', linewidth=0.7)
    plt.title('Storage Level over Time (First 5 Days)')
    plt.xlabel('Timestep (Hour)')
    plt.ylabel('Storage (MWh)')
    plt.legend()
    plt.show()

    # 2) Plot: Prices (First 5 Days)
    plt.figure(figsize=(12, 6))
    plt.plot(prices[:timesteps], label='Price (€/MWh)', color='tab:orange')
    for day in range(1, 6):
        plt.axvline(x=day * 24, color='grey', linestyle='--', linewidth=0.7)
    plt.title('Electricity Prices over Time (First 5 Days)')
    plt.xlabel('Timestep (Hour)')
    plt.ylabel('Price (€/MWh)')
    plt.legend()
    plt.show()

    # 3) Plot: Rewards (First 5 Days)
    plt.figure(figsize=(12, 6))
    plt.plot(reward_history[:timesteps], label='Reward', color='tab:green')
    for day in range(1, 6):
        plt.axvline(x=day * 24, color='grey', linestyle='--', linewidth=0.7)
    plt.title('Rewards over Time (First 5 Days)')
    plt.xlabel('Timestep (Hour)')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()

    # 4) Plot: Actions (First 5 Days) vs. Prices
    #    We'll color-code them. For a discrete action space {0,1,2}, assume:
    #       0 = Hold, 1 = Buy, 2 = Sell
    truncated_prices = prices[:timesteps]
    truncated_actions = actions[:timesteps]

    action_colors = []
    for a in truncated_actions:
        if a == 1:
            action_colors.append('red')   # Buy
        elif a == 2:
            action_colors.append('blue')  # Sell
        else:
            action_colors.append('white') # Hold

    plt.figure(figsize=(14, 7))
    plt.plot(truncated_prices, label="Electricity Prices",
             color="black", linewidth=2, marker='o', mfc='none')
    plt.scatter(range(len(truncated_actions)), truncated_prices,
                c=action_colors, edgecolor='black', s=100, label="Actions")

    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Buy',
               markerfacecolor='red', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', label='Sell',
               markerfacecolor='blue', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', label='Hold',
               markerfacecolor='white', markersize=10, markeredgecolor='black')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    for day in range(1, 6):
        plt.axvline(x=day * 24, color='grey', linestyle='--', linewidth=0.7)

    plt.title("Agent Actions and Electricity Prices (First 5 Days)")
    plt.xlabel("Timestep (Hour)")
    plt.ylabel("Price (€ / MWh)")
    plt.grid(True)
    plt.show()


def visualize_agent_actions(prices, raw_actions, final_actions, timesteps=120):
    # Restrict to first 120 hours
    truncated_prices = prices[:timesteps]
    truncated_raw_actions = raw_actions[:timesteps]
    truncated_final = final_actions[:timesteps]

    # Convert final actions to color codes:
    #   - If final_action != raw_action and final_action > 0 => forced buy => green
    #   - Else if final_action > 0 => buy => red
    #   - Else if final_action < 0 => sell => blue
    #   - Else => hold => white
    action_colors = []
    for ra, fa in zip(truncated_raw_actions, truncated_final):
        # forced buy condition
        if abs(fa - ra) > 1e-9 and fa > 0:
            action_colors.append('green')
        else:
            if fa > 0:
                action_colors.append('red')   # buy
            elif fa < 0:
                action_colors.append('blue')  # sell
            else:
                action_colors.append('white') # hold

    plt.figure(figsize=(14, 7))
    plt.plot(truncated_prices, label="Electricity Prices",
            color="black", linewidth=2, marker='o', mfc='none')
    plt.scatter(range(len(truncated_final)), truncated_prices,
                c=action_colors, edgecolor='black', s=100, label="Actions")

    # Vertical lines for day separation
    for day in range(1, 20):
        plt.axvline(x=day * 24, color='grey', linestyle='--', linewidth=0.7)

    plt.title("Agent Actions and Electricity Prices (First 5 Days)")
    plt.xlabel("Timestep (Hour)")
    plt.ylabel("Price (€ / MWh)")
    plt.legend()
    plt.show()


def replicate_forced_logic(env, state, action):
    """
    Replicate the environment's forced-buy / forced-hold rules to detect if
    the final action was changed from the agent's request.
    We do NOT modify the environment; we only replicate the logic for plotting.
    """
    storage_level, price, hour, day = state
    hour = int(hour)
    day = int(day)

    # Number of hours left in the day (including this one)
    hours_left = 24 - hour

    # Current shortfall
    shortfall = env.daily_energy_demand - storage_level

    # Max possible buy if we use full power for all remaining hours
    max_possible_buy = hours_left * env.max_power_rate

    final_action = float(np.clip(action, -1, 1))

    # (A) Forced buy logic
    if shortfall > max_possible_buy:
        needed_now = shortfall - max_possible_buy
        forced_fraction = min(1.0, needed_now / env.max_power_rate)
        if final_action < forced_fraction:
            final_action = forced_fraction  # forced buy

    # (B) Disallow selling if shortfall cannot be fixed
    if final_action < 0:
        sell_mwh = -final_action * env.max_power_rate
        potential_storage = storage_level - sell_mwh
        potential_shortfall = env.daily_energy_demand - potential_storage
        hours_left_after = hours_left - 1
        max_buy_after = hours_left_after * env.max_power_rate
        if potential_shortfall > max_buy_after:
            final_action = 0.0  # forced hold

    return final_action