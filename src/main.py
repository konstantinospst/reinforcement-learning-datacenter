import os
import tempfile
from agent3_DQN import DQNAgent
import time
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from env import DataCenterEnv
from agent import QAgent
from Utils import run_training,run_training,test_and_collect_data,plot_dqn_results,visualize_agent_actions,replicate_forced_logic



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='train.xlsx',
                        help="Path to dataset.")
    parser.add_argument('--days', type=int, default=150,
                        help="Number of days to slice.")
    parser.add_argument('--episodes', type=int, default=100,
                        help="How many training episodes.")
    parser.add_argument('--mode', type=str, default='test',
                        choices=['train', 'test'],
                        help="Mode: 'train' to train the agent, 'test' to evaluate.")
    parser.add_argument('--model', type=str, default='trained_q_table.npy', help="Path to the pre-trained model.")
    args = parser.parse_args()

    df = pd.read_excel(args.path)
    subset_df = df.head(args.days)
    subset_path = 'train_subset.xlsx'
    subset_df.to_excel(subset_path, index=False)
    

    env = DataCenterEnv(subset_path)
    if not args.model == 'DQN.pth':
        if args.mode == 'train':
            # -------------------------
            # TRAINING MODE
            # -------------------------
            agent = QAgent(
                environment=env,
                alpha=0.05,
                gamma=0.99,
                epsilon=0.3,
                training=True,
                epsilon_decay=0.998,
                epsilon_min=0.01
            )

            episode_rewards = []

            for ep in range(args.episodes):
                state = env.reset()
                terminated = False
                ep_reward = 0.0

                while not terminated:
                    action = agent.act(state)
                    next_state, reward, terminated = env.step(action)
                    agent.update(state, action, reward, next_state)  # Q-table update
                    ep_reward += reward
                    state = next_state

                episode_rewards.append(ep_reward)

                # Decay epsilon
                agent.reduce_epsilon(ep)
                print(f"Episode {ep + 1}/{args.episodes}, "
                    f"Reward={ep_reward:.2f}, "
                    f"Eps={agent.epsilon:.3f}")
                
                
                # Save Q-table every 100 episodes
                # if (ep + 1) % 100 == 0:
                #     q_table_filename = f'trained_q_table_{ep + 1}.npy'
                #     np.save(q_table_filename, agent.Q)
                #     print(f"Q-table saved to '{q_table_filename}'")

            # Save final Q-table
            np.save('trained_q_table.npy', agent.Q)
            print("Q-table saved to 'trained_q_table.npy'")

            # Plot Training Rewards
            cumulative_avg_rewards = np.cumsum(episode_rewards) / (np.arange(len(episode_rewards)) + 1)
            plt.figure(figsize=(8, 4))
            plt.plot(episode_rewards, marker='o', label='Episode Reward', color='blue')
            plt.plot(cumulative_avg_rewards, label='Cumulative Average Reward',
                    color='red', linestyle='--')
            plt.title('Training Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.legend()
            plt.show()

            # -----------------------------------
            # Heatmap of the final Q-table
            # -----------------------------------
            # We have Q-table with shape: [storage_bins, price_bins, hours, seasons, actions].
            # We'll average over hours & seasons => shape: [storage_bins, price_bins, actions]
            # Average over hours only
            q_mean = np.mean(agent.Q, axis=2)  # New axis indices after removing season   # average over hours & seasons
            # q_mean shape => (len(storage_bins), len(price_bins)-1, num_actions)

            storage_size = q_mean.shape[0]  # for y-axis
            price_size = q_mean.shape[1]    # for x-axis
            num_actions = q_mean.shape[2]

            plt.figure(figsize=(15, 4))
            actions_map = { -1: "Sell", 0: "Hold", 1: "Buy" }

            for i in range(num_actions):
                plt.subplot(1, num_actions, i+1)
                # Heatmap for q_mean[..., i]
                plt.imshow(q_mean[:, :, i], aspect='auto', origin='lower', cmap='viridis')
                plt.colorbar(label='Q-value')
                plt.title(f"Q Heatmap (Action={actions_map[agent.action_space[i]]})")
                plt.xlabel('Price bin index')
                plt.ylabel('Storage bin index')

            plt.tight_layout()
            plt.show()

        else:
            # -------------------------
            # TESTING MODE
            # -------------------------
            print("Testing mode...")
            agent = QAgent(
                environment=env,
                alpha=0.1,
                gamma=0.99,
                epsilon=0.0,  # No exploration
                training=False,
                pretrained_q_table=args.model
            )

            state = env.reset()
            terminated = False
            total_reward = 0.0
            state_history = []
            reward_history = []
            actions = []
            final_actions_for_plot = []  # track forced vs. actual

            while not terminated:
                state_history.append(state)
                raw_action = agent.act(state)
                # replicate forced logic for plot color coding
                real_action = replicate_forced_logic(env, state, raw_action)

                actions.append(raw_action)
                final_actions_for_plot.append(real_action)

                # step with the agent's raw action (the environment itself will do the same forcing internally)
                next_state, reward, terminated = env.step(raw_action)
                total_reward += reward
                reward_history.append(reward)
                state = next_state

            print(f"Total Reward: {total_reward}")

            # Convert state history to numpy array
            state_history = np.array(state_history)
            storage_levels = state_history[:, 0]
            prices = state_history[:, 1]

            # -------------------------
            # PLOT: Storage Level (First 5 Days)
            # -------------------------
            plt.figure(figsize=(12, 6))
            plt.plot(storage_levels[:120], label='Storage (MWh)', color='tab:blue')
            # Add vertical lines to separate days
            for day in range(1, 6):  # 5 days
                plt.axvline(x=day * 24, color='grey', linestyle='--', linewidth=0.7)
            plt.title('Storage Level over Time (First 5 Days)')
            plt.xlabel('Timestep (Hour)')
            plt.ylabel('Storage (MWh)')
            plt.legend()
            plt.show()

            # -------------------------
            # PLOT: Prices (First 5 Days)
            # -------------------------
            plt.figure(figsize=(12, 6))
            plt.plot(prices[:120], label='Price (€/MWh)', color='tab:orange')
            # Add vertical lines to separate days
            for day in range(1, 6):
                plt.axvline(x=day * 24, color='grey', linestyle='--', linewidth=0.7)
            plt.title('Electricity Prices over Time (First 5 Days)')
            plt.xlabel('Timestep (Hour)')
            plt.ylabel('Price (€/MWh)')
            plt.legend()
            plt.show()

            # -------------------------
            # PLOT: Rewards (First 5 Days)
            # -------------------------
            plt.figure(figsize=(12, 6))
            plt.plot(reward_history[:120], label='Reward', color='tab:green')
            # Add vertical lines to separate days
            for day in range(1, 6):
                plt.axvline(x=day * 24, color='grey', linestyle='--', linewidth=0.7)
            plt.title('Rewards over Time (First 5 Days)')
            plt.xlabel('Timestep (Hour)')
            plt.ylabel('Reward')
            plt.legend()
            plt.show()

            # -------------------------
            # PLOT: Agent Actions (First 5 Days) vs. Prices
            # -------------------------


            visualize_agent_actions(prices, actions, final_actions_for_plot, timesteps=480)

    else:

        ####### DQN method

        # Hyperparameters
        hyperparams = {
        "alpha": 0.001,
        "gamma": 0.99,
        "epsilon": 0.3,
        "epsilon_decay": 0.99,
        "epsilon_min": 0.01,
        "replay_buffer_size": 10000,
        "batch_size": 64,
        "total_episodes": args.episodes  # Pass total episodes to agent
         }



        if args.mode == 'train':
            print("[TRAIN MODE]")
            agent = DQNAgent(environment=env, state_size=4, action_size=3, **hyperparams)
            agent.steps_done = 0
            agent.replay_freq = 8
            agent.n_step = 3

            rewards = run_training(env, agent, args.episodes, args.model)

            # Plot training rewards
            plt.figure(figsize=(8, 4))
            plt.plot(rewards, label='Episode Reward')
            plt.plot(np.cumsum(rewards) / (np.arange(len(rewards)) + 1),
                    label='Cumulative Avg', linestyle='--')
            plt.title('Training Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()
            plt.show()

        else:
            print("[TEST MODE]")
            agent = DQNAgent(environment=env, state_size=4, action_size=3, **hyperparams)
            agent.load_model(args.model)
            

            # -- NEW: collect the full histories so we can produce the 4 plots --
            state_history, actions, rewards  = test_and_collect_data(env, agent)
            
            # Now plot them in the Q-learning style:
            plot_dqn_results(state_history, actions, rewards, timesteps=120)





if __name__ == "__main__":
    main()

            