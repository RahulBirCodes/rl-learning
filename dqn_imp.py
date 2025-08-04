# Steps:
# 1. Create pytorch simple mlp model
# 2. Implement dqn algorithm with mlp model
# 3. Test algorithm in certain environment

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import random
import time
from collections import deque
import matplotlib.pyplot as plt

class MLP(nn.Module):
  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.fc1 = nn.Linear(in_dim, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fc3 = nn.Linear(64, out_dim)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# Using CartPole-v1 env, so 4 state vars + actions
# We have discrete actions so for efficiently our MLP will output [Q(state, action_1), Q(state, action_2)]
# We will one-hot-encode the input state

# Implement DQN algorithm

# Constants
replay_cap = 10000
obs_space_dim = 4
action_space_dim = 2
episodes = 600
horizon = 500
eps = 0.1
discount = 0.99
C = 100
minibatch_size = 64
env = gym.make('CartPole-v1', render_mode='human', max_episode_steps=horizon)

# Init replay memory with capacity replay_cap
replay_mem = deque(maxlen=replay_cap)
# Init action-value func Q w/ random weights
q = MLP(4, 2)
# Init target action-value func w/ random weights (separate selecting best action and updating val for training stabilization)
q_hat = MLP(4, 2)
# Init optimizer
optimizer = torch.optim.AdamW(q.parameters(), lr=3e-4)
# Init c counter
c_step = 0
# Init episode steps tracking
episode_steps = []

# Each episode represents an entire run through the horizon
for episode in range(episodes):
    # Init state
    obs, _ = env.reset()
    episode_over = False
    step = 0
    episode_reward = 0

    # Run the simulation through the horizon
    while not episode_over:
        step += 1
        # Get random action with prob eps or act greedily
        obs = torch.tensor(obs, dtype=torch.float32)
        if torch.rand(1) < eps:
            action = env.action_space.sample()
            #print("RANDOM ACTION", action)
        else:
            q_out = q(obs)
            #print("q_out", q_out)
            # We use q network to select action
            action = torch.argmax(q(obs)).item()
            #print("GREEDY ACTION", action)


        # checked till here


        # Execute action in emulator and observe reward and next state
        obs_next, reward, terminated, truncated, _ = env.step(action)
        episode_over = terminated or truncated
        episode_reward += reward
        
        # Store new experience in replay memory
        replay_mem.append((obs, action, reward, obs_next, episode_over))

        # Sample random minibatch from replay mem
        minibatch = random.sample(replay_mem, len(replay_mem) if len(replay_mem) < minibatch_size else minibatch_size)
        print("RANDOM SAMPLE:", minibatch)
        y_j = torch.tensor([float(exp[2]) if exp[4] else float(exp[2]) + discount * torch.max(q_hat(torch.tensor(exp[3], dtype=torch.float32))).item() for exp in minibatch], dtype=torch.float32)

        # Calculate loss and perform gradient descent step
        q_vals = q(torch.stack([torch.tensor(exp[0]) for exp in minibatch]))
        actions = torch.tensor([exp[1] for exp in minibatch])
        q_vals = q_vals[torch.arange(len(minibatch)), actions]
        loss = F.mse_loss(q_vals, y_j)

        #if episode == 99:
        #    print(f"Loss: {loss}")
        
        # Do only one gradient descent step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update state
        obs = obs_next
        c_step += 1
        #print("c_step", c_step)

        # Every C steps, copy weights from Q to Q_hat
        if c_step % C == 0:
            q_hat.load_state_dict(q.state_dict())

    # Store episode steps
    episode_steps.append(step)
    print(f"Finished episode {episode}, steps: {step}")
    
    # Plot steps every 20 episodes
    if (episode + 1) % 500 == 0:
        plt.figure(figsize=(10, 6))
        plt.plot(episode_steps, 'o-')
        plt.title(f'Episode Steps Alive (Episodes 1-{episode + 1})')
        plt.xlabel('Episode')
        plt.ylabel('Steps Alive')
        plt.grid(True)
        plt.show()

print("Finished training")

env.close()





# Demo agent running in test environment

input("Press Enter to start demo...")
print("Demo agent running in test environment")

test_env = gym.make('CartPole-v1', render_mode='human')

obs, _ = test_env.reset()
steps = 0
lost = False
while not lost:
    action = torch.argmax(q(torch.tensor(obs))).item()
    obs_next, reward, terminated, truncated, _ = test_env.step(action)
    lost = terminated
    obs = obs_next
    steps += 1

print("Steps:", steps)
print("Finished demo")
test_env.close()
