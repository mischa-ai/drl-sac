# Import libraries for Deep Learning
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
#import cv2

# Import classes for SAC (Soft Actor Critic) Deep Reinforcement Learning
from jetbot_environment import JetbotEnvironment
from actor import Actor
from critic import Critic
from replay_buffer import ReplayBuffer
from sac import SAC

# Define constants
STATE_DIM = 6  # Set the state dimension (number of sensors)
ACTION_DIM = 2  # Set the action dimension (2 motor speeds: left and right)
LEARNING_RATE = 0.0003  # Set the learning rate
GAMMA = 0.99  # Set the discount factor
TAU = 0.005  # Set the target network update rate
BUFFER_SIZE = 100000  # Set the replay buffer size
BATCH_SIZE = 256  # Set the batch size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPISODES = 1000
SAVE_INTERVAL = 50
EVALUATION_INTERVAL = 10
SAVE_PATH = "models"

def reward_function(state, action, next_state, done, collision):
    reward = 0

    # Encourage the agent to move forward
    forward_speed = next_state[-1] - state[-1]
    reward += forward_speed

    # Penalize the agent for colliding or going off the track
    if done or collision:
        reward -= 100

    return reward

# Define the state preprocessing function
def preprocess_state(state):
    img_state, sensor_state = state

    # Preprocess the camera image (e.g., resize, grayscale, normalize)
    #processed_img = cv2.resize(img_state, (84, 84))
    #processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    #processed_img = processed_img.astype(np.float32) / 255.0

    # Combine the preprocessed camera image with the sensor_state
    #processed_state = np.concatenate([processed_img.flatten(), sensor_state])

    # Combine the preprocessed camera image with the sensor_state without CV2
    processed_state = np.concatenate([img_state.flatten(), sensor_state])

    return processed_state

# Define the main training and evaluation loop
def train_and_evaluate():
    env = JetbotEnvironment()
    sac_agent = SAC(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        max_action=1,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        tau=TAU,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        env=env,
        num_episodes=NUM_EPISODES,
        save_interval=SAVE_INTERVAL,
        evaluation_interval=EVALUATION_INTERVAL,
        save_path=SAVE_PATH,
    )
    buffer = ReplayBuffer(state_dim=STATE_DIM, action_dim=ACTION_DIM)

    try:
        for episode in range(NUM_EPISODES):
            state = env.reset()
            state = preprocess_state(state)
            done = False
            episode_reward = 0

            while not done:
                action = sac_agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = preprocess_state(next_state)
                buffer.add(state, action, reward, next_state, done)
                sac_agent.update(buffer)

                state = next_state
                episode_reward += reward

            # Log the episode reward or any other statistics
            print(f"Episode {episode}: Reward = {episode_reward}")

            # Save the model periodically or based on a specific criterion
            if episode % SAVE_INTERVAL == 0:
                sac_agent.save(SAVE_PATH)

            # Evaluate the agent's performance periodically or based on a specific criterion
            if episode % EVALUATION_INTERVAL == 0:
                evaluation_reward = evaluate_single_episode(sac_agent, env)
                print(f"Episode {episode}: Evaluation Reward = {evaluation_reward}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        env.close()

# Define the evaluation function for a single episode
def evaluate_single_episode(agent, env):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.select_action(state, deterministic=True)
        next_state, reward, done = env.step(action)

        state = next_state
        episode_reward += reward

    return episode_reward

if __name__ == "__main__":
    train_and_evaluate()
