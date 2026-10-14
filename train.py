## Wrappers referenced from OpenAI (https://github.com/openai/baselines)

# import gymnasium as gym
# from gymnasium import spaces

import gym
from gym import spaces
import numpy as np
import cv2
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import argparse
import datetime
import os

cv2.ocl.setUseOpenCL(False)

# Reduces flickering. Atari-envs specific issue
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def reset(self):
        return self.env.reset()

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, truncated, info


# Reduces env state size
class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 110), interpolation=cv2.INTER_AREA)
        frame = frame[18:102, :]
        return frame


# Frame stacking helps in capturing motion as part of the state
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames."""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self):
        ob, info = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, truncated, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.stack(self.frames, axis=-1)


class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity

    def append(self, obj):
        self.memory.append(obj)

    def sampleItems(self, batch_size):
        random_indices = np.random.choice(len(self.memory), batch_size, replace=True)
        return [self.memory[i] for i in random_indices]

    def display(self):
        print(self.memory)


class DeepQNetwork:
    def __init__(self, learning_rate=0.0001, input_shape=(84, 84, 4), output_size=4):
        self.model = self.makeModel(input_shape, output_size)
        self.output_size = output_size
        self.optimizer = keras.optimizers.Adam(learning_rate)
        self.loss_fn = tf.keras.losses.MeanAbsoluteError()

    def makeModel(self, input_shape, output_size):
        model = keras.Sequential(name='deep_q_network')
        model.add(Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation='relu', input_shape=input_shape))
        model.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(output_size))
        return model

    def predict(self, input):
        return self.model.predict(input)

    def getModel(self):
        return self.model

    def loadModel(self, path):
        if path is None:
            raise ValueError("No path is passed to loadModel call")
        self.model = keras.models.load_model(path)

    def setWeights(self, weights):
        self.model.set_weights(weights)

    def train(self, batch, target_dqn, gamma):
        states = tf.stack([b[0] for b in batch])
        actions = tf.stack([b[1] for b in batch])
        rewards = tf.stack([b[2] for b in batch])
        next_states = tf.stack([b[3] for b in batch])
        dones = tf.cast(tf.stack([b[4] for b in batch]), dtype=tf.float32)

        masks = keras.ops.one_hot(actions, self.output_size)
        future_q_values = target_dqn.predict(next_states)
        future_rewards = tf.multiply(tf.reduce_max(future_q_values, axis=1), (1 - dones))
        future_rewards = tf.subtract(future_rewards, dones)
        target_values = tf.add(rewards, gamma * future_rewards)

        with tf.GradientTape(persistent=True) as tape:
            observed_values = keras.ops.sum(keras.ops.multiply(self.model(states), masks), axis=1)
            loss = self.loss_fn(target_values, observed_values)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


def train_model(num_episodes=1, model_path=None, save_dir='models'):
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.05
    gamma = 0.99
    alpha = 0.00025
    minibatch_size = 32
    save_interval = 20
    min_steps_before_train = 10
    update_network_frequency = 1000

    replay_memory = ReplayMemory()
    online_dqn = DeepQNetwork(learning_rate=alpha)
    target_dqn = DeepQNetwork()
    if model_path is not None:
        online_dqn.loadModel(model_path)
    target_dqn.setWeights(online_dqn.model.get_weights())

    env = gym.make("ALE/Breakout-v5", render_mode=None)
    # env = gym.make('BreakoutNoFrameskip-v4', render_mode=None)
    env = MaxAndSkipEnv(env, skip=4)
    env = PreprocessFrame(env)
    env = FrameStack(env, k=4)

    episode_rewards = []
    episode_steps = []
    global_steps = 0

    for episode in range(1, num_episodes + 1):
        cur_state = env.reset()
        done = False
        total_reward = 0
        step_no = 0

        while not done:
            step_no += 1
            global_steps += 1

            if global_steps < min_steps_before_train or np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action_values = online_dqn.predict(cur_state[None, :, :, :])[0]
                action = np.argmax(action_values)

            next_state, cur_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += cur_reward

            replay_memory.append((cur_state, action, cur_reward, next_state, done))

            if global_steps > min_steps_before_train and global_steps % minibatch_size == 0:
                batch = replay_memory.sampleItems(minibatch_size)
                online_dqn.train(batch, target_dqn, gamma)

            if global_steps > min_steps_before_train:
                epsilon = max(min_epsilon, epsilon * epsilon_decay)

            if global_steps % update_network_frequency == 0:
                target_dqn.setWeights(online_dqn.model.get_weights())

            cur_state = next_state

        print(f'Episode: {episode}. Total steps count: {step_no}, Total global steps count: {global_steps}, reward count: {total_reward}')
        episode_rewards.append(total_reward)
        episode_steps.append(step_no)

        if episode % save_interval == 0:
            print('Saving the model trained so far...')
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{save_dir}/breakout_dqn_model_{timestamp}_{episode}.keras"
            target_dqn.getModel().save(model_filename)
            print(f"Model saved as {model_filename}")

    env.close()
    print('Training completed!')



def main():
    parser = argparse.ArgumentParser(description="Train a DQN agent on Atari Breakout.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--model", default=None, help="path of a .keras model to resume from")
    parser.add_argument("--save-dir", default="models")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    DeepQNetwork().getModel().summary()
    train_model(args.episodes, args.model, args.save_dir)


if __name__ == "__main__":
    main()
