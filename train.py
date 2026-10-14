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
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Rescaling
import argparse
import datetime
import os
import random

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
    def __init__(self, capacity=50000):
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity

    def append(self, obj):
        self.memory.append(obj)

    def __len__(self):
        return len(self.memory)

    def sampleItems(self, batch_size):
        return random.sample(self.memory, batch_size)


class DeepQNetwork:
    def __init__(self, learning_rate=0.0001, input_shape=(84, 84, 4), output_size=4):
        self.model = self.makeModel(input_shape, output_size)
        self.output_size = output_size
        self.optimizer = keras.optimizers.Adam(learning_rate)
        self.loss_fn = keras.losses.Huber()

    def makeModel(self, input_shape, output_size):
        model = keras.Sequential(name='deep_q_network')
        model.add(keras.Input(shape=input_shape))
        model.add(Rescaling(1.0 / 255))  # frames are uint8 in [0, 255]
        model.add(Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation='relu'))
        model.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(output_size))
        return model

    def predict(self, states):
        """Q-values for a batch of states. Calling the model directly is much faster than .predict()."""
        return self.model(np.asarray(states, dtype=np.float32), training=False).numpy()

    def getModel(self):
        return self.model

    def loadModel(self, path):
        if path is None:
            raise ValueError("No path is passed to loadModel call")
        self.model = keras.models.load_model(path)

    def setWeights(self, weights):
        self.model.set_weights(weights)

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones, target_model, gamma):
        future_q = tf.reduce_max(target_model(next_states, training=False), axis=1)
        target_values = rewards + gamma * future_q * (1.0 - dones)
        masks = tf.one_hot(actions, self.output_size)
        with tf.GradientTape() as tape:
            q_values = tf.reduce_sum(self.model(states, training=True) * masks, axis=1)
            loss = self.loss_fn(target_values, q_values)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self, batch, target_dqn, gamma):
        states = np.asarray([b[0] for b in batch], dtype=np.float32)
        actions = np.asarray([b[1] for b in batch], dtype=np.int32)
        rewards = np.asarray([b[2] for b in batch], dtype=np.float32)
        next_states = np.asarray([b[3] for b in batch], dtype=np.float32)
        dones = np.asarray([b[4] for b in batch], dtype=np.float32)
        return float(self._train_step(states, actions, rewards, next_states, dones,
                                      target_dqn.model, tf.constant(gamma, tf.float32)))


FIRE = 1  # ALE Breakout actions: NOOP, FIRE, RIGHT, LEFT


def make_env():
    env = gym.make("ALE/Breakout-v5", render_mode=None)
    env = MaxAndSkipEnv(env, skip=4)
    env = PreprocessFrame(env)
    return FrameStack(env, k=4)


def train_model(num_episodes=1000, model_path=None, save_dir='models', max_steps=None,
                gamma=0.99, learning_rate=0.0001, minibatch_size=32, train_frequency=4,
                warmup_steps=5000, epsilon_decay_steps=50000, min_epsilon=0.1,
                update_network_frequency=1000, replay_capacity=50000, save_interval=20):
    replay_memory = ReplayMemory(replay_capacity)
    online_dqn = DeepQNetwork(learning_rate=learning_rate)
    target_dqn = DeepQNetwork()
    if model_path is not None:
        online_dqn.loadModel(model_path)
    target_dqn.setWeights(online_dqn.model.get_weights())

    env = make_env()
    global_steps = 0

    for episode in range(1, num_episodes + 1):
        cur_state = env.reset()
        done = False
        total_reward = 0
        step_no = 0
        lives = None
        need_fire = True  # the ball must be launched with FIRE at the start and after each lost life

        while not done:
            step_no += 1
            global_steps += 1
            epsilon = max(min_epsilon, 1.0 - (1.0 - min_epsilon) * global_steps / epsilon_decay_steps)

            if need_fire:
                action = FIRE
                need_fire = False
            elif np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(online_dqn.predict(cur_state[None])[0]))

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if info.get('lives') is not None:
                if lives is not None and info['lives'] < lives:
                    need_fire = True
                lives = info['lives']

            # Clip rewards to [-1, 1] so the loss scale does not depend on brick colour.
            replay_memory.append((cur_state, action, np.sign(reward), next_state, terminated))

            if global_steps > warmup_steps and global_steps % train_frequency == 0:
                online_dqn.train(replay_memory.sampleItems(minibatch_size), target_dqn, gamma)

            if global_steps % update_network_frequency == 0:
                target_dqn.setWeights(online_dqn.model.get_weights())

            cur_state = next_state
            if max_steps is not None and global_steps >= max_steps:
                break

        print(f'Episode: {episode}. Total steps count: {step_no}, Total global steps count: {global_steps}, '
              f'reward count: {total_reward}, epsilon: {epsilon:.3f}', flush=True)

        if episode % save_interval == 0 or (max_steps is not None and global_steps >= max_steps):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{save_dir}/breakout_dqn_model_{timestamp}_{episode}.keras"
            online_dqn.getModel().save(model_filename)
            print(f"Model saved as {model_filename}", flush=True)
        if max_steps is not None and global_steps >= max_steps:
            break

    env.close()
    print('Training completed!')


def evaluate(model_path, episodes=10, epsilon=0.05):
    """Average score of a saved model with a small amount of exploration."""
    dqn = DeepQNetwork()
    dqn.loadModel(model_path)
    env = make_env()
    scores = []
    for _ in range(episodes):
        state, done, total, need_fire, lives, steps = env.reset(), False, 0, True, None, 0
        while not done and steps < 5000:
            steps += 1
            if need_fire:
                action, need_fire = FIRE, False
            elif np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(dqn.predict(state[None])[0]))
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total += reward
            if lives is not None and info.get('lives', lives) < lives:
                need_fire = True
            lives = info.get('lives', lives)
        scores.append(total)
    env.close()
    print(f'Evaluation over {episodes} episodes: mean={np.mean(scores):.2f} max={max(scores):.0f} scores={scores}')
    return scores


def main():
    parser = argparse.ArgumentParser(description="Train a DQN agent on Atari Breakout.")
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--max-steps", type=int, default=None, help="stop after this many environment steps")
    parser.add_argument("--model", default=None, help="path of a .keras model to resume from")
    parser.add_argument("--save-dir", default="models")
    parser.add_argument("--evaluate", default=None, metavar="MODEL", help="evaluate a saved model and exit")
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()

    if args.evaluate:
        evaluate(args.evaluate, args.eval_episodes)
        return
    os.makedirs(args.save_dir, exist_ok=True)
    DeepQNetwork().getModel().summary()
    train_model(args.episodes, args.model, args.save_dir, args.max_steps)


if __name__ == "__main__":
    main()
