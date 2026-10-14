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
import multiprocessing as mp
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
    def __init__(self, learning_rate=0.0001, input_shape=(84, 84, 4), output_size=4, compile_inference=True):
        self.model = self.makeModel(input_shape, output_size)
        self.compile_inference = compile_inference
        self._reset_compiled()
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

    def _q_values(self, x):
        return self.model(x, training=False)

    def _reset_compiled(self):
        # tf.function captures the model's variables, so rebuild it whenever the model is replaced
        self._forward = tf.function(self._q_values, reduce_retracing=True)

    def predict(self, states):
        """Q-values for a batch of states.

        Calling the model directly is much faster than Keras .predict(); wrapping it in tf.function removes
        most of the remaining per-call Python overhead, which dominates for tiny batches.
        """
        x = np.asarray(states, dtype=np.float32)
        out = self._forward(x) if self.compile_inference else self.model(x, training=False)
        return out.numpy()

    def getModel(self):
        return self.model

    def loadModel(self, path):
        if path is None:
            raise ValueError("No path is passed to loadModel call")
        self.model = keras.models.load_model(path)
        self._reset_compiled()

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
N_ACTIONS = 4


class AutoFire(gym.Wrapper):
    """Presses FIRE at the start of the game and after every lost life, whatever the agent chose.

    The ball is only launched by FIRE, so without this a greedy agent can stand still forever. The action that
    was actually executed is returned in info['action'] so it can be stored in the replay buffer.
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.need_fire, self.lives = True, None

    def reset(self):
        self.need_fire, self.lives = True, None
        return self.env.reset()

    def step(self, action):
        if self.need_fire:
            action, self.need_fire = FIRE, False
        obs, reward, terminated, truncated, info = self.env.step(action)
        lives = info.get('lives')
        if self.lives is not None and lives is not None and lives < self.lives:
            self.need_fire = True
        self.lives = lives
        info['action'] = action
        return obs, reward, terminated, truncated, info


def make_env(auto_fire=False):
    env = gym.make("ALE/Breakout-v5", render_mode=None)
    env = MaxAndSkipEnv(env, skip=4)
    env = PreprocessFrame(env)
    env = FrameStack(env, k=4)
    return AutoFire(env) if auto_fire else env


def _worker(conn):
    """Run one game in its own process and answer 'reset' / 'step' / 'close' commands."""
    env = make_env(auto_fire=True)
    try:
        while True:
            command, data = conn.recv()
            if command == 'reset':
                conn.send(env.reset())
            elif command == 'step':
                obs, reward, terminated, truncated, info = env.step(data)
                # restart finished games here, so the learner just keeps stepping
                reset_obs = env.reset() if (terminated or truncated) else None
                conn.send((obs, reward, terminated, truncated, info['action'], reset_obs))
            else:
                break
    finally:
        env.close()
        conn.close()


class ParallelEnvs:
    """Several Breakout games, each in its own process, so the emulators run on separate CPU cores."""

    def __init__(self, num_envs):
        ctx = mp.get_context('spawn')  # fork is unsafe once TensorFlow has started threads
        self.pipes, self.procs = [], []
        for _ in range(num_envs):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child,), daemon=True)
            proc.start()
            child.close()
            self.pipes.append(parent)
            self.procs.append(proc)

    def reset(self):
        for pipe in self.pipes:
            pipe.send(('reset', None))
        return [pipe.recv() for pipe in self.pipes]

    def step(self, actions):
        for pipe, action in zip(self.pipes, actions):
            pipe.send(('step', int(action)))
        return [pipe.recv() for pipe in self.pipes]

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(('close', None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self.procs:
            proc.join(timeout=5)


def train_model(num_episodes=100000, model_path=None, save_dir='models', max_steps=None,
                gamma=0.99, learning_rate=0.0001, minibatch_size=32, train_frequency=4,
                warmup_steps=5000, epsilon_decay_steps=50000, min_epsilon=0.1,
                update_network_frequency=1000, replay_capacity=50000,
                start_step=0, checkpoint_steps=None, num_envs=8, compile_inference=True, progress_callback=None):
    """Train with `num_envs` games played in parallel.

    Everything is counted in environment steps, summed over all games, so the exploration schedule, warm-up,
    one update per `train_frequency` steps and the target-network sync behave the same for any `num_envs`.
    """
    replay_memory = ReplayMemory(replay_capacity)
    online_dqn = DeepQNetwork(learning_rate=learning_rate, compile_inference=compile_inference)
    target_dqn = DeepQNetwork(compile_inference=compile_inference)
    if model_path is not None:
        online_dqn.loadModel(model_path)
    target_dqn.setWeights(online_dqn.model.get_weights())

    envs = ParallelEnvs(num_envs)
    states = envs.reset()  # a list: consecutive transitions share the same array object, which saves memory
    rewards_so_far = [0.0] * num_envs
    steps_so_far = [0] * num_envs
    global_steps, episodes = start_step, 0
    done_training = False

    try:
        while not done_training and episodes < num_episodes:
            epsilon = max(min_epsilon, 1.0 - (1.0 - min_epsilon) * global_steps / epsilon_decay_steps)
            greedy = np.argmax(online_dqn.predict(np.stack(states)), axis=1)
            random_moves = np.random.randint(N_ACTIONS, size=num_envs)
            actions = np.where(np.random.rand(num_envs) < epsilon, random_moves, greedy)

            for i, (next_state, reward, terminated, truncated, action, reset_obs) in enumerate(envs.step(actions)):
                global_steps += 1
                rewards_so_far[i] += reward
                steps_so_far[i] += 1
                # Clip rewards to [-1, 1] so the loss scale does not depend on brick colour.
                replay_memory.append((states[i], action, np.sign(reward), next_state, terminated))
                states[i] = next_state

                if terminated or truncated:
                    episodes += 1
                    print(f'Episode: {episodes}. Total steps count: {steps_so_far[i]}, '
                          f'Total global steps count: {global_steps}, reward count: {rewards_so_far[i]}, '
                          f'epsilon: {epsilon:.3f}', flush=True)
                    rewards_so_far[i], steps_so_far[i], states[i] = 0.0, 0, reset_obs

                if len(replay_memory) >= max(warmup_steps, minibatch_size) and global_steps % train_frequency == 0:
                    online_dqn.train(replay_memory.sampleItems(minibatch_size), target_dqn, gamma)
                if global_steps % update_network_frequency == 0:
                    target_dqn.setWeights(online_dqn.model.get_weights())
                if checkpoint_steps and global_steps % checkpoint_steps == 0:
                    online_dqn.getModel().save(f"{save_dir}/step_{global_steps:09d}.keras")
                    print(f"Checkpoint saved at step {global_steps}", flush=True)
                if progress_callback is not None:
                    progress_callback(global_steps)
                if max_steps is not None and global_steps >= max_steps:
                    done_training = True
                    break
    finally:
        envs.close()

    online_dqn.getModel().save(f"{save_dir}/step_{global_steps:09d}.keras")
    print(f"Model saved as {save_dir}/step_{global_steps:09d}.keras", flush=True)
    print('Training completed!')


def evaluate(model_path, episodes=10, epsilon=0.05, max_steps=5000):
    """Scores of a saved model with a small amount of exploration (model_path=None plays randomly)."""
    dqn = None
    if model_path is not None:
        dqn = DeepQNetwork()
        dqn.loadModel(model_path)
    env = make_env()
    scores = []
    for _ in range(episodes):
        state, done, total, need_fire, lives, steps = env.reset(), False, 0, True, None, 0
        while not done and steps < max_steps:
            steps += 1
            if need_fire:
                action, need_fire = FIRE, False
            elif dqn is None or np.random.rand() < epsilon:
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
    parser.add_argument("--num-envs", type=int, default=8, help="games played in parallel (one CPU process each)")
    parser.add_argument("--checkpoint-steps", type=int, default=100000,
                        help="also save save-dir/step_<N>.keras every this many environment steps")
    parser.add_argument("--start-step", type=int, default=0,
                        help="step count to resume from (with --model), so epsilon continues its schedule")
    parser.add_argument("--epsilon-steps", type=int, default=50000, help="steps to decay epsilon from 1.0 to 0.1")
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--target-update", type=int, default=1000, help="steps between target network syncs")
    parser.add_argument("--evaluate", default=None, metavar="MODEL", help="evaluate a saved model and exit")
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()

    if args.evaluate:
        evaluate(args.evaluate, args.eval_episodes)
        return
    os.makedirs(args.save_dir, exist_ok=True)
    DeepQNetwork().getModel().summary()
    train_model(args.episodes, args.model, args.save_dir, args.max_steps,
                epsilon_decay_steps=args.epsilon_steps, replay_capacity=args.replay_capacity,
                update_network_frequency=args.target_update, start_step=args.start_step,
                checkpoint_steps=args.checkpoint_steps, num_envs=args.num_envs)


if __name__ == "__main__":
    main()
