
import time
import os
import threading
import gym
import multiprocessing
import numpy as np
from queue import Queue
import argparse
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


EPSILON = 0.05
EXPLORATION_DECAY = 1 # 0.995
GAMMA = 0.99


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


class DQNAgent:

    def __init__(self, env, max_eps):
        self.env = env
        self.max_episodes = max_eps
        self.model = self._create_model()
        self.epsilon = EPSILON

    def _create_model(self):
        """
        Builds a neural net model to digest the state
        """
        model = Sequential()
        model.add(Dense(
            20,
            input_shape=self.env.observation_space.shape,
            activation="relu"
        ))
        model.add(Dense(20, activation="relu"))
        model.add(Dense(self.env.action_space.n, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return model

    def train(self):

        times = []

        # train for max_eps episodes
        for episode in range(1, self.max_episodes + 1):

            printProgressBar(episode, self.max_episodes)

            # start at random position
            state, terminal, step = self.env.reset(), False, 0

            # iterate step-by-step
            while not terminal:

                step += 1

                # pick action based on policy
                action = self.policy(state)

                # run action and get reward
                state_next, reward, terminal, info = self.env.step(action)

                if terminal:
                    reward *= -1

                # get expected reward of given states
                # q_state = self.model.predict([[state_next]])[0]

                # compute target Q
                q_target = ( reward + GAMMA * np.amax(self.model.predict([[state_next]])[0]) ) \
                        if not terminal else reward

                # update model
                q_updated = self.model.predict([[state]])[0]
                q_updated[action] = q_target
                self.model.fit([[state]], [[q_updated]], verbose=0)

                # update current state
                state = state_next

            times.append(step)

            # apply exploration decay
            self.epsilon *= EXPLORATION_DECAY

        print(f"max={max(times)} median={np.median(times)} avg={np.average(times)}")

    def policy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.env.action_space.n)
        else:
            expected_rewards = self.model.predict([[state]])[0]
            return np.argmax(expected_rewards)

    def test(self):

        state, done = self.env.reset(), False
        total_reward = 0

        while not done:
            exp_rew = self.model.predict([[state]])[0]
            action = np.argmax(exp_rew)
            new_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            self.env.render()
            time.sleep(0.05)
            state = new_state

        # self.env.close()
        print(f"Total reward: {total_reward}")


if __name__ == "__main__":
    global agent, env
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env, 10**3)
    agent.train()
    agent.test


