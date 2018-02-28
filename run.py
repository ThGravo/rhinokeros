import gym
from agents.agent import Agent
import argparse


def run(env_name='Ant-v2', num_steps=1000):
    env = gym.make(env_name)
    agent = Agent(env.observation_space, env.action_space)

    state = env.reset()
    reward = None
    done = False
    for _ in range(num_steps):
        env.render()
        action, _ = agent.act(state, reward, done)
        state, reward, done, info = env.step(action)
        print(reward)
        if done:
            state = env.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', default='Swimmer-v1')
    args = parser.parse_args()
    run()
