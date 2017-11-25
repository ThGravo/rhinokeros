import gym
from cem_agent import CEMAgent
import argparse


def run(env_name='CartPole-v1', num_steps=10000000, train=True):
    env = gym.make(env_name)
    agent = CEMAgent(env.observation_space, env.action_space)
    state = env.reset()

    if train:
        state, reward, done, info = env.step(env.action_space.sample())
        for _ in range(num_steps):
            #env.render()
            action, reset = agent.act(state, reward, done)
            state, reward, done, info = env.step(action)
            if done or reset:
                state = env.reset()

        agent.save_weights('cem_{}_params.h5f'.format(env_name), overwrite=True)

    else:
        acc_reward = 0
        agent.policy_model.load_weights('cem_{}_params.h5f'.format(env_name))
        for _ in range(num_steps):
            env.render()
            state, reward, done, info = env.step(agent.policy.get_action(state))
            acc_reward += reward
            if done:
                print("Reward in this episode: {}".format(acc_reward))
                acc_reward = 0
                state = env.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', default='CartPole-v1')
    args = parser.parse_args()
    run()
