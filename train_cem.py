import gym
from cem_agent import CEMAgent
import wrapper
import argparse
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

env_id='Ant-v1'


def make_env():
    env = gym.make(env_id)
    # env = bench.Monitor(env, logger.get_dir())
    return env


def run(env_name='Ant-v1', num_steps=10000000, train=False):
    env = gym.make(env_name)
    if isinstance(env.observation_space, gym.spaces.Discrete):
        env = wrapper.Observ2OneHotWrapper(env)

    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    agent = CEMAgent(env.observation_space, env.action_space, policy_based=True,initial_std=90)
    state = env.reset()

    if train:
        state, reward, done, info = env.step(env.action_space.sample())
        for _ in range(num_steps):
            action, reset = agent.act(state, reward, done)
            state, reward, done, info = env.step(action)
            if done or reset:
                state = env.reset()

        agent.save_weights('cem_{}_params.h5f'.format(env_name), overwrite=True)

    else:
        acc_reward = 0
        agent.model.load_weights('cem_{}_params.h5f'.format(env_name))
        for _ in range(num_steps):
            env.venv.envs[0].render()
            state, reward, done, info = env.step(agent.policy.get_action(state))
            acc_reward += reward
            if done:
                print("Reward in this episode: {}".format(acc_reward))
                acc_reward = 0
                state = env.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', default='CartPole-v1')
    # 'SpaceInvaders-v0' 'Taxi-v2' 'LunarLander-v2' 'CartPole-v1' 'Humanoid-v1' 'FrozenLake-v0' 'Swimmer-v1'
    args = parser.parse_args()
    run()
