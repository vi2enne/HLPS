import numpy as np
import gym
from arguments.arguments_hlps import get_args_ant, get_args_chain
from algos.hlps import hlps_agent
from goal_env.mujoco import *
import random
import torch


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params


def launch(args):

    env = gym.make(args.env_name)
    test_env = gym.make(args.test)
    test_env1 = test_env2 = None
    print("test_env", test_env1, test_env2)

    # set random seeds for reproduce
    env.seed(args.seed)
    if args.env_name != "NChain-v1":
        env.env.env.wrapped_env.seed(args.seed)
        test_env.env.env.wrapped_env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != 'cpu':
        torch.cuda.manual_seed(args.seed)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    # gym.spaces.prng.seed(args.seed)
    # get the environment parameters
    if args.env_name[:3] in ["Ant", "Poi", "Swi"]:
        env.env.env.visualize_goal = args.animate
        test_env.env.env.visualize_goal = args.animate
    env_params = get_env_params(env)
    env_params['max_test_timesteps'] = test_env._max_episode_steps
    sac_trainer = hlps_agent(args, env, env_params, test_env, test_env1, test_env2)
    if args.eval:
        if not args.resume:
            print("random policy !!!")
        sac_trainer._eval_hlps_agent(test_env)
    else:
        sac_trainer.learn()


# get the params
args = get_args_ant()

if __name__ == '__main__':
    launch(args)
