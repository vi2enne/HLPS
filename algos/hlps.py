import os
import sys
from itertools import chain

sys.path.append('../')
from datetime import datetime
from tensorboardX import SummaryWriter
from models.networks import *
from algos.replay_buffer import replay_buffer, replay_buffer_energy
from algos.her import her_sampler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
from algos.sac.sac import SAC
from algos.sac.replay_memory import ReplayMemory, Array_ReplayMemory
import gym
import pickle
from PIL import Image
import imageio
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_color_codes()
import pandas as pd
from scipy.linalg import expm


SUBGOAL_RANGE = 200.0


class hlps_agent:
    def __init__(self, args, env, env_params, test_env, test_env1=None, test_env2=None):
        self.args = args
        self.env = env
        self.test_env = test_env
        self.env_params = env_params
        self.device = args.device
        self.resume = args.resume
        self.resume_epoch = args.resume_epoch
        self.not_train_low = False
        self.test_env1 = test_env1
        self.test_env2 = test_env2
        self.old_sample = args.old_sample

        self.marker_set = ["v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "*"]
        self.color_set = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                          'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

        self.low_dim = env_params['obs']
        self.env_params['low_dim'] = self.low_dim
        self.hi_dim = env_params['obs']
        print("hi_dim", self.hi_dim)

        self.learn_goal_space = True
        self.whole_obs = False  # use whole observation space as subgoal space
        self.abs_range = abs_range = args.abs_range  # absolute goal range
        self.feature_reg = 0.0  # feature l2 regularization
        print("abs_range", abs_range)

        self.hi_act_space = self.env.env.maze_space

        if self.learn_goal_space:
            self.hi_act_space = gym.spaces.Box(low=np.array([-abs_range, -abs_range]), high=np.array([abs_range, abs_range]))

        if self.whole_obs:
            vel_low = [-10.] * 4
            vel_high = [10.] * 4
            maze_low = np.concatenate((self.env.env.maze_low, np.array(vel_low)))
            maze_high = np.concatenate((self.env.env.maze_high, np.array(vel_high)))
            self.hi_act_space = gym.spaces.Box(low=maze_low, high=maze_high)

        dense_low = True
        self.low_use_clip = not dense_low  # only sparse reward use clip
        if args.replay_strategy == "future":
            self.low_forward = True
            assert self.low_use_clip is True
        else:
            self.low_forward = False
            assert self.low_use_clip is False
        self.hi_sparse = (self.env.env.reward_type == "sparse")

        # # params of learning phi
        resume_phi = args.resume
        self.not_update_phi = False
        phi_path = args.resume_path

        self.save_model = False
        self.start_update_phi = args.start_update_phi
        self.early_stop = args.early_stop  # after success rate converge, don't update low policy and feature
        if args.env_name in ['AntPush-v1', 'AntFall-v1']:
            if self.not_update_phi:
                self.early_stop_thres = 900
            else:
                self.early_stop_thres = 3500
        elif args.env_name in ["PointMaze1-v1"]:
            self.early_stop_thres = 2000
        elif args.env_name == "AntMaze1-v1":
            self.early_stop_thres = 3000
        else:
            self.early_stop_thres = args.n_epochs
        print("early_stop_threshold", self.early_stop_thres)
        self.success_log = []

        self.count_latent = False
        if self.count_latent:
            self.hash = HashingBonusEvaluator(512, 2)
        self.count_obs = False
        if self.count_obs:
            self.hash = HashingBonusEvaluator(512, env_params['obs'])

        self.high_correct = False
        self.k = args.c
        self.delta_k = 0
        self.prediction_coeff = 0.0
        tanh_output = False
        self.use_prob = False
        print("prediction_coeff", self.prediction_coeff)
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')

        if args.save:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name + "_" + args.algo_name + "_Seed_" + str(args.seed) + "_" + current_time)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

        if args.save_tb:
            self.log_dir = 'runs/hlps/' + str(args.env_name) + '/RB_Decay_' + current_time + \
                            "_C_" + str(args.c) + "_Image_" + str(args.image) + \
                            "_Seed_" + str(args.seed) + "_Reward_" + str(args.low_reward_coeff) + \
                            "_NoPhi_" + str(self.not_update_phi) + "_LearnG_" + str(self.learn_goal_space) + "_Early_" + str(self.early_stop_thres) + str(args.early_stop)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            print("log_dir: ", self.log_dir)

        # init low-level network
        self.real_goal_dim = self.hi_act_space.shape[0]  # low-level goal space and high-level action space
        self.init_network()
        # init high-level agent
        self.hi_agent = SAC(self.hi_dim + env_params['goal'], self.hi_act_space, args, False, env_params['goal'],
                            args.gradient_flow_value, args.abs_range, tanh_output)
        self.env_params['real_goal_dim'] = self.real_goal_dim
        self.hi_buffer = ReplayMemory(args.buffer_size)

        # her sampler
        self.c = self.args.c  # interval of high level action
        self.low_her_module = her_sampler(args.replay_strategy, args.replay_k, args.distance, args.future_step,
                                          dense_reward=dense_low, direction_reward=False, low_reward_coeff=args.low_reward_coeff)
        if args.env_name[:5] == "Fetch":
            self.low_buffer = replay_buffer_energy(self.env_params, self.args.buffer_size,
                                               self.low_her_module.sample_her_energy, args.env_name)
        else:
            self.low_buffer = replay_buffer(self.env_params, self.args.buffer_size, self.low_her_module.sample_her_transitions)

        not_load_buffer, not_load_high = True, False
        if self.resume is True:
            self.start_epoch = self.resume_epoch
            if not not_load_high:
                self.hi_agent.policy.load_state_dict(torch.load(self.args.resume_path + \
                                                              '/hi_actor_model.pt', map_location='cuda:0')[0])

            print("load low !!!")
            self.low_actor_network.load_state_dict(torch.load(self.args.resume_path + \
                                                             '/low_actor_model.pt', map_location='cuda:0')[0])
            self.low_critic_network.load_state_dict(torch.load(self.args.resume_path + \
                                                              '/low_critic_model.pt', map_location='cuda:0')[0])

        # sync target network of low-level
        self.sync_target()

        if hasattr(self.env.env, 'env'):
            self.animate = self.env.env.env.visualize_goal
        else:
            self.animate = self.args.animate
        self.distance_threshold = self.args.distance

        if not (args.gradient_flow or args.use_prediction or args.gradient_flow_value):
            self.representation = RepresentationNetwork(env_params, 3, self.abs_range, self.real_goal_dim).to(args.device)
            if args.use_target:
                self.target_phi = RepresentationNetwork(env_params, 3, self.abs_range, 2).to(args.device)
                # load the weights into the target networks
                self.target_phi.load_state_dict(self.representation.state_dict())
            self.representation_optim = torch.optim.Adam(self.representation.parameters(), lr=self.args.philr)
            if resume_phi is True:
                print("load phi from: ", phi_path)
                self.representation.load_state_dict(torch.load(phi_path + \
                                                               '/phi_model.pt', map_location='cuda:0')[0])
                self.gplayer = torch.load(phi_path + '/gplayer.pt')

        elif args.use_prediction:
            self.representation = DynamicsNetwork(env_params, self.abs_range, 2, tanh_output=tanh_output, use_prob=self.use_prob, device=args.device).to(args.device)
            self.representation_optim = torch.optim.Adam(self.representation.parameters(), lr=0.0001)
            if resume_phi is True:
                print("load phi from: ", phi_path)
                self.representation.load_state_dict(torch.load(phi_path + \
                                                               '/phi_model.pt', map_location='cuda:0')[0])

        print("learn goal space", self.learn_goal_space, " update phi", not self.not_update_phi)
        self.train_success = 0
        self.furthest_task = 0.

    def adjust_lr_actor(self, epoch):
        lr_actor = self.args.lr_actor * (0.5 ** (epoch // self.args.lr_decay_actor))
        for param_group in self.low_actor_optim.param_groups:
            param_group['lr'] = lr_actor

    def adjust_lr_critic(self, epoch):
        lr_critic = self.args.lr_critic * (0.5 ** (epoch // self.args.lr_decay_critic))
        for param_group in self.low_critic_optim.param_groups:
            param_group['lr'] = lr_critic

    def learn(self):
        output_data = {"frames": [], "reward": [], "dist": []}

        for epoch in range(self.start_epoch, self.args.n_epochs):
            if epoch > 0 and epoch % self.args.lr_decay_actor == 0:
                self.adjust_lr_actor(epoch)
            if epoch > 0 and epoch % self.args.lr_decay_critic == 0:
                self.adjust_lr_critic(epoch)

            ep_obs, ep_ag, ep_g, ep_actions, ep_latents, ep_fullobs = [], [], [], [], [], []
            last_hi_obs = None
            success = 0
            observation = self.env.reset()
            obs = observation['observation']
            ag = observation['achieved_goal'][:self.real_goal_dim]
            g = observation['desired_goal']
            # identify furthest task
            if g[1] >= 8:
                self.furthest_task += 1
                is_furthest_task = True
            else:
                is_furthest_task = False
            if self.learn_goal_space:
                if self.args.gradient_flow:
                    if self.args.use_target:
                        ag = self.hi_agent.policy_target.phi(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()
                    else:
                        ag = self.hi_agent.policy.phi(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()
                elif self.args.gradient_flow_value:
                    ag = self.hi_agent.critic.phi(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()[0]
                elif self.args.use_prediction:
                    ag = self.representation.phi(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()[0]
                else:
                    if self.args.use_target:
                        ag = self.target_phi(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()[0]
                    else:
                        ag = self.representation(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()[0]
            if self.whole_obs:
                ag = obs.copy()

            ep_latents.append(torch.Tensor(ag).T)
            ep_fullobs.append(torch.Tensor(obs[:self.low_dim]).T)

            time_window = int(self.args.time_window)
            for _ in range(time_window-1):
                ep_latents.append(ep_latents[-1])
                ep_fullobs.append(ep_fullobs[-1])

            # Batch GP
            X = torch.stack(ep_fullobs[-min(len(ep_fullobs), time_window):], dim = 0).to(self.device).unsqueeze(dim=0)
            Y = torch.stack(ep_latents[-min(len(ep_latents), time_window):], dim = 0).to(self.device).unsqueeze(dim=0)
            D = torch.cdist(X, X)
            ag_Z = self.gplayer(D,Y).to(self.device).squeeze().detach().cpu().numpy()[-1,:]

            for t in range(self.env_params['max_timesteps']):
                act_obs, act_g = self._preproc_inputs(obs, g)
                if t % self.c == 0:
                    hi_act_obs = np.concatenate((obs[:self.hi_dim], g))
                    # append high-level rollouts
                    if last_hi_obs is not None:
                        mask = float(not done)
                        if self.high_correct:
                            last_hi_a = ag_Z
                        self.hi_buffer.push(last_hi_obs, last_hi_a, last_hi_r, hi_act_obs, mask, epoch)
                    if epoch < self.args.start_epoch:
                        hi_action = self.hi_act_space.sample()
                    else:
                        hi_action = self.hi_agent.select_action(hi_act_obs)
                    last_hi_obs = hi_act_obs.copy()
                    last_hi_a = hi_action.copy()
                    last_hi_r = 0.
                    done = False
                    if self.old_sample:
                        hi_action_for_low = hi_action
                    else:
                        # make hi_action a delta phi(s)
                        hi_action_for_low = ag_Z.copy() + hi_action.copy()
                        hi_action_for_low = np.clip(hi_action_for_low, -SUBGOAL_RANGE, SUBGOAL_RANGE)
                    hi_action_tensor = torch.tensor(hi_action_for_low, dtype=torch.float32).unsqueeze(0).to(self.device)
                    # update high-level policy
                    if len(self.hi_buffer) > self.args.batch_size:
                        self.update_hi(epoch)
                with torch.no_grad():
                    if self.not_train_low:
                        action = self.test_policy(act_obs[:, :self.low_dim], hi_action_tensor)
                    else:
                        action = self.explore_policy(act_obs[:, :self.low_dim], hi_action_tensor)
                # feed the actions into the environment
                observation_new, r, _, info = self.env.step(action)
                if info['is_success']:
                    done = True
                    # only record the first success
                    if success == 0 and is_furthest_task:
                        success = t
                        self.train_success += 1
                if self.animate:
                    self.env.render()
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal'][:self.real_goal_dim]
                if self.learn_goal_space:
                    if self.args.gradient_flow:
                        if self.args.use_target:
                            ag_new = self.hi_agent.policy_target.phi(
                                torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()
                        else:
                            ag_new = self.hi_agent.policy.phi(torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()
                    elif self.args.gradient_flow_value:
                        ag_new = self.hi_agent.critic.phi(torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()[0]
                    elif self.args.use_prediction:
                        ag_new = self.representation.phi(torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()[0]
                    else:
                        if self.args.use_target:
                            ag_new = self.target_phi(torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()[0]
                        else:
                            ag_new = self.representation(torch.Tensor(obs_new).to(self.device)).detach().cpu().numpy()[0]
                if self.whole_obs:
                    ag_new = obs_new.copy()
                if done is False:
                    if self.count_latent:
                        self.hash.inc_hash(ag[None])
                        r += self.hash.predict(ag_new[None])[0] * 0.1
                    if self.count_obs:
                        self.hash.inc_hash(obs[None])
                        r += self.hash.predict(obs_new[None])[0] * 0.1
                    last_hi_r += r
                # append rollouts
                ep_obs.append(obs[:self.low_dim].copy())
                ep_ag.append(ag_Z.copy())
                ep_g.append(hi_action_for_low.copy())
                ep_actions.append(action.copy())
                # re-assign the observation
                obs = obs_new
                ag = ag_new

                ep_latents.append(torch.Tensor(ag).T)
                ep_fullobs.append(torch.Tensor(obs[:self.low_dim]).T)

                # batch gp
                X = torch.stack(ep_fullobs[-min(len(ep_fullobs), time_window):], dim = 0).to(self.device).unsqueeze(dim=0)
                Y = torch.stack(ep_latents[-min(len(ep_latents), time_window):], dim = 0).to(self.device).unsqueeze(dim=0)
                D = torch.cdist(X, X)
                ag_Z = self.gplayer(D,Y).to(self.device).squeeze().detach().cpu().numpy()[-1,:]

                #  update phi gp
                if epoch > self.start_update_phi and not self.not_update_phi and not self.args.gradient_flow and not self.args.gradient_flow_value:
                    update_gp = (epoch % self.args.gp_update_interval == 0) and epoch>self.args.start_update_gp
                    self.update_phi_gp(epoch, update_gp)
                    if t % self.args.period == 0 and self.args.use_target:
                        self._soft_update_target_network(self.target_phi, self.representation)

            ep_obs.append(obs[:self.low_dim].copy())
            ep_ag.append(ag_Z.copy())
            mask = float(not done)
            hi_act_obs = np.concatenate((obs[:self.hi_dim], g))
            self.hi_buffer.push(last_hi_obs, last_hi_a, last_hi_r, hi_act_obs, mask, epoch)

            mb_obs = np.array([ep_obs])
            mb_ag = np.array([ep_ag])
            mb_g = np.array([ep_g])
            mb_actions = np.array([ep_actions])
            self.low_buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions, success, False])

            # update low-level
            if not self.not_train_low:
                for n_batch in range(self.args.n_batches):
                    self._update_network(epoch, self.low_buffer, self.low_actor_target_network,
                                         self.low_critic_target_network,
                                         self.low_actor_network, self.low_critic_network, 'max_timesteps',
                                         self.low_actor_optim, self.low_critic_optim, use_forward_loss=self.low_forward, clip=self.low_use_clip)
                    if n_batch % self.args.period == 0:
                        self._soft_update_target_network(self.low_actor_target_network, self.low_actor_network)
                        self._soft_update_target_network(self.low_critic_target_network, self.low_critic_network)


            # start to do the evaluation
            if epoch % self.args.eval_interval == 0 and epoch != 0:
                if self.test_env1 is not None:
                    eval_success1, _ = self._eval_hlps_agent(env=self.test_env1)
                    eval_success2, _ = self._eval_hlps_agent(env=self.test_env2)
                farthest_success_rate, _ = self._eval_hlps_agent(env=self.test_env)
                random_success_rate, _ = self._eval_hlps_agent(env=self.env)
                self.success_log.append(farthest_success_rate)
                mean_success = np.mean(self.success_log[-5:])
                # stop updating phi and low
                if self.early_stop and (mean_success >= 0.9 or epoch > self.early_stop_thres):
                    print("early stop !!!")
                    self.not_update_phi = True
                    self.not_train_low = True
                print('[{}] epoch is: {}, eval hier success rate is: {:.3f}'.format(datetime.now(), epoch, random_success_rate))

                if self.args.save:
                    torch.save([self.hi_agent.critic.state_dict()], self.model_path + '/hi_critic_model.pt')
                    torch.save([self.low_critic_network.state_dict()], self.model_path + '/low_critic_model.pt')

                    torch.save(self.gplayer, self.model_path + '/gplayer.pt')
                    if not self.args.gradient_flow and not self.args.gradient_flow_value:
                        if self.save_model:
                            # self.cal_MIV(epoch)
                            torch.save([self.representation.state_dict()], self.model_path + '/phi_model_{}.pt'.format(epoch))
                            torch.save([self.hi_agent.policy.state_dict()], self.model_path + '/hi_actor_{}.pt'.format(epoch))
                            torch.save([self.low_actor_network.state_dict()], self.model_path + '/low_actor_{}.pt'.format(epoch))
                        else:
                            torch.save([self.representation.state_dict()], self.model_path + '/phi_model.pt')
                            torch.save([self.hi_agent.policy.state_dict()], self.model_path + '/hi_actor_model.pt')
                            torch.save([self.low_actor_network.state_dict()], self.model_path + '/low_actor_model.pt')

                    if self.args.save_tb:
                        self.writer.add_scalar('Success_rate/hier_farthest_' + self.args.env_name, farthest_success_rate, epoch)
                        self.writer.add_scalar('Success_rate/hier_random_' + self.args.env_name, random_success_rate, epoch)
                        self.writer.add_scalar('Explore/furthest_task_' + self.args.env_name, self.furthest_task, epoch)

                    output_data["frames"].append((epoch+1)*self.env_params['max_timesteps'])
                    output_data["reward"].append(random_success_rate)
                    output_data["dist"].append(farthest_success_rate)

                    if self.args.save_tb and self.test_env1 is not None:
                        self.writer.add_scalar('Success_rate/eval1_' + self.args.env_name,
                                               eval_success1, epoch)
                        self.writer.add_scalar('Success_rate/eval2_' + self.args.env_name, eval_success2,
                                               epoch)

        file_name = "{}_{}_{}".format(self.args.env_name, self.args.algo_name, self.args.seed)
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(os.path.join("./results", file_name+".csv"), float_format="%.4f", index=False)

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        g = torch.tensor(g, dtype=torch.float32).unsqueeze(0).to(self.device)
        return obs, g

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        if action.shape == ():
            action = np.array([action])
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        if np.random.rand() < self.args.random_eps:
            action = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                       size=self.env_params['action'])
        return action

    def explore_policy(self, obs, goal):
        pi = self.low_actor_network(obs, goal)
        action = self._select_actions(pi)
        return action

    def update_hi(self, epoch):
        if self.args.gradient_flow or self.args.gradient_flow_value:
            sample_data, _ = self.slow_collect()
            sample_data = torch.tensor(sample_data, dtype=torch.float32).to(self.device)
        else:
            sample_data = None
        critic_1_loss, critic_2_loss, policy_loss, _, _ = self.hi_agent.update_parameters(self.hi_buffer,
                                                                                          self.args.batch_size,
                                                                                          self.env_params,
                                                                                          self.hi_sparse,
                                                                                          sample_data)
        if self.args.save_tb:
            self.writer.add_scalar('Loss/hi_critic_1', critic_1_loss, epoch)
            self.writer.add_scalar('Loss/hi_critic_2', critic_2_loss, epoch)
            self.writer.add_scalar('Loss/hi_policy', policy_loss, epoch)

    def random_policy(self, obs, goal):
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                           size=self.env_params['action'])
        return random_actions

    def test_policy(self, obs, goal):
        pi = self.low_actor_network(obs, goal)
        # convert the actions
        actions = pi.detach().cpu().numpy().squeeze()
        if actions.shape == ():
            actions = np.array([actions])
        return actions

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self, epoch, buffer, actor_target, critic_target, actor, critic, T, actor_optim, critic_optim, use_forward_loss=True, clip=True):
        # sample the episodes
        transitions = buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        o, o_next, g, ag = transitions['obs'], transitions['obs_next'], transitions['g'], transitions['ag']
        transitions['obs'], transitions['g'] = o, g
        transitions['obs_next'], transitions['g_next'] = o_next, g
        ag_next = transitions['ag_next']

        # start to do the update
        obs_cur = transitions['obs']
        g_cur = transitions['g']
        obs_next = transitions['obs_next']
        g_next = transitions['g_next']

        # done
        dist = np.linalg.norm(ag_next - g_next, axis=1)
        not_done = (dist > self.distance_threshold).astype(np.int32).reshape(-1, 1)

        # transfer them into the tensor
        obs_cur = torch.tensor(obs_cur, dtype=torch.float32).to(self.device)
        g_cur = torch.tensor(g_cur, dtype=torch.float32).to(self.device)
        obs_next = torch.tensor(obs_next, dtype=torch.float32).to(self.device)
        g_next = torch.tensor(g_next, dtype=torch.float32).to(self.device)
        ag_next = torch.tensor(ag_next, dtype=torch.float32).to(self.device)
        not_done = torch.tensor(not_done, dtype=torch.int32).to(self.device)

        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32).to(self.device)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32).to(self.device)

        # calculate the target Q value function
        with torch.no_grad():
            actions_next = actor_target(obs_next, g_next)
            q_next_value = critic_target(obs_next, g_next, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + critic_target.gamma * q_next_value * not_done
            target_q_value = target_q_value.detach()
            if clip:
                clip_return = self.env_params[T]
                target_q_value = torch.clamp(target_q_value, -clip_return, 0.)
        # the q loss
        real_q_value = critic(obs_cur, g_cur, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        if use_forward_loss:
            forward_loss = critic(obs_cur, ag_next, actions_tensor).pow(2).mean()
            critic_loss += forward_loss
        # the actor loss
        actions_real = actor(obs_cur, g_cur)
        actor_loss = -critic(obs_cur, g_cur, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()

        # start to update the network
        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.low_actor_network.parameters(), 1.0)
        actor_optim.step()

        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.low_critic_network.parameters(), 1.0)
        critic_optim.step()

        if self.args.save_tb:
            if T == 'max_timesteps':
                name = 'low'
            else:
                name = 'high'
            self.writer.add_scalar('Loss/' + name + '_actor_loss' + self.args.metric, actor_loss, epoch)
            self.writer.add_scalar('Loss/' + name + '_critic_loss' + self.args.metric, critic_loss, epoch)

    def _eval_hlps_agent(self, env, n_test_rollouts=1):
        total_success_rate = []
        if not self.args.eval:
            n_test_rollouts = self.args.n_test_rollouts
        discount_reward = np.zeros(n_test_rollouts)
        for roll in range(n_test_rollouts):
            per_success_rate = []
            observation = env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            ep_latents = []
            ep_fullobs = []

            ell = np.exp(self.gplayer.ell.data)
            gamma2 = np.exp(self.gplayer.gamma2.data)
            sigma2 = np.exp(self.gplayer.sigma2.data)

            lam = np.sqrt(3) / ell
            F = np.array([[0, 1], [-lam ** 2, -2 * lam]], dtype=object)
            Pinf = np.array([[gamma2, 0], [0, gamma2 * lam ** 2]], dtype=object)
            h = np.array([[1], [0]], dtype=object)

            # State mean and covariance
            M = np.zeros((F.shape[0], 2))
            P = np.zeros((F.shape[0], F.shape[0]))
            P = Pinf

            time_window = int(self.args.time_window)
            for num in range(self.env_params['max_test_timesteps']):
                with torch.no_grad():
                    act_obs, act_g = self._preproc_inputs(obs, g)
                    ag = self.representation(torch.Tensor(obs).to(self.device)).detach().cpu().numpy()[0]
                    ep_fullobs.append(torch.Tensor(obs[:self.low_dim]).T)
                    ep_latents.append(torch.Tensor(ag).T)

                    if num==0:
                        ## duplicate first state
                        ep_latents.append(ep_latents[-1])
                        ep_fullobs.append(ep_fullobs[-1])

                    ## online gp
                    X = torch.stack(ep_fullobs[-2:], dim = 0).to(self.device)
                    D = torch.cdist(X, X)
                    A = expm(F * D.cpu().numpy())
                    Q = Pinf - A.dot(Pinf).dot(A.T)
                    M = A.dot(M)
                    P = A.dot(P).dot(A.T) + Q

                    # Update step
                    v = ag.T - h.T.dot(M)
                    s = h.T.dot(P).dot(h) + sigma2.numpy()
                    k = P.dot(h) / s
                    M = np.add(M, k.dot(v))
                    P = np.add(P, -k.dot(h.T).dot(P))

                    ag_Z = M[0].astype(np.float64)

                    if num % self.c == 0:

                        hi_act_obs = np.concatenate((obs[:self.hi_dim], g))
                        hi_action = self.hi_agent.select_action(hi_act_obs, evaluate=True)
                        if self.old_sample:
                            new_hi_action = hi_action
                        else:
                            new_hi_action = ag_Z + hi_action
                            new_hi_action = np.clip(new_hi_action, -SUBGOAL_RANGE, SUBGOAL_RANGE)

                        hi_action_tensor = torch.tensor(new_hi_action, dtype=torch.float32).unsqueeze(0).to(self.device)
                    action = self.test_policy(act_obs[:, :self.low_dim], hi_action_tensor)

                observation_new, rew, done, info = env.step(action)
                if self.animate:
                    env.render()
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                if done:
                    per_success_rate.append(info['is_success'])
                    if bool(info['is_success']):
                        # print("t:", num)
                        discount_reward[roll] = 1 - 1. / self.env_params['max_test_timesteps'] * num
                    break
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        global_success_rate = np.mean(total_success_rate[:, -1])
        global_reward = np.mean(discount_reward)
        if self.args.eval:
            print("hier success rate", global_success_rate, global_reward)

        return global_success_rate, global_reward

    def init_network(self):
        # create gp layer
        self.gplayer = GPlayer(self.args.gamma2, self.args.ell, self.args.sigma2)
        self.gp_optim = torch.optim.Adam(self.gplayer.parameters(), lr=self.args.gplr)

        self.low_actor_network = actor(self.env_params, self.real_goal_dim).to(self.device)
        self.low_actor_target_network = actor(self.env_params, self.real_goal_dim).to(self.device)
        self.low_critic_network = criticWrapper(self.env_params, self.args, self.real_goal_dim).to(self.device)
        self.low_critic_target_network = criticWrapper(self.env_params, self.args, self.real_goal_dim).to(self.device)

        self.start_epoch = 0

        # create the optimizer
        self.low_actor_optim = torch.optim.Adam(self.low_actor_network.parameters(), lr=self.args.lr_actor)
        self.low_critic_optim = torch.optim.Adam(self.low_critic_network.parameters(), lr=self.args.lr_critic, weight_decay=1e-5)


    def sync_target(self):
        # load the weights into the target networks
        self.low_actor_target_network.load_state_dict(self.low_actor_network.state_dict())
        self.low_critic_target_network.load_state_dict(self.low_critic_network.state_dict())

    def update_phi_gp(self, epoch, update_gp=False):
        sample_data, hi_action = self.slow_collect()
        l, b, c = sample_data.shape
        sample_data = torch.tensor(sample_data, dtype=torch.float32).to(self.device)
        obs, obs_next = self.representation(sample_data[0]), self.representation(sample_data[1])
        hi_obs, hi_obs_next = self.representation(sample_data[2]), self.representation(sample_data[3])

        if update_gp:
            emb_combined = torch.cat((obs, obs_next, hi_obs, hi_obs_next), dim=0)
            emb_combined = torch.transpose(emb_combined,0,1).reshape(1, b*l, obs.size()[-1])

            D = torch.cdist(torch.transpose(sample_data,0,1).reshape(1, b*l, -1),
                            torch.transpose(sample_data,0,1).reshape(1, b*l, -1))
            ag_Z = self.gplayer(D, emb_combined).to(self.device).reshape(b, l, obs.size()[-1])

            obs_z = ag_Z[:,0,:]
            obs_next_z = ag_Z[:,1,:]
            hi_obs_z = ag_Z[:,2,:]
            hi_obs_next_z = ag_Z[:,3,:]

        if not self.args.use_prediction:
            min_dist = torch.clamp((obs - obs_next).pow(2).mean(dim=1), min=0.)
            max_dist = torch.clamp(1 - (hi_obs - hi_obs_next).pow(2).mean(dim=1), min=0.)
            representation_loss = (min_dist + max_dist).mean()
            # add l2 regularization
            representation_loss += self.feature_reg * (obs / self.abs_range).pow(2).mean()
        else:
            hi_action = torch.tensor(hi_action, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                target_next_obs = self.representation.phi(sample_data[3])
            obs, obs_next = self.representation.phi(sample_data[0]), self.representation.phi(sample_data[1])
            min_dist = torch.clamp((obs - obs_next).pow(2).mean(dim=1), min=0.)
            hi_obs, hi_obs_next = self.representation.phi(sample_data[2]), self.representation.phi(sample_data[3])
            max_dist = torch.clamp(1 - (hi_obs - hi_obs_next).pow(2).mean(dim=1), min=0.)
            representation_loss = (min_dist + max_dist).mean()
            # prediction loss
            if self.use_prob:
                predict_distribution = self.representation(sample_data[2], hi_action)
                prediction_loss = - predict_distribution.log_prob(target_next_obs).mean()
            else:
                predict_state = self.representation(sample_data[2], hi_action)
                prediction_loss = (predict_state - target_next_obs).pow(2).mean()
            representation_loss += self.prediction_coeff * prediction_loss

        if update_gp:
            min_dist = torch.clamp((obs_z - obs_next_z).pow(2).mean(dim=1), min=0.)
            max_dist = torch.clamp(1 - (hi_obs_z - hi_obs_next_z).pow(2).mean(dim=1), min=0.)
            representation_loss = (min_dist + max_dist).mean()
            # add l2 regularization
            representation_loss += self.feature_reg * (obs / self.abs_range).pow(2).mean()
            representation_loss = torch.log(1 + torch.exp(representation_loss))
            min_dist = torch.clamp((obs - obs_next).pow(2).mean(dim=1), min=0.)
            max_dist = torch.clamp((hi_obs - hi_obs_next).pow(2).mean(dim=1), min=0.)
            representation_loss *= ((1 + min_dist)/(1 + max_dist)).mean()

            self.gp_optim.zero_grad()
            representation_loss.backward()
            self.gp_optim.step()
            self.gplayer.ell.data.clamp_(self.args.elll, self.args.ellh)
            self.gplayer.gamma2.data.clamp_(self.args.gamma2l, self.args.gamma2h)
            self.gplayer.sigma2.data.clamp_(self.args.sigma2l, self.args.sigma2h)
        else:
            self.representation_optim.zero_grad()
            representation_loss.backward()
            self.representation_optim.step()

        if self.args.save_tb:
            self.writer.add_scalar('Loss/phi_loss' + self.args.metric, representation_loss, epoch)

    def slow_collect(self, batch_size=100):
        if self.args.use_prediction:
            transitions, _ = self.low_buffer.sample(batch_size)
            obs, obs_next = transitions['obs'], transitions['obs_next']

            hi_obs, hi_action, _, hi_obs_next, _ = self.hi_buffer.sample(batch_size)
            hi_obs, hi_obs_next = hi_obs[:, :self.env_params['obs']], hi_obs_next[:, :self.env_params['obs']]
            train_data = np.array([obs, obs_next, hi_obs, hi_obs_next])
            return train_data, hi_action
        else:
            # new negative samples
            episode_num = self.low_buffer.current_size
            obs_array = self.low_buffer.buffers['obs'][:episode_num]
            episode_idxs = np.random.randint(0, episode_num, batch_size)
            t_samples = np.random.randint(self.env_params['max_timesteps'] - self.k - self.delta_k, size=batch_size)
            if self.delta_k > 0:
                delta = np.random.randint(self.delta_k, size=batch_size)
            else:
                delta = 0

            hi_obs = obs_array[episode_idxs, t_samples]
            hi_obs_next = obs_array[episode_idxs, t_samples + self.k + delta]
            obs = hi_obs
            obs_next = obs_array[episode_idxs, t_samples + 1 + delta]

            # filter data when the robot is ant
            if self.args.env_name[:3] == "Ant":
                good_index = np.where((hi_obs[:, 2] >= 0.3) & (hi_obs_next[:, 2] >= 0.3) & (obs_next[:, 2] >= 0.3))[0]
                hi_obs = hi_obs[good_index]
                hi_obs_next = hi_obs_next[good_index]
                obs = hi_obs
                obs_next = obs_next[good_index]
                assert len(hi_obs) == len(hi_obs_next) == len(obs_next)

            train_data = np.array([obs, obs_next, hi_obs, hi_obs_next])
            return train_data, None

    def prioritized_collect(self, batch_size=100):
        # new negative samples
        episode_num = self.low_buffer.current_size
        obs_array = self.low_buffer.buffers['obs'][:episode_num]

        candidate_idxs = self.candidate_idxs[:episode_num * (self.low_buffer.T - self.k + 1)]
        p = self.low_buffer.get_all_data()['p']
        p = p[:, :self.low_buffer.T - self.k + 1]
        p = p.reshape(-1)
        p_old = p / p.sum()
        selected = np.random.choice(len(candidate_idxs), size=batch_size, replace=False, p=p_old)
        if self.add_reg:
            # select the regularization data
            p_new = 1. / np.sqrt(1 + p)
            p_new_norm = p_new / p_new.sum()
            selected_new = np.random.choice(len(candidate_idxs), size=batch_size, replace=False, p=p_new_norm)
            selected_idx_new = candidate_idxs[selected_new]
            episode_idxs_new = selected_idx_new[:, 0]
            t_samples_new = selected_idx_new[:, 1]
            reg_obs = obs_array[episode_idxs_new, t_samples_new]
        else:
            reg_obs = None

        selected_idx = candidate_idxs[selected]
        episode_idxs = selected_idx[:, 0]
        t_samples = selected_idx[:, 1]

        hi_obs = obs_array[episode_idxs, t_samples]
        hi_obs_next = obs_array[episode_idxs, t_samples + self.k]
        obs = hi_obs
        obs_next = obs_array[episode_idxs, t_samples + 1]

        train_data = np.array([obs, obs_next, hi_obs, hi_obs_next])
        return train_data, selected_idx, reg_obs

    def quick_prioritized_collect(self, batch_size=100):
        # new negative samples
        episode_num = self.low_buffer.current_size
        obs_array = self.low_buffer.buffers['obs'][:episode_num]

        random_index = np.random.randint(len(self.high_p), size=batch_size)
        selected = self.high_p[random_index]
        if self.add_reg:
            random_index_new = np.random.randint(len(self.low_p), size=batch_size)
            selected_new = self.low_p[random_index_new]
            selected_idx_new = self.cur_candidate_idxs[selected_new]
            episode_idxs_new = selected_idx_new[:, 0]
            t_samples_new = selected_idx_new[:, 1]
            reg_obs = obs_array[episode_idxs_new, t_samples_new]
        else:
            reg_obs = None

        selected_idx = self.cur_candidate_idxs[selected]
        episode_idxs = selected_idx[:, 0]
        t_samples = selected_idx[:, 1]

        hi_obs = obs_array[episode_idxs, t_samples]
        hi_obs_next = obs_array[episode_idxs, t_samples + self.k]
        obs = hi_obs
        obs_next = obs_array[episode_idxs, t_samples + 1]

        train_data = np.array([obs, obs_next, hi_obs, hi_obs_next])
        return train_data, selected_idx, reg_obs

    def new_prioritized_collect(self, batch_size=100):
        # new negative samples
        episode_num = self.low_buffer.current_size
        obs_array = self.low_buffer.buffers['obs'][:episode_num]

        candidate_idxs = self.candidate_idxs[:episode_num * (self.low_buffer.T - self.k + 1)]
        # sample in replay buffer
        selected = self.low_buffer._sample_for_phi(batch_size)
        selected_idx = candidate_idxs[selected]
        episode_idxs = selected_idx[:, 0]
        t_samples = selected_idx[:, 1]

        hi_obs = obs_array[episode_idxs, t_samples]
        hi_obs_next = obs_array[episode_idxs, t_samples + self.k]
        obs = hi_obs
        obs_next = obs_array[episode_idxs, t_samples + 1]

        # filter data when the robot is ant
        if self.args.env_name[:3] == "Ant":
            good_index = np.where((hi_obs[:, 2] >= 0.3) & (hi_obs_next[:, 2] >= 0.3) & (obs_next[:, 2] >= 0.3))[0]
            hi_obs = hi_obs[good_index]
            hi_obs_next = hi_obs_next[good_index]
            obs = hi_obs
            obs_next = obs_next[good_index]
            selected_idx = selected_idx[good_index]
            selected = selected[good_index]
            assert len(hi_obs) == len(hi_obs_next) == len(obs_next) == len(selected_idx)

        train_data = np.array([obs, obs_next, hi_obs, hi_obs_next])
        return train_data, selected


    def cal_stable(self):
        transitions, _ = self.low_buffer.sample(100)
        obs = transitions['obs']

        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        self.representation.load_state_dict(torch.load(self.args.resume_path + \
                                                       '/phi_model_3000.pt', map_location='cuda:1')[0])

        feature1 = self.representation(obs).detach().cpu().numpy()

        self.representation.load_state_dict(torch.load(self.args.resume_path + \
                                                       '/phi_model_4000.pt', map_location='cuda:1')[0])

        feature2 = self.representation(obs).detach().cpu().numpy()
        distance = np.linalg.norm(feature1 - feature2)
        print("distance", distance)

    def select_by_count(self, hi_obs, t, epoch):
        transitions, _ = self.low_buffer.sample(1000)
        obs, ag_record = transitions['obs'], transitions['ag_record'][:, :self.real_goal_dim]
        obs_tensor = torch.Tensor(obs).to(self.device)
        features = self.representation(obs_tensor[:, :self.hi_dim]).detach().cpu().numpy()

        # current state
        hi_obs_tensor = torch.Tensor(hi_obs).to(self.device)
        hi_feature = self.representation(hi_obs_tensor).detach().cpu().numpy()

        distances = np.linalg.norm(features - hi_feature, axis=-1)
        near_indexs = np.where((distances < 20) & (distances > self.min_dist))[0]

        new_features = features[near_indexs]
        new_ag_record = ag_record[near_indexs]
        if self.future_count_coeff == 0.:
            count = np.array(self.hash.predict(new_features))
        else:
            count = np.array(self.future_hash.predict(new_features)) * self.future_count_coeff
        distances_new = distances[near_indexs]
        obs_new = obs[near_indexs]
        if len(count) == 0:
            # no nearby feature
            subgoal = features[0]
            xy_select = ag_record[0]
            obs_select = obs[0]
        else:
            # select subgoal with less count and larger distance
            score = count + 1. / np.sqrt(distances_new + 1) * self.distance_coeff

            if self.history_subgoal_coeff != 0:
                # select the subgoal that rarely selected in the past
                count_history = np.array(self.subgoal_hash.predict(new_features))
                score += count_history * self.history_subgoal_coeff

            # select the subgoal that can success
            if epoch > self.start_count_success:
                dis_to_goal = np.array(self.success_hash.predict(new_features))
                score += dis_to_goal * self.success_coeff

            min_index = score.argmin()
            subgoal = new_features[min_index]
            xy_select = new_ag_record[min_index]
            obs_select = obs_new[min_index]

        current_hi_step = int(t / self.c)
        self.count_xy_record[current_hi_step].append(xy_select)
        self.subgoal_record[current_hi_step].append(subgoal)
        self.valid_times += 1

        # record all history subgoal
        if self.history_subgoal_coeff != 0:
            self.subgoal_xy_hash.inc_hash(xy_select.copy()[None])
            self.subgoal_hash.inc_hash(subgoal.copy()[None])
            self.all_history_xy.append(xy_select)
            if self.usual_update_history:
                self.all_history_obs.append(obs_select)
            else:
                self.all_history_subgoal.append(subgoal)

        return subgoal, xy_select


    def cal_fall_over(self):
        self.low_buffer = torch.load(self.args.resume_path + '/low_buffer.pt', map_location='cuda:1')
        print("load buffer !!!")
        transitions, _ = self.low_buffer.sample(100000)
        obs, ag_record = transitions['obs'], transitions['ag_record']
        select_index = np.where((obs[:, 0] < -1.0) & (obs[:, 1] < -1.0))[0]
        print("select", len(select_index))
        total_count = len(select_index)
        obs_new = obs[select_index]
        not_fall = np.where((obs_new[:, 2] >= 0.3) & (obs_new[:, 2] <= 1.0))[0]
        smaller = np.where(obs_new[:, 2] < 0.3)[0]
        larger = np.where(obs_new[:, 2] > 1.0)[0]
        print("not fall over", len(not_fall))
        not_fall_rate = len(not_fall) / total_count
        print("not fall rate", not_fall_rate)
        print("smaller", len(smaller) / total_count)
        print("larger", len(larger) / total_count)
        print("#" * 20)

        # new negative samples
        batch_size = 10000
        episode_num = self.low_buffer.current_size
        obs_array = self.low_buffer.buffers['obs'][:episode_num]
        episode_idxs = np.random.randint(0, episode_num, batch_size)
        t_samples = np.random.randint(self.env_params['max_timesteps'] - self.k - self.delta_k, size=batch_size)
        if self.delta_k > 0:
            delta = np.random.randint(self.delta_k, size=batch_size)
        else:
            delta = 0

        hi_obs = obs_array[episode_idxs, t_samples]
        hi_obs_next = obs_array[episode_idxs, t_samples + self.k + delta]
        obs = hi_obs
        obs_next = obs_array[episode_idxs, t_samples + 1 + delta]

        # filter data when the robot is ant
        if self.args.env_name[:3] == "Ant":
            good_index = np.where((hi_obs[:, 2] >= 0.3) & (hi_obs_next[:, 2] >= 0.3) & (obs_next[:, 2] >= 0.3))[0]
            hi_obs = hi_obs[good_index]
            hi_obs_next = hi_obs_next[good_index]
            obs = hi_obs
            obs_next = obs_next[good_index]

            select_index = np.where((hi_obs[:, 0] < -1.0) & (hi_obs[:, 1] < -1.0) & (hi_obs_next[:, 0] < -1.0) & (hi_obs_next[:, 1] < -1.0) & \
                                    (obs_next[:, 0] < -1.0) & (obs_next[:, 1] < -1.0))[0]
            hi_obs = hi_obs[select_index]
            hi_obs_next = hi_obs_next[select_index]
            obs = hi_obs
            obs_next = obs_next[select_index]

            c_change = np.linalg.norm(hi_obs - hi_obs_next, axis=1)
            one_change = np.linalg.norm(obs - obs_next, axis=1)
            delta_change = c_change - one_change
            print("c_change", len(c_change))
            print("len_obs", len(hi_obs))
            x_change = np.linalg.norm(hi_obs[:, 0:2] - hi_obs_next[:, 0:2], axis=1) - np.linalg.norm(obs[:, 0:2] - obs_next[:, 0:2], axis=1)
            print("delta_change", delta_change)
            print("x_change", x_change)


        train_data = np.array([obs, obs_next, hi_obs, hi_obs_next])




    def cal_phi_loss(self):
        self.low_buffer = torch.load(self.args.resume_path + '/low_buffer.pt', map_location='cuda:1')
        self.representation.load_state_dict(torch.load(self.args.resume_path + \
                                                       '/phi_model_{}.pt'.format(3000), map_location='cuda:1')[0])

        print("load buffer and phi !!!")

        # new negative samples
        episode_num = self.low_buffer.current_size
        obs_array = self.low_buffer.buffers['obs'][:episode_num]
        ag_record_array = self.low_buffer.buffers['ag_record'][:episode_num]

        candidate_idxs = np.array([[i, j] for i in range(episode_num) for j in range(self.low_buffer.T - self.k + 1)])
        p = np.ones(len(candidate_idxs)) * 1. / len(candidate_idxs)
        selected = np.random.choice(len(candidate_idxs), size=500000, replace=False, p=p)
        print("origin selected", len(selected))
        selected_idx = candidate_idxs[selected]
        episode_idxs = selected_idx[:, 0]
        t_samples = selected_idx[:, 1]

        hi_obs = obs_array[episode_idxs, t_samples]
        ag_record = ag_record_array[episode_idxs, t_samples]
        hi_obs_next = obs_array[episode_idxs, t_samples + self.k]
        obs = hi_obs
        obs_next = obs_array[episode_idxs, t_samples + 1]

        # filter data when the robot is ant
        if self.args.env_name[:3] == "Ant":
            good_index = np.where((hi_obs[:, 2] >= 0.3) & (hi_obs_next[:, 2] >= 0.3) & (obs_next[:, 2] >= 0.3))[0]
            hi_obs = hi_obs[good_index]
            hi_obs_next = hi_obs_next[good_index]
            obs = hi_obs
            obs_next = obs_next[good_index]
            assert len(hi_obs) == len(hi_obs_next) == len(obs_next)
            selected_idx = selected_idx[good_index]
            print("selected data", len(good_index))

        sample_data = np.array([obs, obs_next, hi_obs, hi_obs_next])
        sample_data = torch.tensor(sample_data, dtype=torch.float32).to(self.device)
        obs, obs_next = self.representation(sample_data[0]), self.representation(sample_data[1])
        min_dist = torch.clamp((obs - obs_next).pow(2).mean(dim=1), min=0.)
        hi_obs, hi_obs_next = self.representation(sample_data[2]), self.representation(sample_data[3])
        max_dist = torch.clamp(1 - (hi_obs - hi_obs_next).pow(2).mean(dim=1), min=0.)
        print("loss", max_dist + min_dist)
        representation_loss = (min_dist + max_dist).mean()
        print("phi loss near start: ", representation_loss)
