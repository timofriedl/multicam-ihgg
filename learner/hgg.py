import copy
import numpy as np
import sys
from algorithm.replay_buffer import Trajectory, goal_concat
from datetime import datetime
from envs import make_env
from envs.distance_graph import DistanceGraph
from envs.utils import goal_distance
from gym.envs.robotics.utils import capture_image_by_cam
from utils.gcc_utils import gcc_load_lib, c_double

"""
Code by James Li
https://github.com/hakrrr/I-HGG

Modifications by Timo Friedl
"""

# Training data settings
generate_train_data = False
dataset_size = 16384
use_lefe = True
lefe_duration = 2048

# Randomly generated positions for LEFE training data generation
if use_lefe:
    print("Generating random positions...")
    obj_xs = np.random.uniform(0.55, 2.05, dataset_size)  # 1.05, 1.55, dataset_size)
    obj_ys = np.random.uniform(0.00, 1.50, dataset_size)  # 0.40, 1.10, dataset_size)
    obj_zs = np.random.uniform(0.20, 0.80, dataset_size)  # 0.40, 0.80, dataset_size)
    obj_pos = np.array(list(zip(obj_xs, obj_ys, obj_zs)))
    del obj_xs
    del obj_ys
    del obj_zs
    # obj_pos = np.load("./vae/data/mvae_lefe_positions_fetch_pick_and_place_front_side_top_64_GRIP.npy")[:dataset_size]


class TrajectoryPool:
    def __init__(self, args, pool_length):
        self.args = args
        self.length = pool_length

        self.pool = []
        self.pool_init_state = []
        self.counter = 0

    def insert(self, trajectory, init_state):
        if self.counter < self.length:
            self.pool.append(trajectory.copy())
            self.pool_init_state.append(init_state.copy())
        else:
            self.pool[self.counter % self.length] = trajectory.copy()
            self.pool_init_state[self.counter % self.length] = init_state.copy()
        self.counter += 1

    def pad(self):
        if self.counter >= self.length:
            return copy.deepcopy(self.pool), copy.deepcopy(self.pool_init_state)
        pool = copy.deepcopy(self.pool)
        pool_init_state = copy.deepcopy(self.pool_init_state)
        while len(pool) < self.length:
            pool += copy.deepcopy(self.pool)
            pool_init_state += copy.deepcopy(self.pool_init_state)
        return copy.deepcopy(pool[:self.length]), copy.deepcopy(pool_init_state[:self.length])


class MatchSampler:
    def __init__(self, args, achieved_trajectory_pool):
        self.args = args
        self.env = make_env(args)
        self.env_test = make_env(args)
        self.dim = np.prod(self.env.reset()['achieved_goal'].shape)
        self.delta = self.env.distance_threshold

        self.length = args.episodes
        init_goal = self.env.reset()['achieved_goal'].copy()
        self.pool = np.tile(init_goal[np.newaxis, :], [self.length, 1]) + np.random.normal(0, self.delta,
                                                                                           size=(self.length, self.dim))
        self.init_state = self.env.reset()['observation'].copy()

        self.match_lib = gcc_load_lib('learner/cost_flow.c')
        self.achieved_trajectory_pool = achieved_trajectory_pool

        if self.args.graph:
            self.create_graph_distance()

        # estimating diameter
        # 1.6 Normal Hand Block # 3.7259765 Image-based Hand Block
        # 1.843 Normal Hand Egg # 5.692177 Image-based Hand Egg
        # 0.119 Normal Hand Reach # 8.03 Image-based Hand Reach
        # 1.7953 / 1.56 Normal Hand Pen # ? Image-based Hand Pen
        # 0.46 Normal Fetch Push # 5.2 Image-based Fetch Push
        # 0.25 Normal Fetch Reach # 5.263001 Image-based Fetch Reach
        # 0.648 Normal Fetch Slide # 2.92 Image-based Fetch Slide
        # 0.616 Normal Fetch Pick # 2.1 Image-based Fetch Pick

        self.max_dis = 0
        for i in range(1000):
            obs = self.env.reset()
            dis = self.get_graph_goal_distance(obs['achieved_goal'], obs['desired_goal'])
            if dis > self.max_dis:
                self.max_dis = dis
        print('Max Distance: ', self.max_dis)

    # Pre-computation of graph-based distances
    def create_graph_distance(self):
        obstacles = list()
        field = self.env.env.env.adapt_dict["field"]
        obstacles = self.env.env.env.adapt_dict["obstacles"]
        num_vertices = self.args.num_vertices
        graph = DistanceGraph(args=self.args, field=field, num_vertices=num_vertices, obstacles=obstacles)
        graph.compute_cs_graph()
        graph.compute_dist_matrix()
        self.graph = graph

    def get_graph_goal_distance(self, goal_a, goal_b):
        if self.args.graph:
            d, _ = self.graph.get_dist(goal_a, goal_b)
            if d == np.inf:
                d = 9999
            return d
        else:
            return np.linalg.norm(goal_a - goal_b, ord=2)

    def add_noise(self, pre_goal, noise_std=None):
        goal = pre_goal.copy()
        dim = 2 if self.args.env[:5] == 'Fetch' else self.dim
        if noise_std is None: noise_std = self.delta
        goal[:dim] += np.random.normal(0, noise_std, size=dim)
        return goal.copy()

    def sample(self, idx):
        if self.args.env[:5] == 'Fetch':
            return self.add_noise(self.pool[idx])
        else:
            return self.pool[idx].copy()

    def find(self, goal):
        res = np.sqrt(np.sum(np.square(self.pool - goal), axis=1))
        idx = np.argmin(res)
        if test_pool:
            self.args.logger.add_record('Distance/sampler', res[idx])
        return self.pool[idx].copy()

    def update(self, initial_goals, desired_goals):
        if self.achieved_trajectory_pool.counter == 0:
            self.pool = copy.deepcopy(desired_goals)
            return

        achieved_pool, achieved_pool_init_state = self.achieved_trajectory_pool.pad()
        candidate_goals = []
        candidate_edges = []
        candidate_id = []

        agent = self.args.agent
        achieved_value = []
        for i in range(len(achieved_pool)):
            obs = [goal_concat(achieved_pool_init_state[i], achieved_pool[i][j]) for j in
                   range(achieved_pool[i].shape[0])]
            feed_dict = {
                agent.raw_obs_ph: obs
            }
            value = agent.sess.run(agent.q_pi, feed_dict)[:, 0]
            value = np.clip(value, -1.0 / (1.0 - self.args.gamma), 0)
            achieved_value.append(value.copy())

        n = 0
        graph_id = {'achieved': [], 'desired': []}
        for i in range(len(achieved_pool)):
            n += 1
            graph_id['achieved'].append(n)
        for i in range(len(desired_goals)):
            n += 1
            graph_id['desired'].append(n)
        n += 1
        self.match_lib.clear(n)

        for i in range(len(achieved_pool)):
            self.match_lib.add(0, graph_id['achieved'][i], 1, 0)
        for i in range(len(achieved_pool)):
            for j in range(len(desired_goals)):
                if self.args.graph:
                    size = achieved_pool[i].shape[0]
                    res_1 = np.zeros(size)
                    for k in range(size):
                        res_1[k] = self.get_graph_goal_distance(achieved_pool[i][k], desired_goals[j])
                    res = res_1 - achieved_value[i] / (self.args.hgg_L / self.max_dis / (1 - self.args.gamma))
                else:
                    res = np.sqrt(np.sum(np.square(achieved_pool[i] - desired_goals[j]), axis=1)) - achieved_value[
                        i] / (self.args.hgg_L / self.max_dis / (1 - self.args.gamma))

                match_dis = np.min(res) + goal_distance(achieved_pool[i][0], initial_goals[j]) * self.args.hgg_c
                match_idx = np.argmin(res)

                edge = self.match_lib.add(graph_id['achieved'][i], graph_id['desired'][j], 1, c_double(match_dis))
                candidate_goals.append(achieved_pool[i][match_idx])
                candidate_edges.append(edge)
                candidate_id.append(j)
        for i in range(len(desired_goals)):
            self.match_lib.add(graph_id['desired'][i], n, 1, 0)

        match_count = self.match_lib.cost_flow(0, n)
        assert match_count == self.length

        explore_goals = [0] * self.length
        for i in range(len(candidate_goals)):
            if self.match_lib.check_match(candidate_edges[i]) == 1:
                explore_goals[candidate_id[i]] = candidate_goals[i].copy()
        assert len(explore_goals) == self.length
        self.pool = np.array(explore_goals)


class HGGLearner:
    def __init__(self, args):
        self.args = args
        self.env = make_env(args)
        self.env_test = make_env(args)

        self.env_List = []
        for i in range(args.episodes):
            self.env_List.append(make_env(args))

        self.achieved_trajectory_pool = TrajectoryPool(args, args.hgg_pool_size)
        self.sampler = MatchSampler(args, self.achieved_trajectory_pool)

        self.stop_hgg_threshold = self.args.stop_hgg_threshold
        self.stop = False
        self.learn_calls = 0

        self.cams = ["front", "side", "top"] if generate_train_data else self.args.cams

        self.count = 0
        if generate_train_data:
            self.train_data = np.empty(
                [dataset_size, len(self.cams), self.args.img_height, self.args.img_width, 3],
                dtype=np.uint8)

            if use_lefe:
                self.lefe_data = np.empty(self.train_data.shape, dtype=np.uint32)

                if "FetchPush" in args.env:
                    obj_pos = obj_pos[:, :2]

    @staticmethod
    def set_obj_pos(env, pos):
        object_qpos = env.sim.data.get_joint_qpos('object0:joint')
        object_qpos[:len(pos)] = pos
        env.sim.data.set_joint_qpos('object0:joint', object_qpos)
        env.sim.data.set_joint_qvel('object0:joint', np.zeros(6, dtype=np.float64))
        env.sim.forward()

    def generate_train_data(self, timestep, i):
        """
        Generates training data for MultiCamVae HGG training

        :param timestep: the current timestep
        :param i: the current environment index
        """
        # Capture images every 5 timesteps
        if timestep % 5 == 0 and self.count < dataset_size:
            # Setup viewer if missing
            env = self.env_List[i]
            if not hasattr(env, 'viewer'):
                env.viewer = env.sim.render_contexts[0]

            # For FetchPush set object location to random position
            if 'FetchPush' in env.args.env and np.random.randint(7) == 0:
                x = np.random.uniform(1.05, 1.55)
                y = np.random.uniform(0.40, 1.10)
                HGGLearner.set_obj_pos(env, [x, y])

            # For each camera capture image
            for c in range(len(self.cams)):
                self.train_data[self.count][c] = capture_image_by_cam(env, self.cams[c], self.args.img_width,
                                                                      self.args.img_height)

            # Print message every 100 image vectors
            if self.count % 100 == 0:
                print('Captured {} situations from {} perspectives'.format(self.count, len(self.cams)))

            self.count += 1

        if self.count == dataset_size:
            # Randomize dataset
            np.random.shuffle(self.train_data)

            # Save
            np.save('./vae/data/mvae_train_data_NEW_TODO_RENAME.npy', self.train_data)
            print('Finished!')
            self.count += 1
            sys.exit()

    def generate_train_data_lefe(self, timestep: int, i: int):
        """
        Generates training data for MultiCamVae HGG training with Long Exposure Feature Extraction (LEFE)

        :param timestep: the current timestep
        :param i: the current environment index
        """
        # Setup environment viewer if missing
        env = self.env_List[i]
        if not hasattr(env, 'viewer'):
            env.viewer = env.sim.render_contexts[0]

        # Take photo every 5 timesteps
        if self.count < dataset_size and timestep % 5 == 0:
            imgs = np.empty(self.train_data.shape[1:], dtype=np.uint8)

            # Iterate through some object positions
            for pos in range(self.count - lefe_duration, self.count):
                # Set object position
                HGGLearner.set_obj_pos(env, obj_pos[pos])

                # For all cameras capture image
                for c in range(len(self.cams)):
                    imgs[c] = capture_image_by_cam(env, self.cams[c], self.args.img_width, self.args.img_height)

                # Add images to lefe dataset to create blurr effect
                self.lefe_data[pos] += imgs.astype(np.uint32)

            # Set one single training data image
            self.train_data[self.count - 1] = imgs

            self.count += 1
            print('Captured {} situations from {} perspectives'.format(self.count, len(self.cams)))

            if self.count == dataset_size:
                # Finally normalize images
                self.lefe_data = (self.lefe_data / lefe_duration).astype(np.uint8)

                # Randomize order
                permut = np.random.permutation(dataset_size)
                self.train_data = self.train_data[permut]
                self.lefe_data = self.lefe_data[permut]

                # Save datasets
                now = datetime.now().time()
                np.save('./vae/data/mvae_train_data_NEW_TODO_RENAME_{}.npy'.format(now), self.train_data)
                np.save('./vae/data/mvae_lefe_data_NEW_TODO_RENAME_{}.npy'.format(now), self.lefe_data)

                print('Finished!')
                self.count += 1
                sys.exit()

    def learn(self, args, env, env_test, agent, buffer, write_goals=0):
        # Actual learning cycle takes place here!
        initial_goals = []
        desired_goals = []
        goal_list = []

        # get initial position and goal from environment for each epsiode
        for i in range(args.episodes):
            obs = self.env_List[i].reset()
            goal_a = obs['achieved_goal'].copy()
            goal_d = obs['desired_goal'].copy()
            initial_goals.append(goal_a.copy())
            desired_goals.append(goal_d.copy())

        # if HGG has not been stopped yet, perform crucial HGG update step here
        # by updating the sampler, a set of intermediate goals is provided and stored in sampler
        # based on distance to target goal distribution, similarity of initial states and expected reward (see paper)
        # by bipartite matching
        if not self.stop:
            self.sampler.update(initial_goals, desired_goals)

        achieved_trajectories = []
        achieved_init_states = []

        explore_goals = []
        test_goals = []
        inside = []

        for i in range(args.episodes):
            obs = self.env_List[i].get_obs()
            init_state = obs['observation'].copy()

            # if HGG has not been stopped yet, sample from the goals provided by the update step
            # if it has been stopped, the goal to explore is simply the one generated by the environment
            if not self.stop:
                explore_goal = self.sampler.sample(i)
            else:
                explore_goal = desired_goals[i]

            # store goals in explore_goals list to check whether goals are within goal space later
            explore_goals.append(explore_goal)
            # test_goals.append(self.env.generate_goal())
            test_goals.append(self.env.env.env._sample_goal())

            # Perform HER training by interacting with the environment
            self.env_List[i].goal = explore_goal.copy()
            if write_goals != 0 and len(goal_list) < write_goals:
                goal_list.append(explore_goal.copy())
            obs = self.env_List[i].get_obs()
            current = Trajectory(obs, args.mvae_mode)
            trajectory = [obs['achieved_goal'].copy()]
            for timestep in range(args.timesteps):
                # get action from the ddpg policy
                action = agent.step(obs, explore=True)
                # feed action to environment, get observation and reward
                obs, reward, done, info = self.env_List[i].step(action)
                trajectory.append(obs['achieved_goal'].copy())
                if timestep == args.timesteps - 1:
                    done = True
                current.store_step(action, obs, reward, done)
                if done:
                    break
                if generate_train_data:
                    if use_lefe:
                        self.generate_train_data_lefe(timestep, i)
                    else:
                        self.generate_train_data(timestep, i)

            achieved_trajectories.append(np.array(trajectory))
            achieved_init_states.append(init_state)
            # Trajectory is stored in replay buffer, replay buffer can be normal or EBP
            buffer.store_trajectory(current)
            agent.normalizer_update(buffer.sample_batch())

            if buffer.steps_counter >= args.warmup:
                for _ in range(args.train_batches):
                    # train with Hindsight Goals (HER step)
                    info = agent.train(buffer.sample_batch())
                    args.logger.add_dict(info)
                # update target network
                agent.target_update()

        selection_trajectory_idx = {}
        for i in range(self.args.episodes):
            # only add trajectories with movement to the trajectory pool --> use default (L2) distance measure!
            if goal_distance(achieved_trajectories[i][0], achieved_trajectories[i][-1]) > 0.01:
                selection_trajectory_idx[i] = True
        for idx in selection_trajectory_idx.keys():
            self.achieved_trajectory_pool.insert(achieved_trajectories[idx].copy(), achieved_init_states[idx].copy())

        # unless in first call:
        # Check which of the explore goals are inside the target goal space
        # target goal space is represented by a sample of test_goals directly generated from the environment
        # an explore goal is considered inside the target goal space, if it is closer than the distance_threshold to one of the test goals
        # (i.e. would yield a non-negative reward if that test goal was to be achieved)
        '''
        if self.learn_calls > 0:
            assert len(explore_goals) == len(test_goals)
            for ex in explore_goals:
                is_inside = 0
                for te in test_goals:
                    #TODO: check: originally with self.sampler.get_graph_goal_distance, now trying with goal_distance (L2)
                    if goal_distance(ex, te) <= self.env.env.env.distance_threshold:
                        is_inside = 1
                inside.append(is_inside)
            assert len(inside) == len(test_goals)
            inside_sum = 0
            for i in inside:
                inside_sum += i
            # If more than stop_hgg_threshold (e.g. 0.9) of the explore goals are inside the target goal space, stop HGG
            # and continue with normal HER.
            # By default, stop_hgg_threshold is disabled (set to a value > 1)
            average_inside = inside_sum / len(inside)
            self.args.logger.info("Average inside: {}".format(average_inside))
            if average_inside > self.stop_hgg_threshold:
                self.stop = True
                self.args.logger.info("Continue with normal HER")
        self.learn_calls += 1
        return goal_list if len(goal_list)>0 else None
        '''
