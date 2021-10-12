import os
import random

import numpy as np
from torchvision.utils import save_image

import gym.utils
import vae.utils
from gym.envs.robotics import fetch_env
from gym.envs.robotics.utils import capture_image_by_cam
from vae.import_vae import import_vae, import_goal_set

# from vae.import_vae import goal_set_fetch_pick_1

# edit envs/fetch/interval
# edit fetch_env: sample_goal
# edit fetch_env: get_obs
# edit here: sample_goal !
# edit here: dist_threshold (optional)
# edit robot_env: render (between hand and fetch env)
# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')


class FetchPickAndPlaceEnv(fetch_env.FetchEnv, gym.utils.EzPickle):
    def __init__(self, args, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }

        self.args = args
        self.mvae = import_vae(self.args.env, self.args.cams, self.args.mvae_mode, self.args.img_width,
                               self.args.img_height)
        self.reach_mvae = import_vae("FetchReach-v1", self.args.cams, self.args.mvae_mode, self.args.img_width,
                                     self.args.img_height)
        self.goal_set = import_goal_set(self.args.env, self.args.cams, self.args.img_width, self.args.img_height)
        self.arm_factor = 0.5  # influence of arm position to observation vector

        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        gym.utils.EzPickle.__init__(self)

    '''
    def _viewer_setup(self):
    body_id = self.sim.model.body_name2id('robot0:gripper_link')
    lookat = self.sim.data.body_xpos[body_id]
    for idx, value in enumerate(lookat):
        self.viewer.cam.lookat[idx] = value
    self.viewer.cam.distance = 1.
    self.viewer.cam.azimuth = 180.
    self.viewer.cam.elevation = 90.
    '''

    def _sample_goal(self):
        goal_imgs = self.goal_set[np.random.randint(self.goal_set.shape[0])]
        cat_img = np.concatenate(goal_imgs, axis=1)
        cat_img = vae.utils.image_to_tensor(cat_img)
        save_image(cat_img.cpu().view(-1, 3, self.mvae.height, cat_img.shape[3]), 'videos/goal/goal.png')

        lat_puck = vae.utils.tensor_to_np(self.mvae.encode(goal_imgs))
        lat_arm = vae.utils.tensor_to_np(self.reach_mvae.encode(goal_imgs)) * self.arm_factor
        return np.concatenate((lat_puck, lat_arm))

    def _get_image(self):
        images = np.empty([self.mvae.num_cams, self.mvae.height, self.mvae.width, 3])
        for c in range(self.mvae.num_cams):
            images[c] = capture_image_by_cam(self, self.args.cams[c], self.mvae.width, self.mvae.height)

        lat_puck = vae.utils.tensor_to_np(self.mvae.encode(images))
        lat_arm = vae.utils.tensor_to_np(self.reach_mvae.encode(images)) * self.arm_factor
        return np.concatenate((lat_puck, lat_arm))

    def _generate_state(self):
        """ Only care about visible arm
        if self.visible:
            self._set_arm_visible(False)
            self.visible = False
        """
        goal = [random.uniform(1.15, 1.45), random.uniform(0.6, 1.0), 0.43]
        # goal = [1.3, .7, .432]
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        object_qpos[:3] = goal[:3]
        object_qpos[3:] = [1, 0, 0, 0]
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        for _ in range(15):
            self.sim.step()

        # Check if inside checkbox:
        pos = self.sim.data.get_joint_qpos('object0:joint').copy()
        if pos[0] < 1.15 or pos[0] > 1.45 or pos[1] < 0.6 or pos[1] > 1.0 or pos[2] < 0.42 or pos[2] > .7:
            self._generate_state()
        # Image.fromarray(np.array(self.render(mode='rgb_array', width=300, height=300, cam_name="cam_0"))).show()

        # latent = self._get_image()

        '''
        goal = [1.31, .71, .4321]
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        object_qpos[:3] = goal[:3]
        object_qpos[3:] = [1, 0, 0, 0]
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        for _ in range(2):
            self.sim.step()
        # Check if inside checkbox:
        pos1 = self.sim.data.get_joint_qpos('object0:joint')
        if pos1[0] < 1.15 or pos1[0] > 1.45 or pos1[1] < 0.6 or pos1[1] > 1.0 or pos1[2] < 0.42 or pos1[2] > .7:
            self._generate_state()
        latent1 = self._get_image()

        print(np.linalg.norm(pos - pos1, axis=-1))
        print(np.linalg.norm(latent1[:2] - latent[:2], axis=-1))
        print(np.linalg.norm(latent1[2:] - latent[2:], axis=-1))
        '''
        self._step_callback()
