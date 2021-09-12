import os
import random

import numpy as np
from torchvision.utils import save_image

import vae.utils
from gym import utils
from gym.envs.robotics import fetch_env
from gym.envs.robotics.utils import capture_image_by_cam
from settings import cams
from vae.import_vae import goal_set_fetch_push as goal_set, vae_fetch_push as mvae

# edit envs/fetch/interval
# edit fetch_env: sample_goal
# edit fetch_env: get_obs
# edit here: sample_goal
# edit here: dist_threshold (0.05 original)
# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push.xml')


class FetchPushEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.63, 0.4, 1., 0., 0., 0.],  # origin 0.53
        }

        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def _sample_goal(self) -> np.ndarray:
        goal_imgs = goal_set[np.random.randint(goal_set.shape[0])]
        cat_img = np.concatenate(goal_imgs, axis=1)
        cat_img = vae.utils.image_to_tensor(cat_img)
        save_image(cat_img.cpu().view(-1, 3, mvae.height, cat_img.shape[3]), 'videos/goal/goal.png')
        return vae.utils.tensor_to_np(mvae.encode(goal_imgs))

    def _get_image(self) -> np.ndarray:
        images = np.empty([mvae.num_cams, mvae.height, mvae.width, 3])
        for c in range(mvae.num_cams):
            images[c] = capture_image_by_cam(self, cams[c], mvae.width, mvae.height)

        """
        r = np.random.randint(1000) == 0
        if r:
            cat_img = np.concatenate(images, axis=1)
            cat_img = vae.utils.image_to_tensor(cat_img)
            save_image(cat_img.cpu().view(-1, 3, h, cat_img.shape[3]), 'ach_fetch_push.png')  # TODO
        """

        lat = mvae.encode(images)

        """
        if r:
            recon_img = vae_fetch_push.decode(lat)
            recon_img = np.concatenate(recon_img, axis=1)
            recon_img = vae.utils.image_to_tensor(recon_img)
            save_image(recon_img.cpu().view(-1, 3, h, recon_img.shape[3]), 'rec_fetch_push.png')
        """

        return vae.utils.tensor_to_np(lat)

    def _generate_state(self):
        """ Only caring about visible arm
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
