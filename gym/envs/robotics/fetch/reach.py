import numpy as np
import os
import random
import vae.utils
from gym import utils
from gym.envs.robotics import fetch_env
from gym.envs.robotics.utils import capture_image_by_cam
from torchvision.utils import save_image
from vae.import_vae import import_vae, import_goal_set

"""
Code by James Li
https://github.com/hakrrr/I-HGG

Modifications for MultiCamVae by Timo Friedl
"""

# edit envs/fetch/interval
# edit fetch_env: sample_goal
# edit fetch_env: get_obs
# edit here: sample_goal !
# edit here: dist_threshold (optional)
# edit robot_env: render (between hand and fetch env)
# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')


class FetchReachEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, args, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }

        self.args = args
        # Import VAE and goal set
        self.mvae = import_vae(self.args.env, self.args.cams, self.args.mvae_mode, self.args.img_width,
                               self.args.img_height)
        self.goal_set = import_goal_set(self.args.env, self.args.cams, self.args.img_width, self.args.img_height)

        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def _sample_goal(self) -> np.ndarray:
        """
        Samples a random goal from the corresponding goal set

        :return: a numpy array with shape [latent_dim] that contains the encoded representation of one goal image vector
        """
        self.goal_n = np.random.randint(self.goal_set.shape[0])
        goal_imgs = self.goal_set[self.goal_n]
        cat_img = np.concatenate(goal_imgs, axis=1)
        cat_img = vae.utils.image_to_tensor(cat_img)
        save_image(cat_img.cpu().view(-1, 3, self.mvae.height, cat_img.shape[3]), './videos/goal/goal.png')
        return vae.utils.tensor_to_np(self.mvae.encode(goal_imgs))

    def _get_image(self) -> np.ndarray:
        """
        Captures a vector of images

        :return: a numpy array with shape [num_cams, height, width, 3] that contains the captured images
        """
        images = np.empty([self.mvae.num_cams, self.mvae.height, self.mvae.width, 3])
        for c in range(self.mvae.num_cams):
            images[c] = capture_image_by_cam(self, self.args.cams[c], self.mvae.width, self.mvae.height)

        return vae.utils.tensor_to_np(self.mvae.encode(images))

    def _generate_state(self):
        goal = [random.uniform(1.15, 1.45), random.uniform(0.6, 1.0), 0.43]
        self._set_gripper(goal)
        self.sim.forward()
        for _ in range(15):
            self.sim.step()
        self._step_callback()

        # Image.fromarray(np.array(self.render(mode='rgb_array', width=300, height=300, cam_name="cam_0"))).show()
        # latent = self._get_image()
