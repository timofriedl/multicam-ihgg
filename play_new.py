import os
import subprocess

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from algorithm.replay_buffer import goal_based_process
from common import get_args
from envs import make_env
from gym.envs.robotics.utils import capture_image_by_cam
# Video export settings
from vae.import_vae import import_vae

res_y = 512
video_path = './videos/{}_{}_{}_{}.avi'


class Player:
    def __init__(self, args):
        # initialize environment
        self.args = args
        self.env = make_env(args)
        self.args.timesteps = self.env.env.env.spec.max_episode_steps
        self.env_test = make_env(args)
        self.info = []
        self.test_rollouts = 10
        self.timesteps = 50

        # get current policy from path (restore tf session + graph)
        self.play_dir = args.play_path
        self.play_epoch = args.play_epoch
        self.meta_path = "{}saved_policy-{}.meta".format(self.play_dir, self.play_epoch)
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(self.meta_path)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.play_dir))
        graph = tf.get_default_graph()
        self.raw_obs_ph = graph.get_tensor_by_name("raw_obs_ph:0")
        self.pi = graph.get_tensor_by_name("main/policy/net/pi/Tanh:0")

        """
        self.goal_imgs = []  
        self.positions = np.empty([4100, 7], dtype=np.float32)
        self.counter = 0  
        """

    def my_step_batch(self, obs):
        # compute actions from obs based on current policy by running tf session initialized before
        actions = self.sess.run(self.pi, {self.raw_obs_ph: obs})
        return actions

    def play(self):
        # play policy on env
        env = self.env
        if not hasattr(env, "mvae"):
            env.mvae = import_vae(env.args.env, env.args.cams, env.args.mvae_mode, env.args.img_width,
                                  env.args.img_height)

        res_x = int(res_y * self.args.img_width / self.args.img_height)

        acc_sum, obs = 0.0, []
        np.random.seed(18)
        for i in tqdm(range(self.test_rollouts)):
            obs.append(goal_based_process(env.reset()))
            # Get Goal Image & resize
            goal_img = Image.open('./videos/goal/goal.png')
            goal_img = goal_img.resize((res_x * len(self.args.cams), res_y))
            goal_img.putalpha(70)

            for timestep in range(self.timesteps):
                # body_id = env.sim.model.body_name2id('robot0:thbase')
                # x = env.sim.data.body_xpos[body_id]
                actions = self.my_step_batch(obs)
                obs, infos = [], []
                ob, _, _, info = env.step(actions[0])
                obs.append(goal_based_process(ob))
                infos.append(info)

                if not hasattr(env, 'viewer'):
                    env.viewer = env.sim.render_contexts[0]

                hq_imgs = np.empty([len(self.args.cams), res_y, res_x, 3], dtype=np.uint8)
                lq_imgs = np.empty([len(self.args.cams), env.mvae.height, env.mvae.width, 3], dtype=np.uint8)
                for c, cam in enumerate(self.args.cams):
                    hq_imgs[c] = capture_image_by_cam(env, cam, res_x, res_y)
                    lq_imgs[c] = capture_image_by_cam(env, cam, env.mvae.width, env.mvae.height)

                """
                if timestep % 5 == 0:
                    goal_imgs = np.empty([3, 64, 64, 3])
                    goal_imgs[0] = capture_image_by_cam(env, "front", 64, 64)
                    goal_imgs[1] = capture_image_by_cam(env, "side", 64, 64)
                    goal_imgs[2] = capture_image_by_cam(env, "top", 64, 64)
                    self.goal_imgs.append(goal_imgs)
                    pos = env.sim.data.get_joint_qpos('object0:joint')
                    self.positions[self.counter] = pos
                    self.counter += 1
                """

                rgb_array = np.concatenate(hq_imgs, axis=1)

                recs = env.mvae.forward(lq_imgs)
                recs_array = np.concatenate(recs, axis=1)

                path = 'videos/frames/frame_' + str(i * self.timesteps + timestep) + '.png'

                # Overlay Images
                bg = Image.fromarray(rgb_array)
                bg.putalpha(288)
                bg = Image.alpha_composite(bg, goal_img)
                rc = Image.fromarray(recs_array).resize((len(self.args.cams) * res_x, res_y))
                self.get_concat_v(bg, rc).save(path)

    def make_video(self, path_to_folder, ext_end, path):
        image_files = [f for f in os.listdir(path_to_folder) if f.endswith(ext_end)]
        image_files.sort(key=lambda x: int(x.replace('frame_', '').replace(ext_end, '')))
        img_array = []
        for i in tqdm(range(len(image_files))):
            filename = image_files[i]
            img = cv2.imread(path_to_folder + filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), 4, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

    def get_concat_h(self, im1, im2):
        dst = Image.new('RGB', (im1.img_width + im2.img_width, im1.img_height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.img_width, 0))
        return dst

    def get_concat_v(self, im1, im2):
        dst = Image.new('RGB', (im1.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst


if __name__ == "__main__":
    try:
        args = get_args()
        player = Player(args)
        print("Playing...")
        player.play()

        """
        np.save("./data/Fetch_Env/mvae_goal_set_NEW.npy", np.array(player.goal_imgs, dtype=np.uint8)) 
        np.save("./data/Fetch_Env/mvae_goal_set_NEW_positions.npy", player.positions)
        """

        path = video_path.format(args.env, args.base_name, args.mvae_mode, args.play_epoch)
        print("Making video...")
        player.make_video('videos/frames/', '.png', path)
        subprocess.call(['vlc', path])
    except RuntimeError as err:
        msg = str(err)
        if 'Failed to initialize OpenGL' in msg:
            print('{0}. Please try to execute "unset LD_PRELOAD" before running training.'
                  .format(msg))
            exit(0)
        else:
            raise err
