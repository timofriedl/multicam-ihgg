import os.path
import sys

import numpy as np

from vae.multicamvae import ConcatEncodeVae, EncodeConcatVae, EncodeConcatEncodeVae
from settings import img_width, img_height, cams, base_name, mvae_mode, latent_dim

sys.path.append("./vae")


def load_multicam_vae(scenario):
    size = str(img_width) if img_width == img_height else "{}_{}".format(img_width, img_height)
    base_path = "./vae/models/mvae_model_{}_{}_{}_{}".format(scenario, base_name, size, mvae_mode)

    try:
        if mvae_mode == "ce":
            return ConcatEncodeVae.load(base_path, len(cams), img_width, img_height, latent_dim)
        elif mvae_mode == "ec":
            return EncodeConcatVae.load(base_path, len(cams), img_width, img_height, latent_dim)
        elif mvae_mode == "ece":
            return EncodeConcatEncodeVae.load(base_path, len(cams), img_width, img_height, latent_dim)
        else:
            raise RuntimeError("Multicam mode '{}' does not exist.".format(mvae_mode))
    except FileNotFoundError:
        return None


def load_goal_set(scenario):
    env = scenario.split("_")[0].capitalize()
    size = str(img_width) if img_width == img_height else "{}_{}".format(img_width, img_height)
    goal_set_path = "./data/{}_Env/mvae_goal_set_{}_{}_{}.npy".format(env, scenario, base_name, size)

    if os.path.isfile(goal_set_path):
        return np.load(goal_set_path)
    else:
        return None


""" GOALS """
# fetch
goal_set_fetch_reach = load_goal_set("fetch_reach")
goal_set_fetch_push = load_goal_set("fetch_push")
goal_set_fetch_pick_0 = load_goal_set("fetch_pick")
goal_set_fetch_slide = load_goal_set("fetch_slide")

# hand
goal_set_reach = load_goal_set("hand_reach")
goal_set_block = load_goal_set("hand_block")
goal_set_egg = load_goal_set("hand_egg")
goal_set_pen = load_goal_set("hand_pen")

""" VAEs """
# fetch
vae_fetch_reach = load_multicam_vae("fetch_reach")
vae_fetch_push = load_multicam_vae("fetch_push")
vae_fetch_pick_0 = load_multicam_vae("fetch_pick")
vae_fetch_slide = load_multicam_vae("fetch_slide")

# hand
vae_hand_reach = load_multicam_vae("hand_reach")
vae_block = load_multicam_vae("hand_block")
vae_egg = load_multicam_vae("hand_egg")
vae_pen = load_multicam_vae("hand_pen")
