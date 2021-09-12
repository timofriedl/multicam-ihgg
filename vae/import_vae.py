import os.path
import re
import sys

import numpy as np

from vae.multicamvae import ConcatEncodeVae, EncodeConcatVae, EncodeConcatEncodeVae

sys.path.append("./vae")


def load_goal_set(env, cams, img_width, img_height):
    env_parts = re.findall("[A-Z][^A-Z]*", env.split("-")[0])
    env_lower = "_".join(env_parts).lower()
    size = str(img_width) if img_width == img_height else "{}_{}".format(img_width, img_height)
    goal_set_path = "./data/{}_Env/mvae_goal_set_{}_{}_{}.npy".format(env_parts[0], env_lower, "_".join(cams), size)

    if os.path.isfile(goal_set_path):
        return np.load(goal_set_path)
    else:
        return None


def load_multicam_vae(env, cams, mvae_mode, img_width, img_height):
    env_parts = re.findall("[A-Z][^A-Z]*", env.split("-")[0])
    env_lower = "_".join(env_parts).lower()
    size = str(img_width) if img_width == img_height else "{}_{}".format(img_width, img_height)
    base_path = "./vae/models/mvae_model_{}_{}_{}_{}".format(env_lower, "_".join(cams), size, mvae_mode)

    try:
        if mvae_mode == "ce":
            return ConcatEncodeVae.load(base_path, len(cams))
        elif mvae_mode == "ec":
            return EncodeConcatVae.load(base_path, len(cams))
        elif mvae_mode == "ece":
            return EncodeConcatEncodeVae.load(base_path, len(cams))
        else:
            raise RuntimeError("Multicam mode '{}' does not exist.".format(mvae_mode))
    except FileNotFoundError:
        return None


""" GOALS """
goals = {}

"""
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
"""

""" VAEs """
vaes = {}

"""
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
"""


def import_goal_set(env, cams, img_width, img_height):
    key = (env, "_".join(cams), img_width, img_height)
    if key in goals:
        return goals[key]

    goal = load_goal_set(env, cams, img_width, img_height)
    goals[key] = goal
    return goal


def import_vae(env, cams, mvae_mode, img_width, img_height):
    key = (env, "_".join(cams), mvae_mode, img_width, img_height)
    if key in vaes:
        return vaes[key]

    vae = load_multicam_vae(env, cams, mvae_mode, img_width, img_height)
    vaes[key] = vae
    return vae
