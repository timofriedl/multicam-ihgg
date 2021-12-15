import numpy as np
import os.path
import re
import sys
from vae.multicamvae import MultiCamVae, ConcatEncodeVae, EncodeConcatVae, EncodeConcatEncodeVae

"""
Functions to load VAEs and goal sets from files.
"""

sys.path.append("./vae")


def load_goal_set(env: str, cams: list, img_width: int, img_height: int) -> np.ndarray:
    """
    Loads an existing goal set with the specified properties

    :param env: the name of the environment, e.g. FetchReach-v1
    :param cams: the list of camera names, e.g. ["front", "side"]
    :param img_width: the horizontal image size
    :param img_height: the vertical image size
    :return: a numpy array with shape [goal_set_size, len(cams), height, width, 3] containing the goal set images
             or None if the specified file does not exist
    """
    env_parts = re.findall("[A-Z][^A-Z]*", env.split("-")[0])
    env_lower = "_".join(env_parts).lower()
    size = str(img_width) if img_width == img_height else "{}_{}".format(img_width, img_height)
    goal_set_path = "./data/{}_Env/mvae_goal_set_{}_{}_{}.npy".format(env_parts[0], env_lower, "_".join(cams), size)

    if os.path.isfile(goal_set_path):
        return np.load(goal_set_path)
    else:
        return None


def load_multicam_vae(env: str, cams: list, mvae_mode: str, img_width: int, img_height: int) -> MultiCamVae:
    """
    Loads an existing Multi-Camera VAE with the specified properties

    :param env: the name of the environment, e.g. FetchReach-v1
    :param cams: the list of camera names, e.g. ["front", "side"]
    :param mvae_mode: the compression mode of the Multi-Camera VAE, e.g. "ec" for EncodeConcatVAE
    :param img_width: the horizontal image size
    :param img_height: the vertical image size
    :return: an instance of the corresponding MultiCamVAE or None if the file could not be found
    """
    env_parts = re.findall("[A-Z][^A-Z]*", env.split("-")[0])
    env_lower = "_".join(env_parts).lower()
    size = str(img_width) if img_width == img_height else "{}_{}".format(img_width, img_height)
    base_path = "./vae/models/mvae_model_{}_{}_{}_{}".format(env_lower, "_".join(cams), size, mvae_mode)
    print("Loading VAE from '{}'".format(base_path))

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


# A cache for the goal sets
goals = {}

# A cache for the vaes
vaes = {}


def import_goal_set(env: str, cams: list, img_width: int, img_height: int) -> np.ndarray:
    """
    Loads an existing Multi-Camera VAE with the specified properties, or returns an already cached VAE

    :param env: the name of the environment, e.g. FetchReach-v1
    :param cams: the list of camera names, e.g. ["front", "side"]
    :param img_width: the horizontal image size
    :param img_height: the vertical image size
    :return: a numpy array with shape [goal_set_size, len(cams), height, width, 3] containing the goal set images
    """
    key = (env, "_".join(cams), img_width, img_height)
    if key in goals:
        return goals[key]

    goal = load_goal_set(env, cams, img_width, img_height)
    goals[key] = goal
    return goal


def import_vae(env: str, cams: list, mvae_mode: str, img_width: int, img_height: int) -> MultiCamVae:
    """
    Loads an existing vae set with the specified properties, or returns an already cached goal set

    :param env: the name of the environment, e.g. FetchReach-v1
    :param cams: the list of camera names, e.g. ["front", "side"]
    :param mvae_mode: the compression mode of the Multi-Camera VAE, e.g. "ec" for EncodeConcatVAE
    :param img_width: the horizontal image size
    :param img_height: the vertical image size
    :return: an instance of the corresponding MultiCamVAE
    """
    key = (env, "_".join(cams), mvae_mode, img_width, img_height)
    if key in vaes:
        return vaes[key]

    vae = load_multicam_vae(env, cams, mvae_mode, img_width, img_height)
    vaes[key] = vae
    return vae
