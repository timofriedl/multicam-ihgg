import numpy as np

from gym import error

"""
Code adopted from James Li
https://github.com/hakrrr/I-HGG

Modifications for MultiCamVae by Timo Friedl
"""

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e))


def robot_get_obs(sim):
    """Returns all joint positions and velocities associated with
    a robot.
    """
    if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names if n.startswith('robot')]
        return (
            np.array([sim.data.get_joint_qpos(name) for name in names]),
            np.array([sim.data.get_joint_qvel(name) for name in names]),
        )
    return np.zeros(0), np.zeros(0)


def ctrl_set_action(sim, action):
    """For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    if sim.model.nmocap > 0:
        _, action = np.split(action, (sim.model.nmocap * 7,))
    if sim.data.ctrl is not None:
        for i in range(action.shape[0]):
            if sim.model.actuator_biastype[i] == 0:
                sim.data.ctrl[i] = action[i]
            else:
                idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
                sim.data.ctrl[i] = sim.data.qpos[idx] + action[i]


def mocap_set_action(sim, action):
    """The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    constraint optimizer tries to center the welded body on the mocap.
    """
    if sim.model.nmocap > 0:
        action, _ = np.split(action, (sim.model.nmocap * 7,))
        action = action.reshape(sim.model.nmocap, 7)

        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]

        reset_mocap2body_xpos(sim)
        sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta
        sim.data.mocap_quat[:] = sim.data.mocap_quat + quat_delta


def reset_mocap_welds(sim):
    """Resets the mocap welds that we use for actuation.
    """
    if sim.model.nmocap > 0 and sim.model.eq_data is not None:
        for i in range(sim.model.eq_data.shape[0]):
            if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                sim.model.eq_data[i, :] = np.array(
                    [0., 0., 0., 1., 0., 0., 0.])
    sim.forward()


def reset_mocap2body_xpos(sim):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if (sim.model.eq_type is None or
            sim.model.eq_obj1id is None or
            sim.model.eq_obj2id is None):
        return
    for eq_type, obj1_id, obj2_id in zip(sim.model.eq_type,
                                         sim.model.eq_obj1id,
                                         sim.model.eq_obj2id):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue

        mocap_id = sim.model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert (mocap_id != -1)
        sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]


def capture_image(env, azimuth, eleviation, distance, lookat_0, lookat_1, lookat_2, width, height):
    """
    Captures an image from the specified perspective

    :param env: the simulation environment
    :param azimuth: the horizontal angle of the camera
    :param eleviation: the vertical angle of the camera
    :param distance: the distance between camera and specified lookat position
    :param lookat_0: the x coordinate of the lookat position
    :param lookat_1: the y coordinate of the lookat position
    :param lookat_2: the z coordinate of the lookat position
    :param width: the horizontal size of the output image
    :param height: the vertical size of the output image
    :return: a numpy array with shape [height, width, 3]
    """
    cam = env.viewer.cam
    cam.azimuth = azimuth
    cam.elevation = eleviation
    cam.distance = distance
    cam.lookat[0] = lookat_0
    cam.lookat[1] = lookat_1
    cam.lookat[2] = lookat_2
    rgb_array = np.array(env.sim.render(mode='offscreen', width=width, height=height))

    if np.all((rgb_array == 0)):
        return capture_image(env, azimuth, eleviation, distance, lookat_0, lookat_1, lookat_2, width, height)

    return np.rot90(rgb_array, 2)


def capture_image_by_cam(env, cam_name, width, height):
    """
    Captures an image from the specified camera

    :param env: the simulation environment
    :param cam_name: the name of the camera, e.g. "front" or "side"
    :param width: the horizontal size of the output image
    :param height: the vertical size of the output image
    :return: a numpy array with shape [height, width, 3]
    """
    if cam_name == "front":
        return capture_image(env, 180, 0, 1.1, 1.3, .75, .6, width, height)
    elif cam_name == "side":
        return capture_image(env, 90, 0, 1.1, 1.3, .75, .6, width, height)
    elif cam_name == "top":
        return capture_image(env, 0, -90, 0.8, 1.3, .75, .6, width, height)
    elif cam_name == "overview":
        return capture_image(env, 150, -15, 2.0, 1.0, .74, .4, width, height)
    else:
        raise RuntimeError("Camera '{}' does not exist.".format(cam_name))
