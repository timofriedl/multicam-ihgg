import numpy as np

from gym.envs.robotics import robot_env


class HandEnv(robot_env.RobotEnv):
    def __init__(self, model_path, n_substeps, initial_qpos, relative_control):
        self.relative_control = relative_control
        self.ok = False
        super(HandEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=20,
            initial_qpos=initial_qpos)

    # RobotEnv methods
    # ----------------------------

    def _set_action(self, action):
        assert action.shape == (20,)

        ctrlrange = self.sim.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.
        if self.relative_control:
            actuation_center = np.zeros_like(action)
            for i in range(self.sim.data.ctrl.shape[0]):
                actuation_center[i] = self.sim.data.get_joint_qpos(
                    self.sim.model.actuator_names[i].replace(':A_', ':'))
            for joint_name in ['FF', 'MF', 'RF', 'LF']:
                act_idx = self.sim.model.actuator_name2id(
                    'robot0:A_{}J1'.format(joint_name))
                actuation_center[act_idx] += self.sim.data.get_joint_qpos(
                    'robot0:{}J0'.format(joint_name))
        else:
            actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
        self.sim.data.ctrl[:] = actuation_center + action * actuation_range
        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

    def render(self, mode='human', width=500, height=500, cam_name='cam_0'):
        return super(HandEnv, self).render(mode, width, height, cam_name)
