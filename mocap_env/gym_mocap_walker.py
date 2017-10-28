import numpy as np
from gym.envs.registration import EnvSpec
import os

from roboschool.scene_abstract import cpp_household

from roboschool.gym_mujoco_walkers import RoboschoolForwardWalkerMujocoXML


class RoboschoolMocapHumanoid(RoboschoolForwardWalkerMujocoXML):
    foot_list = ["rfoot", "lfoot"]
    TASK_WALK, TASK_STAND_UP, TASK_ROLL_OVER, TASKS = range(4)

    def __init__(self, model_xml=os.path.join(os.path.dirname(__file__), 'humanoid_mocap.xml')):
        RoboschoolForwardWalkerMujocoXML.__init__(self, model_xml, 'torso', action_dim=56, obs_dim=44, power=0.41)
        # 56 joints, ? of them important for walking (hip, knee), others may as well be turned off, 17/4 = 4.25
        # TODO fix these cost parameters
        self.electricity_cost = 4.25 * RoboschoolForwardWalkerMujocoXML.electricity_cost
        self.stall_torque_cost = 4.25 * RoboschoolForwardWalkerMujocoXML.stall_torque_cost
        self.initial_z = 0.8

    def robot_specific_reset(self):
        RoboschoolForwardWalkerMujocoXML.robot_specific_reset(self)
        self.motor_names = [
            'upperneck_x', 'upperneck_y', 'upperneck_z', 'rfemur_x', 'rfemur_y', 'rfemur_z', 'lradius_x', 'rwrist_y',
            'lthumb_x', 'lthumb_z', 'lfemur_x', 'lfemur_y', 'lfemur_z', 'rthumb_x', 'rthumb_z', 'lhand_x', 'lhand_z',
            'rfingers_x', 'lowerback_x', 'lowerback_y', 'lowerback_z', 'rtoes_x', 'lfingers_x', 'head_x', 'head_y',
            'head_z', 'ltibia_x', 'ltoes_x', 'lhumerus_x', 'lhumerus_y', 'lhumerus_z', 'thorax_x', 'thorax_y',
            'thorax_z', 'lowerneck_x', 'lowerneck_y', 'lowerneck_z', 'upperback_x', 'upperback_y', 'upperback_z',
            'rtibia_x', 'rradius_x', 'rfoot_x', 'rfoot_z', 'rhumerus_x', 'rhumerus_y', 'rhumerus_z', 'lfoot_x',
            'lfoot_z', 'lwrist_y', 'rhand_x', 'rhand_z', 'lclavicle_y', 'lclavicle_z', 'rclavicle_y', 'rclavicle_z']
        self.motor_power = [100] * len(self.motor_names)
        self.motors = [self.jdict[n] for n in self.motor_names]
        self.humanoid_task()

    def humanoid_task(self):
        self.set_initial_orientation(self.TASK_WALK, yaw_center=0, yaw_random_spread=np.pi / 16)

    def set_initial_orientation(self, task, yaw_center, yaw_random_spread):
        self.task = task
        cpose = cpp_household.Pose()
        yaw = yaw_center # + self.np_random.uniform(low=-yaw_random_spread, high=yaw_random_spread)
        #XXX currently, only walking is supported
        if task == self.TASK_WALK:
            pitch = 0
            roll = 0
            cpose.set_xyz(self.start_pos_x, self.start_pos_y, self.start_pos_z + 1.05)
        elif task == self.TASK_STAND_UP:
            pitch = np.pi / 2
            roll = 0
            cpose.set_xyz(self.start_pos_x, self.start_pos_y, self.start_pos_z + 0.45)
        elif task == self.TASK_ROLL_OVER:
            pitch = np.pi * 3 / 2 - 0.15
            roll = 0
            cpose.set_xyz(self.start_pos_x, self.start_pos_y, self.start_pos_z + 0.22)
        else:
            assert False
        cpose.set_rpy(roll, pitch, yaw)
        self.cpp_robot.set_pose_and_speed(cpose, 0, 0, 0)
        self.initial_z = 0.8

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        for i, m, power in zip(range(len(self.motors)), self.motors, self.motor_power):
            m.set_motor_torque(float(power * self.power * np.clip(a[i], -1, +1)))

    def alive_bonus(self, z, pitch):
        # TODO make sure this is correct
        return +2 if z > 0.78 else -1  # 2 here because 56 joints produce a lot of electricity cost just from policy noise, living must be better than dying
