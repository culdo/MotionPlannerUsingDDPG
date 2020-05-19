#!/usr/bin/env python3
import math
import random
from math import sin, cos

import matplotlib.pyplot as plt
import numpy as np
import rospy
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan, JointState, Imu
from tf.transformations import euler_from_quaternion


class Env(object):
    def __init__(self, is_training):
        plt.xlim(-3.6, 3.6)
        plt.ylim(-3.6, 3.6)
        self.goal_scat = plt.scatter([], [], c='orange', marker="*", s=200)
        self.tb3_scat = plt.scatter([], [], c='green', marker="+", s=200)

        self.goal_range = {"x": (-3.6, 3.6),
                           "y": (-3.6, 3.6)}
        # Maybe max to 4
        self.obstacle_num = 2
        self.goal_distance = None

        self.diagonal_dis = math.sqrt(2) * (3.6 + 3.8)
        self.position = Point(x=0.0, y=0.0)
        self.goal_pose = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.prev_distance = 0.

        if is_training:
            self.threshold_arrive = 0.2
        else:
            self.threshold_arrive = 0.4
        self.prev_action = [0., 0.]
        self.prev_wheel = None
        self.prev_yaw = 0.0
        self.yaw = 0.0

    def _get_joint(self):
        joint_states = rospy.wait_for_message('/joint_states', JointState)
        # print("joint_states.position: %s" % str(joint_states.position))
        wheel_position = joint_states.position
        if self.prev_wheel is None:
            self.prev_wheel = joint_states.position
        # print("wheel_position:%s" % str(self.wheel_position))
        return wheel_position

    def _get_raw_yaw(self):
        while True:
            try:
                imu = rospy.wait_for_message('/imu', Imu, timeout=5)
                break
            except rospy.ROSException:
                print("Not got imu data...")

        quaternion = (
            imu.orientation.x,
            imu.orientation.y,
            imu.orientation.z,
            imu.orientation.w)
        yaw_rad = euler_from_quaternion(quaternion)[-1]
        print("yaw:%.4f" % yaw_rad)
        return yaw_rad

    def _calc_odom(self, wheel_radius=0.033):
        wheel_position = self._get_joint()
        yaw_rad = self._get_raw_yaw()

        diff_l = wheel_position[0] - self.prev_wheel[0]
        diff_r = wheel_position[1] - self.prev_wheel[1]
        # print("diff_l: %s, diff_r: %s" % (diff_l, diff_r))
        delta_s = wheel_radius * (diff_l + diff_r) / 2.0
        self.prev_wheel = wheel_position

        delta_yaw = yaw_rad - self.prev_yaw

        self.position.x += delta_s * cos(self.prev_yaw + (delta_yaw / 2.0))
        self.position.y += delta_s * sin(self.prev_yaw + (delta_yaw / 2.0))

        self.prev_yaw = yaw_rad

        self.yaw = np.rad2deg(yaw_rad)
        if self.yaw >= 0:
            self.yaw = self.yaw
        else:
            self.yaw = self.yaw + 360

    def _calc_state(self):
        rel_dis_x = round(self.goal_pose.position.x - self.position.x, 1)
        rel_dis_y = round(self.goal_pose.position.y - self.position.y, 1)
        # Calculate the angle between robot and target
        if rel_dis_x > 0 and rel_dis_y > 0:
            theta = math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x > 0 and rel_dis_y < 0:
            theta = 2 * math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y < 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y > 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x == 0 and rel_dis_y > 0:
            theta = 1 / 2 * math.pi
        elif rel_dis_x == 0 and rel_dis_y < 0:
            theta = 3 / 2 * math.pi
        elif rel_dis_y == 0 and rel_dis_x > 0:
            theta = 0
        else:
            theta = math.pi
        self.rel_theta = round(math.degrees(theta), 2)
        self.diff_angle = abs(self.rel_theta - self.yaw)
        self.diff_angle = round(self.diff_angle, 2)

    def _get_distance(self):
        self.current_distance = math.hypot(self.goal_pose.position.x - self.position.x,
                                           self.goal_pose.position.y - self.position.y)

    def _set_reward(self):
        self._get_distance()
        if self.current_distance <= self.threshold_arrive:
            self.arrive = True

        distance_rate = (self.prev_distance - self.current_distance)

        reward = 500. * distance_rate
        self.prev_distance = self.current_distance

        if self.collision:
            reward = -100.
        elif self.arrive:
            reward = 120.

        # self.pub_cmd_vel.publish(Twist())

        return reward

    def common_reset(self):
        # Set the target
        self.set_target()
        # First step we set current distance as past distance
        self._get_distance()
        self.prev_distance = self.current_distance

    def step(self, action):
        self._plot_goal()
        linear_vel, ang_vel = action

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel / 2
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        state = self._laser_scan()

        for pa in self.prev_action:
            state.append(pa)
        self.prev_action = action

        state = self._pack_state(state)
        reward = self._set_reward()

        return np.asarray(state), reward, self.collision, self.arrive

    def _laser_scan(self):
        while True:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
                break
            except rospy.ROSException:
                pass
        self.scan_range = []
        min_range = 0.2
        self.collision = False
        self.arrive = False
        for laser_range in data.ranges:
            if laser_range == float('Inf') or laser_range == 0.0:
                self.scan_range.append(3.5)
            else:
                self.scan_range.append(laser_range)
        if min_range > min(self.scan_range) > 0:
            # print(min(self.scan_range))
            self.collision = True
        return [i / 3.5 for i in self.scan_range]

    def reset(self):
        self.common_reset()

        self.position = Point(x=0.0, y=0.0)
        self.yaw = 0.0
        self.prev_yaw = 0.0

        state = self._laser_scan()

        # Add default past action (vel_cmd) to states.
        state.extend([0, 0])

        state = self._pack_state(state)

        return np.asarray(state)

    def _pack_state(self, state):
        self._calc_odom()
        self._calc_state()
        return state + [self.current_distance / self.diagonal_dis, self.yaw / 360, self.rel_theta / 360,
                        self.diff_angle / 360]

    def _plot_goal(self):
        self.goal_scat.set_offsets([self.goal_pose.position.x, self.goal_pose.position.y])
        self.tb3_scat.set_offsets([self.position.x, self.position.y])
        plt.draw()
        plt.pause(0.0001)

    def set_target(self):
        self.goal_pose.position.x = random.uniform(*self.goal_range["x"])
        self.goal_pose.position.y = random.uniform(*self.goal_range["y"])
        # self.goal_pose.position.x = 1
        # self.goal_pose.position.y = 1
