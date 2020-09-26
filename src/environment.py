#!/usr/bin/env python
import math
import random
from math import sin, cos

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rospy
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan, JointState, Imu
from tf.transformations import euler_from_quaternion


class Env(object):
    def __init__(self, is_training):
        # matplotlib.rcParams.update({'font.size': 20})
        # plt.figure(figsize=(12, 10))
        plt.xlim(-3.6, 3.6)
        plt.ylim(-3.6, 3.6)
        plt.grid()
        self.goal_scat = plt.scatter([], [], c='orange', marker="*", s=200)
        self.pos_scat = plt.scatter([], [], c='green', marker="+", s=200)

        self.goal_range = {"x": (-3.6, 3.6),
                           "y": (-3.6, 3.6)}
        # Maybe max to 4
        self.obstacle_num = 2
        self.goal_distance = None
        self.collision = False
        self.arrive = False

        self.diagonal_dis = math.sqrt(2) * (3.6 + 3.8)
        self.position = Point()
        self.goal_pose = Pose()
        self.goal_pose.position.x = 0.
        self.goal_pose.position.y = 0.
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        _ = rospy.Subscriber('odom', Odometry, self._cb_odom)
        _ = rospy.Subscriber('scan', LaserScan, self._cb_laser)
        self.past_distance = 0.

        if is_training:
            self.threshold_arrive = 0.2
        else:
            self.threshold_arrive = 0.2

    def _cb_odom(self, data):
        self.odom = data

    def _get_odometry(self):
        self.position = self.odom.pose.pose.position
        self.position.x -= self.init_position.x
        self.position.y -= self.init_position.y

        orientation = self.odom.pose.pose.orientation
        q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
        yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))
        self.yaw = yaw - self.init_yaw

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

    def _get_state(self):
        self.scan_range = []
        min_range = 0.2
        self.collision = False
        self.arrive = False

        for laser_range in self._laser_data.ranges:
            if laser_range == float('Inf') or laser_range == 0.0:
                self.scan_range.append(3.5)
            else:
                self.scan_range.append(laser_range)

        if min_range > min(self.scan_range) > 0:
            self.collision = True

    def _set_reward(self):
        self._get_distance()
        if self.current_distance <= self.threshold_arrive:
            self.arrive = True
        self.bonus = 1 / self.current_distance

        self.state = [i / 3.5 for i in self.scan_range]

    def _get_distance(self):
        self.current_distance = math.hypot(self.goal_pose.position.x - self.position.x,
                                           self.goal_pose.position.y - self.position.y)
        return self.current_distance

    def _set_reward(self):
        distance_rate = (self.past_distance - self.current_distance)

        reward = 500. * distance_rate
        self.prev_distance = self.current_distance

        if self.collision:
            reward += -100.
        elif self.arrive:
            reward += 120.

        return reward

    def common_reset(self):
        # Set the target
        self.set_target()

    def step(self, action):
        self._plot_goal()
        linear_vel, ang_vel = action

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel / 2
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

<<<<<<< HEAD
        state = self._laser_scan()
=======
        # prev_time = rospy.get_time()
        # while rospy.get_time() - prev_time < 0.5 and self.collision is False:
        self._get_state()
            state.append(pa)
        self.prev_action = action

        reward = self._set_reward()

        return np.asarray(state), reward, self.collision, self.arrive

<<<<<<< HEAD
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
=======
    def _cb_laser(self, data):
        self._laser_data = data
>>>>>>> orig_odom

    def reset(self):
        self.common_reset()

<<<<<<< HEAD
        self.position = Point(x=0.0, y=0.0)
        self.yaw = 0.0
        self.prev_yaw = 0.0

        state = self._laser_scan()
=======
        self.init_position = self.odom.pose.pose.position
        orientation = self.odom.pose.pose.orientation
        q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
        self.init_yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))

        # First step we set current distance as past distance
        self.past_distance = self._get_distance()

        self._get_state()
>>>>>>> orig_odom

        # Add default past action (vel_cmd) to states.
        state.extend([0, 0])

        state = self._pack_state(state)

        return np.asarray(state)

<<<<<<< HEAD
    def _pack_state(self, state):
        self._calc_odom()
        self._calc_state()
        return state + [self.current_distance / self.diagonal_dis, self.yaw / 360, self.rel_theta / 360,
                        self.diff_angle / 360]

    def _plot_goal(self):
        self.goal_scat.set_offsets([self.goal_pose.position.x, self.goal_pose.position.y])
        self.tb3_scat.set_offsets([self.position.x, self.position.y])
=======
    def _pack_state(self):
        self._get_odometry()
        return self.state + [self.current_distance / self.diagonal_dis, self.yaw / 360, self.rel_theta / 360,
                             self.diff_angle / 360]

    def _plot_goal(self):
        self.goal_scat.set_offsets([self.goal_pose.position.x, self.goal_pose.position.y])
        self.pos_scat.set_offsets([self.position.x, self.position.y])
>>>>>>> orig_odom
        plt.draw()
        plt.pause(0.0001)

    def set_target(self):
        self.goal_pose.position.x = random.uniform(*self.goal_range["x"])
        self.goal_pose.position.y = random.uniform(*self.goal_range["y"])
<<<<<<< HEAD
=======
        plt.title("Goal_X: %.2f, Goal_Y: %.2f" % (self.goal_pose.position.x, self.goal_pose.position.y))
>>>>>>> orig_odom
        # self.goal_pose.position.x = 1
        # self.goal_pose.position.y = 1
