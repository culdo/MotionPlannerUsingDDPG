#!/usr/bin/env python3
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import rospy
from geometry_msgs.msg import Twist, Pose, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class Env:
    def __init__(self, is_training):
        super().__init__()

        self.goal_range = {"x": (-3.6, 3.6),
                           "y": (-3.6, 3.6)}
        # Maybe max to 4
        self.obstacle_num = 2
        self.goal_distance = None

        self.diagonal_dis = math.sqrt(2) * (3.6 + 3.8)
        self.position = Point()
        self.goal_position = Point()
        self.goal_position.x = 0.
        self.goal_position.y = 0.
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self._get_odometry)
        self.past_distance = 0.

        if is_training:
            self.threshold_arrive = 0.2
        else:
            self.threshold_arrive = 0.4

    def _get_odometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
        self.yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))

        if self.yaw >= 0:
            self.yaw = self.yaw
        else:
            self.yaw = self.yaw + 360

        rel_dis_x = round(self.goal_position.x - self.position.x, 1)
        rel_dis_y = round(self.goal_position.y - self.position.y, 1)

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

    def _get_state(self, scan):
        self.scan_range = []
        min_range = 0.2
        self.collision = False
        self.arrive = False

        for laser_range in scan.ranges:
            if laser_range == float('Inf') or laser_range == 0.0:
                self.scan_range.append(3.5)
            else:
                self.scan_range.append(laser_range)

        if min_range > min(self.scan_range) > 0:
            print(min(self.scan_range))
            self.collision = True

        self._get_distance()
        if self.current_distance <= self.threshold_arrive:
            self.arrive = True

    def _get_distance(self):
        self.current_distance = math.hypot(self.goal_position.x - self.position.x,
                                           self.goal_position.y - self.position.y)
        return self.current_distance

    def _set_reward(self):
        distance_rate = (self.past_distance - self.current_distance)

        reward = 500. * distance_rate
        self.past_distance = self.current_distance

        if self.collision:
            reward = -100.
        elif self.arrive:
            reward = 120.

        self.pub_cmd_vel.publish(Twist())

        return reward

    def set_obstacle(self):
        for i in range(self.obstacle_num):
            pose = Pose()
            goal_x = self.goal_position.x
            goal_y = self.goal_position.y
            tb3_x = self.position.x
            tb3_y = self.position.y
            while True:
                box_x = random.uniform(-3.6, 3.6)
                box_y = random.uniform(-3.6, 3.6)
                if goal_x + 2.0 > box_x > goal_x - 2.0 and \
                        goal_y + 2.0 > box_y > goal_y - 2.0 and \
                        not goal_x + 0.5 > box_x > goal_x - 0.5 and \
                        not goal_y + 0.5 > box_y > goal_y - 0.5 and \
                        not tb3_x + 0.5 > box_x > tb3_x - 0.5 and \
                        not tb3_y + 0.5 > box_y > tb3_y - 0.5:
                    break
            pose.position.x = box_x
            pose.position.y = box_y
            pose.position.z = 1.0

    def common_reset(self):
        # Set the target
        self.set_target()
        # self.set_obstacle()
        # First step we set current distance as past distance
        self.past_distance = self._get_distance()

    def step(self, action, past_action):
        self._plot_goal()
        linear_vel = action[0]
        ang_vel = action[1]

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel / 4
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        self._laser_scan()

        for pa in past_action:
            self.state.append(pa)

        state = self._pack_state()
        reward = self._set_reward()

        return np.asarray(state), reward, self.collision, self.arrive

    def _laser_scan(self):
        while True:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
                break
            except rospy.ROSException:
                pass
        self._get_state(data)
        self.state = [i / 3.5 for i in self.scan_range]

    def reset(self):
        self.common_reset()

        self._laser_scan()

        # Add default past action (vel_cmd) to states.
        self.state.extend([0, 0])

        state = self._pack_state()

        return np.asarray(state)

    def _pack_state(self):
        return self.state + [self.current_distance / self.diagonal_dis, self.yaw / 360, self.rel_theta / 360,
                             self.diff_angle / 360]

    def _plot_goal(self):
        plt.clf()
        plt.xlim(-3.6, 3.6)
        plt.ylim(-3.6, 3.6)
        plt.scatter(self.goal_position.x, self.goal_position.y, c='orange', marker="*", s=200)
        print(self.position.x, self.position.y)
        plt.scatter(self.position.x, self.position.y, c='green', marker="+", s=200)
        plt.pause(0.1)

    def set_target(self):
        # self.goal_position.x = random.uniform(*self.goal_range["x"])
        # self.goal_position.y = random.uniform(*self.goal_range["y"])
        self.goal_position.x = 1
        self.goal_position.y = 1
