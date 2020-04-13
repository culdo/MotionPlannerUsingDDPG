#!/usr/bin/env python3
import math
import os
import random

import numpy as np
import rospy
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from ros_gazebo import Gazebo, dynamic_recfg
from sensor_msgs.msg import LaserScan

diagonal_dis = math.sqrt(2) * (3.6 + 3.8)
goal_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', 'gz_models', 'Target', 'model.sdf')


class Env(Gazebo):
    def __init__(self, is_training):
        super().__init__()

        # Make simulation faster.
        dynamic_recfg()

        self.goal_distance = None

        self.position = Pose()
        self.goal_position = Pose()
        self.goal_position.position.x = 0.
        self.goal_position.position.y = 0.
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self._get_odometry)
        self.past_distance = 0.
        if is_training:
            self.threshold_arrive = 0.2
        else:
            self.threshold_arrive = 0.4

    def _set_goal_distance(self):
        self.goal_distance = self._get_distance()
        self.past_distance = self.goal_distance

    def _get_odometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
        yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))

        if yaw >= 0:
            yaw = yaw
        else:
            yaw = yaw + 360

        rel_dis_x = round(self.goal_position.position.x - self.position.x, 1)
        rel_dis_y = round(self.goal_position.position.y - self.position.y, 1)

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
        rel_theta = round(math.degrees(theta), 2)

        diff_angle = abs(rel_theta - yaw)

        diff_angle = round(diff_angle, 2)

        self.rel_theta = rel_theta
        self.yaw = yaw
        self.diff_angle = diff_angle

    def _get_state(self, scan):
        scan_range = []
        yaw = self.yaw
        rel_theta = self.rel_theta
        diff_angle = self.diff_angle
        min_range = 0.2
        collision = False
        arrive = False

        for laser_range in scan.ranges:
            if laser_range == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(laser_range):
                scan_range.append(0)
            else:
                scan_range.append(laser_range)

        if min_range > min(scan_range) > 0:
            collision = True

        current_distance = self._get_distance()
        if current_distance <= self.threshold_arrive:
            # done = True
            arrive = True

        return scan_range, current_distance, yaw, rel_theta, diff_angle, collision, arrive

    def _get_distance(self):
        current_distance = math.hypot(self.goal_position.position.x - self.position.x,
                                      self.goal_position.position.y - self.position.y)
        return current_distance

    def _set_reward(self, done, arrive):
        current_distance = self._get_distance()
        distance_rate = (self.past_distance - current_distance)

        reward = 500. * distance_rate
        self.past_distance = current_distance

        if done:
            reward = -100.
            self.pub_cmd_vel.publish(Twist())

        if arrive:
            reward = 120.
            self.pub_cmd_vel.publish(Twist())
            self.delete_model("target")

        return reward

    def arrive_reset(self):
        self._spawn_target()
        self._set_goal_distance()

    def step(self, action, past_action):
        linear_vel = action[0]
        ang_vel = action[1]

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel / 4
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        arrive, diff_angle, done, rel_dis, rel_theta, state, yaw = self._scan()

        for pa in past_action:
            state.append(pa)

        state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 360]
        reward = self._set_reward(done, arrive)

        return np.asarray(state), reward, done, arrive

    def _scan(self):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except rospy.ROSException:
                pass
        state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self._get_state(data)
        state = [i / 3.5 for i in state]
        return arrive, diff_angle, done, rel_dis, rel_theta, state, yaw

    def reset(self):
        # Reset the env #
        self.delete_model('target')

        self.reset_world()

        # Build the targetz
        self._spawn_target()

        arrive, diff_angle, done, rel_dis, rel_theta, state, yaw = self._scan()
        self._set_goal_distance()

        state.append(0)
        state.append(0)

        state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 360]

        return np.asarray(state)

    def _spawn_target(self):
        with open(goal_model_dir, "r") as f:
            goal_sdf = f.read()
        self.goal_position.position.x = random.uniform(-3.6, 3.6)
        self.goal_position.position.y = random.uniform(-3.6, 3.6)
        self.spawn_sdf_model("target", goal_sdf, self.goal_position)
        self.unpause_physics()
