#!/usr/bin/env python3
import os
import random

from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose
from ros_gazebo import Gazebo

from environment import Env


class SimEnv(Env, Gazebo):
    def __init__(self, is_training, is_hard=True):
        Env.__init__(self, is_training)
        Gazebo.__init__(self)
        self.is_hard = is_hard
        spp = self.get_physics_properties()
        self.set_physics_properties(time_step=spp.time_step,
                                    max_update_rate=5000.0,
                                    gravity=spp.gravity,
                                    ode_config=spp.ode_config)
        self._spawn_models()

    def _spawn_models(self):
        model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', 'gz_models')
        goal_model = os.path.join(model_dir, 'target', 'model.sdf')
        with open(goal_model, "r") as f:
            goal_sdf = f.read()
        self.spawn_sdf_model("target", goal_sdf, "target", self.goal_pose, 'world')
        # Spawn obstacle boxes
        if self.is_hard:
            self._spawn_obs(model_dir)

    def _spawn_obs(self, model_dir):
        obstacle_model = os.path.join(model_dir, 'unit_box', 'model.sdf')
        with open(obstacle_model, "r") as f:
            obstacle_sdf = f.read()
        box_pose = Pose()
        box_pose.position.x = 3
        box_pose.position.z = 1
        for i in range(self.obstacle_num):
            box_pose.position.y = i
            self.spawn_sdf_model("unit_box_%d" % i, obstacle_sdf, "unit_box_%d" % i, box_pose, 'world')

    def set_target(self):
        self.goal_pose.position.x = random.uniform(-3.6, 3.6)
        self.goal_pose.position.y = random.uniform(-3.6, 3.6)
        target_state = ModelState(model_name="target",
                                  pose=self.goal_pose)
        self.set_model_state(target_state)

    def set_obstacle(self):
        for i in range(self.obstacle_num):
            box_state = ModelState()
            box_state.model_name = "unit_box_%d" % i
            goal_x = self.goal_pose.position.x
            goal_y = self.goal_pose.position.y
            tb3_x = self.position.x
            tb3_y = self.position.y
            while True:
                box_x = random.uniform(-3.6, 3.6)
                box_y = random.uniform(-3.6, 3.6)
                if not goal_x + 0.5 > box_x > goal_x - 0.5 and \
                        not goal_y + 0.5 > box_y > goal_y - 0.5 and \
                        not tb3_x + 0.5 > box_x > tb3_x - 0.5 and \
                        not tb3_y + 0.5 > box_y > tb3_y - 0.5:
                    break
            box_state.pose.position.x = box_x
            box_state.pose.position.y = box_y
            box_state.pose.position.z = 1.0
            self.set_model_state(box_state)

    def common_reset(self):
        # Set the target
        self.set_target()
        if self.is_hard:
            self.set_obstacle()
        # First step we set current distance as past distance
        self.past_distance = self._get_distance()

    def reset(self):
        self.reset_world()
        # self.reset_simulation()
        return super(SimEnv, self).reset()
