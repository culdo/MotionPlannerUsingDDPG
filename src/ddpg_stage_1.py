#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Twist

from core.ddpg import DDPG
from environment import Env
from sim_env import SimEnv


class DDPGStage:
    def __init__(self, is_training=False):
        self.exploration_decay_start_step = 50000
        state_dim = 366
        action_dim = 2
        self.action_linear_max = 0.25  # m/s
        self.action_angular_max = 0.5  # rad/s
        self.is_training = is_training

        rospy.init_node('ddpg_stage_1')
        if ['/gazebo/model_states', 'gazebo_msgs/ModelStates'] in rospy.get_published_topics():
            self.env = SimEnv(self.is_training)
            print("Gazebo mode")
        else:
            self.env = Env(self.is_training)
            print("Real world mode")

        self.agent = DDPG(self.env, state_dim, action_dim)
        print('State Dimensions: ' + str(state_dim))
        print('Action Dimensions: ' + str(action_dim))
        print('Action Max: ' + str(self.action_linear_max) + ' m/s and ' + str(self.action_angular_max) + ' rad/s')

    def _train(self):
        print('Training mode')
        avg_reward_his = []
        total_reward = 0
        var = 1.

        while True:
            state = self.env.reset()
            one_round_step = 0

            while True:
                a = self.agent.action(state)
                print("action: %s" % a)
                a[0] = np.clip(np.random.normal(a[0], var), 0., 1.)
                a[1] = np.clip(np.random.normal(a[1], var), -0.5, 0.5)

                state_, r, collision, arrive = self.env.step(a)
                time_step = self.agent.perceive(state, a, r, state_, collision)

                if time_step > 0:
                    total_reward += r

                if time_step % 10000 == 0 and time_step > 0:
                    print('---------------------------------------------------')
                    avg_reward = total_reward / 10000
                    print('Average_reward = ', avg_reward)
                    avg_reward_his.append(round(avg_reward, 2))
                    print('Average Reward:', avg_reward_his)
                    total_reward = 0

                if time_step % 5 == 0 and time_step > self.exploration_decay_start_step and var > 0.2:
                    var *= 0.9999

                state = state_
                one_round_step += 1

                result = 'Step: %3i | Var: %.2f | Time step: %i |' % (one_round_step, var, time_step)
                if arrive:
                    print(result, 'Success')
                    one_round_step = 0
                    self.env.common_reset()
                elif collision:
                    print(result, 'Collision')
                    break
                elif one_round_step >= 500:
                    print(result, 'Failed')
                    break

    def _evaluate(self):
        print('Testing mode')
        while True:
            state = self.env.reset()
            one_round_step = 0

            while True:
                a = self.agent.action(state)
                print("action: %s" % a)
                a[0] = np.clip(a[0], 0., 1.)
                a[1] = np.clip(a[1], -0.5, 0.5)
                state_, r, collision, arrive = self.env.step(a)
                state = state_
                one_round_step += 1

                result = 'Step: %3i |' % one_round_step
                if arrive:
                    print(result, 'Success')
                    one_round_step = 0
                    self.env.common_reset()
                elif collision:
                    print(result, 'Collision')
                    break
                elif one_round_step >= 500:
                    print(result, 'Failed')
                    break

    def run(self):
        try:
            if self.is_training:
                self._train()
            else:
                self._evaluate()
        except KeyboardInterrupt:
            self.env.pub_cmd_vel.publish(Twist())


if __name__ == '__main__':
    process = DDPGStage(is_training=True)
    process.run()
