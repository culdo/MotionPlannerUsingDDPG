#!/usr/bin/env python3
import os

import numpy as np
import tensorflow as tf

from core.actor_network_bn import ActorNetwork
from core.critic_network import CriticNetwork
from core.replay_buffer import ReplayBuffer

REPLAY_BUFFER_SIZE = 100000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 128
GAMMA = 0.99

model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'model', 'with_obstacle')

class DDPG:
    def __init__(self, env, state_dim, action_dim):
        self.name = 'DDPG'
        self.environment = env
        self.time_step = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.actor_network = ActorNetwork(self.sess, self.state_dim, self.action_dim)
        self.critic_network = CriticNetwork(self.sess, self.state_dim, self.action_dim)
        self._load_network()

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    def _load_network(self):
        self.saver = tf.train.Saver(save_relative_paths=True, max_to_keep=1)
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded network")
        else:
            print("Could not find old network weights")

    def _save_network(self, time_step):
        print('save DDPG network...', time_step)
        path = os.path.join(model_dir, "DDPG")
        self.saver.save(self.sess, path, global_step=time_step)

    def train(self):
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])
        # for action_dim = 1
        action_batch = np.resize(action_batch, [BATCH_SIZE, self.action_dim])

        next_action_batch = self.actor_network.target_actions(next_state_batch)
        q_value_batch = self.critic_network.target_q(next_state_batch, next_action_batch)
        y_batch = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch, [BATCH_SIZE, 1])
        # Update critic by minimizing the loss L
        self.critic_network.train(y_batch, state_batch, action_batch)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch, action_batch_for_gradients)

        self.actor_network.train(q_gradient_batch, state_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()

    def action(self, state):
        action = self.actor_network.action(state)

        return action

    def perceive(self, state, action, reward, next_state, done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        if self.replay_buffer.count() == REPLAY_START_SIZE:
            print('\n---------------Start training---------------')
        # Store transitions to replay start size then start training
        if self.replay_buffer.count() > REPLAY_START_SIZE:
            self.time_step += 1
            self.train()
        elif self.replay_buffer.count() % 500 == 0:
            print("REPLAY_SIZE: %d" % self.replay_buffer.count())

        if self.time_step % 10000 == 0 and self.time_step > 0:
            self._save_network(self.time_step)

        return self.time_step
