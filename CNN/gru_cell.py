#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

import tensorflow as tf
import numpy as np

logger = logging.getLogger('final.gru')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class GRUCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, input_size, state_size):
        self.input_size = input_size
        self._state_size = state_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state):
        """Updates the state using the previous @state and @inputs.
        Args:
            inputs: is the input vector of size [None, self.input_size]
            state: is the previous state vector of size [None, self.state_size]
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            a pair of the output vector and the new state vector.
        """
        with tf.variable_scope('GRU'):
            W_r = tf.get_variable('W_r', shape=[self.state_size, self.state_size], initializer=tf.contrib.layers.xavier_initializer())
            U_r = tf.get_variable('U_r', shape=[self.input_size, self.state_size], initializer=tf.contrib.layers.xavier_initializer())
            b_r = tf.get_variable('b_r', shape=[self.state_size,], initializer = tf.constant_initializer(0.0))


            W_z = tf.get_variable('W_z', shape=[self.state_size, self.state_size], initializer=tf.contrib.layers.xavier_initializer())
            U_z = tf.get_variable('U_z', shape=[self.input_size, self.state_size], initializer=tf.contrib.layers.xavier_initializer())
            b_z = tf.get_variable('b_z', shape=[self.state_size,], initializer = tf.constant_initializer(0.0))

            W_o = tf.get_variable('W_o', shape=[self.state_size, self.state_size], initializer=tf.contrib.layers.xavier_initializer())
            U_o = tf.get_variable('U_o', shape=[self.input_size, self.state_size], initializer=tf.contrib.layers.xavier_initializer())
            b_o = tf.get_variable('b_o', shape=[self.state_size,], initializer = tf.constant_initializer(0.0))

            z_t = tf.nn.sigmoid(tf.matmul(inputs, U_z) + tf.matmul(state, W_z) + b_z)
            r_t = tf.nn.sigmoid(tf.matmul(inputs, U_r) + tf.matmul(state, W_r) + b_r)
            o_t = tf.nn.tanh(tf.matmul(inputs, U_o) + tf.matmul(r_t * state, W_o)+ b_o)
            new_state = z_t * state + (1 - z_t) * o_t

        output = new_state
        return output, new_state