#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Project Name:  vae
   File Name：     vae.py
   Description :
   date：          18-4-12
   Change Activity:
                   
-------------------------------------------------
"""
import tensorflow as tf
import os
import numpy as np


def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


def vae_layer(input_tensor, output_dim, name, param_dict, transfer_function=tf.nn.elu):
    """
    layer of vae (not the bottle-neck layer)
    :param input_tensor:
    :param output_dim:
    :param name:
    :param param_dict:
    :param transfer_function:
    :return:
    """
    input_dim = input_tensor.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        weight = tf.get_variable(scope+"weight",
                                 shape=[input_dim, output_dim],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[output_dim]), trainable=True)
        activation = transfer_function(tf.matmul(input_tensor, weight)+bias)
        param_dict[scope+"weight"] = weight
        param_dict[scope+"bias"] = bias

    return activation


def vae_sample_layer(input_tensor, output_dim, name, param_dict):
    """
    sample_layer of vae
    :param input_tensor:
    :param output_dim:
    :param name:
    :param param_dict:
    :return:
    """
    input_dim = input_tensor.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        weight_mean = tf.get_variable(scope+"hiddenLayer_mean",
                                      shape=[input_dim, output_dim],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias_mean = tf.Variable(tf.constant(0.0, shape=[output_dim]), trainable=True, dtype=tf.float32)
        gaussian_mean = tf.matmul(input_tensor, weight_mean) + bias_mean

        weight_var = tf.get_variable(scope + "hiddenLayer_var",
                                     shape=[input_dim, output_dim],
                                     dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias_var = tf.Variable(tf.constant(0.0, shape=[output_dim]), trainable=True, dtype=tf.float32)
        # add a small epsilon for numerical stability
        gaussian_var = tf.nn.softplus(tf.matmul(input_tensor, weight_var) + bias_var) + 1e-8

        epsilon = tf.random_normal(tf.shape(bias_var))
        z = tf.sqrt(tf.exp(gaussian_var)) * epsilon + gaussian_mean
        param_dict[scope + "weight"] = weight_mean
        param_dict[scope + "bias"] = bias_mean
        param_dict[scope + "weight"] = weight_var
        param_dict[scope + "bias"] = bias_var
    return gaussian_mean, gaussian_var, z


class Vae(object):
    def __init__(self, input_placeholder, model_path):
        """
        :param model_path
        :param input_placeholder:
        """
        self.input_dim = input_placeholder.get_shape()[-1].value
        print(self.input_dim)
        self.param_dict = dict()
        self.input = input_placeholder
        # encode part
        self.f1 = vae_layer(self.input, 512, 'enc_fc1', self.param_dict)
        self.f2 = vae_layer(self.f1, 384, 'enc_fc2', self.param_dict)
        self.f3 = vae_layer(self.f2, 256, 'enc_fc3', self.param_dict)
        self.f4 = vae_layer(self.f3, 128, 'enc_fc4', self.param_dict)
        self.f5 = vae_layer(self.f4, 64, 'enc_fc4', self.param_dict)
        self.mean, self.log_var, self.z = vae_sample_layer(self.f5, 25, 'sample_layer', self.param_dict)
        # decode part
        self.df0 = vae_layer(self.z, 64, 'dec_fc0', self.param_dict)
        self.df1 = vae_layer(self.df0, 128, 'dec_fc1', self.param_dict)
        self.df2 = vae_layer(self.df1, 256, 'dec_fc2', self.param_dict)
        self.df3 = vae_layer(self.df2, 384, 'dec_fc3', self.param_dict)
        self.df4 = vae_layer(self.df3, 512, 'dec_fc4', self.param_dict)
        self.reconstruct = vae_layer(self.df4, self.input_dim, 'dec_fc5', self.param_dict, tf.nn.sigmoid)
        # loss
        self.kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1+self.log_var-tf.square(self.mean)-tf.exp(self.log_var),
                                                           reduction_indices=1))
        self.margin_likelihood = tf.reduce_mean(-tf.reduce_sum(self.input * tf.log(self.reconstruct + 1e-8) +
                                                (1 - self.input) * tf.log(1e-8 + 1 - self.reconstruct), axis=1))

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.param_dict["Global_Step"] = self.global_step
        # initialization
        # load parameters
        self.sess = tf.Session()
        self.modelPath = model_path
        if not os.path.exists(model_path+'.index'):
            init = tf.global_variables_initializer()
            self.sess.run(init)
            print("Model not trained, should start training.")
        else:
            self.load(model_path)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            print("Loaded parameters from file.")

    def train_op(self, learning_rate=0.001):
        loss = tf.reduce_mean(self.margin_likelihood + self.kl_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=self.global_step)
        return loss, optimizer

    def load(self, model_path):
        print(model_path)
        saver = tf.train.Saver(self.param_dict)
        saver.restore(self.sess, model_path)

    def predict(self):
        return self.reconstruct

    def save(self, model_path):
        saver = tf.train.Saver(self.param_dict)
        path = saver.save(self.sess, model_path)
        print("Model saved in:" + path)

if __name___ == "__main__":
    print("lsdfa")