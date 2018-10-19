import tensorflow as tf
import numpy as np
import random
import time
from tensorflow.python.client import device_lib
import os
import json
import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import dataformat
from train import *


def broadcast(input, shape):
    return input + tf.zeros(shape, dtype=input.dtype)


class VAE:

    def residue_layer(self, i, input):
        input_size = input.get_shape()[1]
        w = tf.get_variable(str(i) + "w", [input_size, input_size], dtype=tf.float32, initializer=tf.random_normal_initializer(0.0, 0.001))
        u = tf.get_variable(str(i) + "u", [input_size, input_size], dtype=tf.float32, initializer=tf.random_normal_initializer(0.0, 0.001))
        ub = tf.get_variable(str(i) + "ub", input_size, dtype=tf.float32, initializer=tf.constant_initializer(1.0))
        wb = tf.get_variable(str(i) + "wb", input_size, dtype=tf.float32, initializer=tf.constant_initializer(1.0))

        residue = tf.nn.elu(tf.matmul(input, u) + ub) * input
        output = tf.nn.elu(tf.matmul(residue, w) + wb) + residue
        return output

    def body(self, input, size, layers):
        input_size = input.get_shape()[1]
        with tf.variable_scope("vae", reuse=tf.AUTO_REUSE):
            a = input
            for i in range(layers):
                a = self.residue_layer(i, a)

            v = tf.get_variable("v", [input_size, size], dtype=tf.float32, initializer=tf.random_normal_initializer(0.0, 0.001))
            vb = tf.get_variable("vb", size, dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            output = tf.nn.elu(tf.matmul(a, v) + vb)

            x = tf.get_variable("x", [input_size, size], dtype=tf.float32, initializer=tf.random_normal_initializer(0.0, 0.001))
            xb = tf.get_variable("xb", size, dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            sigma = tf.nn.elu(tf.matmul(a, x) + xb)

        return output, sigma

    def decode(self, input, size, layers):
        input_size = input.get_shape()[1]
        with tf.variable_scope("vae", reuse=tf.AUTO_REUSE):

            a = input
            v = tf.get_variable("va", [input_size, size], dtype=tf.float32, initializer=tf.random_normal_initializer(0.0, 0.001))
            vb = tf.get_variable("vab", size, dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            a = tf.nn.elu(tf.matmul(a, v) + vb)

            for i in range(layers):
                a = self.residue_layer(-i, a)

        return a

    def __init__(self, input_dimension, num_moments, layers=2):
        self.num_moments = num_moments
        self.total_input_size = input_dimension[0] * input_dimension[1]
        self.layers = layers

        self.inputs = tf.placeholder(tf.float32, [None, self.total_input_size])
        self.embeded, sigmas = self.body(self.inputs, self.num_moments, self.layers)

        z = self.embeded + sigmas * tf.random_normal(tf.shape(self.embeded), 0, 1, dtype=tf.float32)

        # decoding
        y = self.decode(z, self.total_input_size, self.layers)

        # loss
        marginal_likelihood = tf.reduce_mean(tf.exp(-tf.squared_difference(self.inputs, y)), 1)
        KL_divergence = 0.5 * tf.reduce_mean(tf.square(self.embeded) + tf.square(sigmas) - tf.log(1e-10 + tf.square(sigmas)) - 1, 1)

        marginal_likelihood = tf.reduce_mean(marginal_likelihood)
        KL_divergence = tf.reduce_mean(KL_divergence)

        ELBO = marginal_likelihood - KL_divergence

        self.overall_cost = 1.0 - ELBO

        scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        print([(x.name, x.dtype) for x in scope])

        self.saver = tf.train.Saver(var_list=scope, keep_checkpoint_every_n_hours=1)

    def build_training_graph(self, data_size, batch_size, shuffle):

        scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        print([x.name for x in scope])

        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.0001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.99, staircase=True)

        """ out of many algorithms, only Adam converge! A remarkable job for Kingma and Lei Ba!"""
        self.training_op = tf.train.AdamOptimizer(learning_rate).minimize(self.overall_cost, var_list=scope)

    def train(self, data, session_name="weight_sets/test", session=None, shuffle=True, batch_size=5, max_iteration=100, continue_from_last=False):

        # builder = tf.profiler.ProfileOptionBuilder
        # opts = builder(builder.time_and_memory()).order_by('micros').build()

        # pwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "profile")
        # with tf.contrib.tfprof.ProfileContext(pwd, trace_steps=[], dump_steps=[]) as pctx:

        if session is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.log_device_placement = True
            sess = tf.Session(config=config)
        else:
            sess = session

        batch_size = min(batch_size, data.shape[0] * data.shape[1])
        self.build_training_graph(data.shape, batch_size, shuffle)

        sess.run(tf.global_variables_initializer())

        if continue_from_last:
            self.load_session(sess, session_name)

        start_time = time.time()
        for step in range(max_iteration):
            sum_loss = 0.0
            total_batches = int(data.shape[0] * data.shape[1] / batch_size)
            flat = np.reshape(data, [data.shape[0] * data.shape[1], -1])
            for i in range(total_batches):
                # pctx.trace_next_step()
                batch = flat[i * batch_size:(i + 1) * batch_size, ...]
                _, loss = sess.run((self.training_op, self.overall_cost), feed_dict={self.inputs: batch})
                sum_loss += loss
            print(step, " : ", sum_loss / (total_batches))
            if (step + 1) % 100 == 0:
                self.saver.save(sess, session_name)
                print("Checkpoint ...")

                # pctx.dump_next_step()

        elapsed_time = time.time() - start_time

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts", "log.txt"), "w") as file:
            file.write(str(device_lib.list_local_devices()))
            file.write("Total time ... " + str(elapsed_time) + " seconds")

        # pctx.profiler.profile_operations(options=opts)

        self.saver.save(sess, session_name)

        if session is None:
            sess.close()

    def load_session(self, sess, session_name):
        print("loading from last save...")
        self.saver.restore(sess, session_name)

    def load_last(self, sess, directory):
        self.saver.restore(sess, tf.train.latest_checkpoint(directory))

    def embed(self, sess, data):
        embeded = sess.run((self.embeded), feed_dict={self.inputs: data})
        return embeded


if __name__ == "__main__":

    set_list = ["neo_boon", "neo_kim", "neo_tao", "neo_pear", "neo_o", "large9", "gdd_shadow"]

    # (depth, num_moments)
    num_layers = 5

    input_size = (256, 32)

    weight_set_name = "vae"

    num_classes = 16

    iterations = 2000

    pwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weight_sets", weight_set_name)
    if not os.path.exists(pwd):
        os.makedirs(pwd)

    session_name = weight_set_name + "/test"
    print(session_name, pwd, input_size, num_layers)

    with open(pwd + "/set.json", "w") as file:
        json.dump({
            "size": [input_size[0], input_size[1]],
            "api_version": dataformat.api_version,
            "set_list": set_list,
            "from_date": None,
            "to_date": None,
            "session_name": session_name,
            "output_classes": num_classes,
            "num_layers": num_layers,
            "date_created": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, file, sort_keys=True, indent=4)

    formatter = dataformat.DataFormat(input_size[0])

    data, labels = dataformat.read_data_directory(formatter, None, None, set_list)
    unflip_data(data, labels)
    data, labels = balance_labels(data, labels, num_classes)
    data = flip_data(data, as_diff_class=True)
    print(data.shape, labels.shape)

    net = VAE((2, input_size[0]), input_size[1], layers=num_layers)
    net.train(data, session_name="../weight_sets/" + session_name, batch_size=min(100, labels.shape[0] * 2), max_iteration=iterations, continue_from_last=False)
