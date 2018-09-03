import tensorflow as tf
import numpy as np
import random
import time
from tensorflow.python.client import device_lib
import os
import generator


def broadcast(input, shape):
    return input + tf.zeros(shape, dtype=input.dtype)


class Comparator:

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
        with tf.variable_scope("moment", reuse=not self.first_time):
            a = input
            for i in range(layers):
                a = self.residue_layer(i, a)

            v = tf.get_variable("v", [input_size, size], dtype=tf.float32, initializer=tf.random_normal_initializer(0.0, 0.001))
            b = tf.get_variable("b", size, dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            output = tf.nn.elu(tf.matmul(a, v) + b)

        self.first_time = False
        return output

    def moment_compare(self, f0, f1):
        return - tf.exp(-(tf.reduce_sum(tf.squared_difference(f0, f1), axis=2)))

    def __init__(self, input_dimension, num_moments, num_intra_class=10, num_inter_class=20, layers=2, lambdas=(5, 0.5, 5)):
        self.num_moments = num_moments
        self.total_input_size = input_dimension[0] * input_dimension[1]
        self.layers = layers
        self.num_intra_class = num_intra_class
        self.num_inter_class = num_inter_class

        self.sample_generator = generator.Sample_generator(input_dimension, self.num_intra_class, self.num_inter_class)

        self.inputs = tf.placeholder(tf.float32, [None, self.total_input_size])
        self.templates = tf.placeholder(tf.float32, [None, self.total_input_size])

        self.first_time = True

        a = self.body(self.inputs, self.num_moments, self.layers)
        t = self.body(self.templates, self.num_moments, self.layers)

        # compute comparison graph
        self.raw_results = self.moment_compare(tf.expand_dims(a, axis=1), tf.expand_dims(t, axis=0))
        self.results = tf.argmin(self.raw_results, axis=1)
        self.raw_confidence = - self.raw_results

        scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        print([(x.name, x.dtype) for x in scope])

        self.saver = tf.train.Saver(var_list=scope, keep_checkpoint_every_n_hours=1)

    def build_training_graph(self, data_size, batch_size, shuffle):

        self.input_data = tf.placeholder(tf.float32, data_size)
        self.data_cache = tf.get_variable("data_cache", data_size, dtype=np.float32, trainable=False)
        self.samples = self.sample_generator.build_all(self.data_cache)

        data = tf.reshape(self.data_cache, [-1, self.total_input_size])
        samples = tf.reshape(self.samples, [-1, self.num_intra_class + self.num_inter_class, self.total_input_size])

        total_data = data_size[0] * data_size[1]
        total_batches = int(total_data / batch_size)

        self.batch_index = tf.get_variable("batch_index", (), dtype=np.int32, trainable=False)
        self.dataset = tf.get_variable("dataset", [total_batches * batch_size, self.total_input_size], dtype=np.float32, trainable=False)
        self.sampleset = tf.get_variable("sampleset", [total_batches * batch_size, self.num_intra_class + self.num_inter_class, self.total_input_size], dtype=np.float32, trainable=False)

        print(total_data, " vs ", total_batches * batch_size, " of ", batch_size)

        indices = tf.mod(tf.range(total_data), (total_batches * batch_size))
        if shuffle:
            indices = tf.random_shuffle(indices)

        self.dataset_init = tf.scatter_update(self.dataset, indices, data)
        self.sampleset_init = tf.scatter_update(self.sampleset, indices, samples)
        self.iter_init = tf.assign(self.batch_index, 0)
        self.iter_next = tf.assign(self.batch_index, tf.mod(self.batch_index + 1, total_batches))

        self.train_inputs = tf.reshape(self.dataset, [-1, batch_size, self.total_input_size])[self.batch_index]
        self.train_samples = tf.reshape(self.sampleset, [-1, batch_size, self.num_intra_class + self.num_inter_class, self.total_input_size])[self.batch_index]

        a = self.body(self.train_inputs, self.num_moments, self.layers)
        z = self.body(tf.reshape(self.train_samples, [-1, self.total_input_size]), self.num_moments, self.layers)
        z = tf.reshape(z, [-1, self.num_intra_class + self.num_inter_class, self.num_moments])

        # compute cost
        self.intra_class_loss = self.moment_compare(tf.expand_dims(a, axis=1), z[:, 0:self.num_intra_class, :])
        self.inter_class_loss = self.moment_compare(tf.expand_dims(a, axis=1), z[:, self.num_intra_class:, :])

        self.overall_cost = tf.reduce_mean(self.intra_class_loss) - tf.reduce_mean(self.inter_class_loss) + 1.0

        scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        print([x.name for x in scope])

        """ out of many algorithms, only Adam converge! A remarkable job for Kingma and Lei Ba!"""
        self.training_op = (tf.train.AdamOptimizer(0.00001).minimize(self.overall_cost, var_list=scope), self.iter_next)

        self.upload_ops = tf.assign(self.data_cache, self.input_data)
        self.rebatch_ops = (self.iter_init, self.dataset_init, self.sampleset_init)

    def train(self, data, session_name="weight_sets/test", session=None, shuffle=True, batch_size=5, max_iteration=100, continue_from_last=False):

        # builder = tf.profiler.ProfileOptionBuilder
        # opts = builder(builder.time_and_memory()).order_by('micros').build()

        # pwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "profile")
        # with tf.contrib.tfprof.ProfileContext(pwd, trace_steps=[], dump_steps=[]) as pctx:

        if session is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            # config.log_device_placement = True
            sess = tf.Session(config=config)
        else:
            sess = session

        batch_size = min(batch_size, data.shape[0] * data.shape[1])
        self.build_training_graph(data.shape, batch_size, shuffle)

        sess.run(tf.global_variables_initializer())

        if continue_from_last:
            self.load_session(sess, session_name)

        sess.run(self.upload_ops, feed_dict={self.input_data: data})

        sub_epoch = 10
        start_time = time.time()
        for step in range(max_iteration):
            sess.run(self.rebatch_ops)
            sum_loss = 0.0
            total_batches = int(data.shape[0] * data.shape[1] / batch_size)
            for i in range(total_batches * sub_epoch):
                # pctx.trace_next_step()
                _, loss = sess.run((self.training_op, self.overall_cost))
                sum_loss += loss
            print(sum_loss / (total_batches * sub_epoch))
            if (step + 1) % 100 == 0:
                self.saver.save(sess, session_name)
                print("Checkpoint ...")

                # pctx.dump_next_step()

        elapsed_time = time.time() - start_time

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "log.txt"), "w") as file:
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

    def process(self, sess, data, templates):
        results, raw_confs = sess.run((self.results, self.raw_confidence), feed_dict={self.inputs: data, self.templates: templates})
        return results, raw_confs


def sample_shift_shuffle(x, count):
    temp = []
    for i in range(count):
        index = random.randrange(0, x.shape[0])
        temp.append(np.roll(x[index, :], random.randint(0, x.shape[1])))

    return np.stack(temp, axis=0)


if __name__ == "__main__":
    data = np.asarray([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]])
    print(sample_shift_shuffle(data, 4))

    data = np.random.rand(5, 1, 2, 20) * 2 - 0.5
    templates = np.roll(data, 5, axis=3)

    net = Comparator((2, 20), 5, num_intra_class=20, num_inter_class=20, layers=5)

    sess = tf.Session()
    net.train(data, session=sess)

    classes, raw = net.process(sess, np.reshape(data, [-1, 2 * 20]), np.reshape(templates, [-1, 2 * 20]))
    print(classes)
    print(raw)

    sess.close()
