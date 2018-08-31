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

    def build_tpu_graph(self, data, samples):
        a = self.body(tf.reshape(data, [-1, self.total_input_size]), self.num_moments, self.layers)
        z = self.body(tf.reshape(samples, [-1, self.total_input_size]), self.num_moments, self.layers)
        z = tf.reshape(z, [-1, self.num_intra_class + self.num_inter_class, self.num_moments])

        # compute cost
        intra_class_loss = self.moment_compare(tf.expand_dims(a, axis=1), z[:, 0:self.num_intra_class, :])
        inter_class_loss = self.moment_compare(tf.expand_dims(a, axis=1), z[:, self.num_intra_class:, :])

        overall_cost = tf.reduce_mean(intra_class_loss) - tf.reduce_mean(inter_class_loss) + 1.0
        return overall_cost

    def build_training_graph(self, data_size, batch_size, shuffle, sub_epoch=1):

        self.input_data = tf.placeholder(tf.float32, data_size)
        self.data_cache = tf.get_variable("data_cache", data_size, dtype=np.float32, trainable=False)
        self.samples = self.sample_generator.build_all(self.data_cache)

        data = tf.reshape(self.data_cache, [-1, self.total_input_size])
        samples = tf.reshape(self.samples, [-1, self.num_intra_class + self.num_inter_class, self.total_input_size])

        self.dataset = tf.data.Dataset.from_tensor_slices((data, samples)).batch(batch_size).repeat(sub_epoch)
        if shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=data_size[0] * data_size[1])

        self.data_iter = self.dataset.make_initializable_iterator()  # create the iterator
        self.next_element = self.data_iter.get_next()

        self.train_inputs = self.next_element[0]
        self.train_samples = self.next_element[1]

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
        self.training_op = tf.train.AdamOptimizer(0.00001).minimize(self.overall_cost, var_list=scope)

        self.upload_ops = tf.assign(self.data_cache, self.input_data)
        self.rebatch_ops = self.data_iter.initializer

    def train(self, data, session_name="weight_sets/test", session=None, shuffle=True, batch_size=5, max_iteration=100, continue_from_last=False):

        if session is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.log_device_placement = True
            sess = tf.Session(config=config)
        else:
            sess = session

        self.build_training_graph(data.shape, batch_size, shuffle, sub_epoch=10)

        sess.run(tf.global_variables_initializer())

        if continue_from_last:
            self.load_session(sess, session_name)

        sess.run(self.upload_ops, feed_dict={self.input_data: data})

        start_time = time.time()
        for step in range(max_iteration):
            sess.run(self.rebatch_ops)
            sum_loss = 0.0
            counter = 0
            while True:
                try:
                    _, loss = sess.run((self.training_op, self.overall_cost))
                    sum_loss += loss
                    counter += 1
                except tf.errors.OutOfRangeError:
                    break
            print(sum_loss / counter)
            if (step + 1) % 100 == 0:
                self.saver.save(sess, session_name)
                print("Checkpoint ...")
        elapsed_time = time.time() - start_time

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "log.txt"), "w") as file:
            file.write(str(device_lib.list_local_devices()))
            file.write("Total time ... " + str(elapsed_time) + " seconds")

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
