import tensorflow as tf
import numpy as np


class Comparator:

    def residue_layer(self, i, input, size):
        input_size = input.get_shape()[1]
        w = tf.get_variable(str(i) + "w", [input_size, size], dtype=tf.float32, initializer=tf.random_normal_initializer(0.0, 0.001))
        u = tf.get_variable(str(i) + "u", [input_size, input_size], dtype=tf.float32, initializer=tf.random_normal_initializer(0.0, 0.001))
        v = tf.get_variable(str(i) + "v", [input_size, size], dtype=tf.float32, initializer=tf.random_normal_initializer(0.0, 0.001))
        b = tf.get_variable(str(i) + "b", size, dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        return tf.nn.elu(tf.matmul(tf.nn.elu(tf.matmul(input, u)), v) + tf.matmul(input, w) + b)

    def moment_compare(self, f0, f1):
        return tf.squared_difference(f0, f1)

    def __init__(self, input_dimension, num_moments, num_intra_class=10, num_inter_class=20, layers=2, lambdas=(5, 0.5, 5)):
        self.num_moments = num_moments
        self.input_dimension = input_dimension
        self.layers = layers
        self.num_intra_class = num_intra_class
        self.num_inter_class = num_inter_class

        self.inputs = tf.placeholder(tf.float32, [None, self.input_dimension])
        self.samples = tf.placeholder(tf.float32, [None, self.num_intra_class + self.num_inter_class, self.input_dimension])

        self.templates = tf.placeholder(tf.float32, [None, self.input_dimension])

        with tf.variable_scope("comparator") as scope:
            a = self.inputs
            for i in range(self.layers):
                a = self.residue_layer(i, a, int(self.input_dimension / 2))
            a = self.residue_layer(self.layers, a, self.num_moments)

        with tf.variable_scope(scope, reuse=True):
            z = tf.reshape(self.samples, [-1, self.input_dimension])
            for i in range(self.layers):
                z = self.residue_layer(i, z, int(self.input_dimension / 2))
            z = self.residue_layer(self.layers, z, self.num_moments)
            z = tf.stop_gradient(tf.reshape(z, [-1, self.num_intra_class + self.num_inter_class, self.num_moments]))

        # compute comparison graph

        with tf.variable_scope(scope, reuse=True):
            t = self.templates
            for i in range(self.layers):
                t = self.residue_layer(i, t, int(self.input_dimension / 2))
            t = self.residue_layer(self.layers, t, self.num_moments)

        self.raw_results = tf.reduce_sum(self.moment_compare(tf.expand_dims(a, axis=1), tf.expand_dims(t, axis=0)), axis=2)
        self.results = tf.argmin(self.raw_results, axis=1)

        # compute cost
        self.intra_class_loss = tf.reduce_mean(self.moment_compare(tf.expand_dims(a, axis=1), z[:, 0:self.num_intra_class, :]))
        self.inter_class_loss = tf.reduce_mean(self.moment_compare(tf.expand_dims(a, axis=1), z[:, self.num_intra_class:, :]))

        self.overall_cost = self.intra_class_loss - self.inter_class_loss

        scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        print([x.name for x in scope])
        """ out of many algorithms, only Adam converge! A remarkable job for Kingma and Lei Ba!"""
        self.training_op = tf.train.AdamOptimizer(0.0001).minimize(self.overall_cost, var_list=scope)

        self.saver = tf.train.Saver(var_list=scope, keep_checkpoint_every_n_hours=1)

    def train(self, sess, data, samples, session_name="../artifacts/test", shuffle=True, batch_size=2, max_iteration=1000, continue_from_last=False):
        if continue_from_last:
            self.load_session(sess, session_name)

        indices = np.arange(data.shape[0])
        if shuffle:
            np.random.shuffle(indices)

        for step in range(max_iteration):
            sum_loss = 0.0
            total_batches = data.shape[0] // batch_size
            for b in range(total_batches):
                db = data[indices[(b * batch_size):((b + 1) * batch_size)], ...]
                sb = samples[indices[(b * batch_size):((b + 1) * batch_size)], ...]
                _, loss = sess.run((self.training_op, self.overall_cost), feed_dict={self.inputs: db, self.samples: sb})
                sum_loss += loss
            print(sum_loss / total_batches)
            if (step + 1) % 100 == 0:
                self.saver.save(sess, session_name)
                print("Checkpoint ...")

        self.saver.save(sess, session_name)

    def load_session(self, sess, session_name):
        print("loading from last save...")
        self.saver.restore(sess, session_name)

    def load_last(self, sess, directory):
        self.saver.restore(sess, tf.train.latest_checkpoint(directory))

    def process(self, sess, data, templates):
        results, raw_results = sess.run((self.results, self.raw_results), feed_dict={self.inputs: data, self.templates: templates})
        return results


if __name__ == "__main__":

    sess = tf.Session()

    data = np.random.rand(10, 20)
    samples = np.concatenate((np.tile(data, [10, 1, 1]), np.random.rand(10, 20, 20)), axis=1)
    templates = data + np.random.rand(10, 20) * 0.1

    net = Comparator(20, 5, num_intra_class=10, num_inter_class=20, layers=5)
    # print(labels)

    sess.run(tf.global_variables_initializer())

    net.train(sess, data, samples)
    classes = net.process(sess, data, templates)
    print(classes)

    sess.close()
