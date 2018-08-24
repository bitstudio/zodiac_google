import tensorflow as tf
import numpy as np


class Sample_generator:

    def __init__(self, input_size, num_intra_class=10, num_inter_class=20):

        self.input_size = input_size
        self.num_intra_class = int(num_intra_class)
        self.num_inter_class = int(num_inter_class)

        self.inputs = tf.placeholder(tf.float32, [None, None, self.input_size[0], self.input_size[1]])

        sampled = self.build_sample_graph(self.inputs)

        shifted = self.build_shift_graph(tf.reshape(sampled, [-1, self.input_size[0], self.input_size[1]]))
        self.samples = tf.reshape(shifted, [-1, self.num_intra_class + self.num_inter_class, self.input_size[0], self.input_size[1]])

    def build_sample_graph(self, input):

        def body_intra(X):
            selected_intra_indices = tf.random_uniform([tf.shape(input)[1] * self.num_intra_class], 0, tf.shape(X)[0], dtype=tf.int32)
            return tf.gather(X, selected_intra_indices)

        intra_sample = tf.map_fn(body_intra, input)

        flat = tf.reshape(input, [-1, self.input_size[0], self.input_size[1]])

        def body_inter(index):
            all_indices = tf.concat([
                tf.range(index * tf.shape(input)[1]),
                tf.range((index + 1) * tf.shape(input)[1], tf.shape(input)[0] * tf.shape(input)[1])], axis=0)
            selected_indices = tf.random_uniform([tf.shape(input)[1] * self.num_inter_class], 0, tf.shape(all_indices)[0], dtype=tf.int32)
            return tf.gather(flat, tf.gather(all_indices, selected_indices))

        inter_sample = tf.map_fn(body_inter, tf.range(tf.shape(input)[0]), dtype=tf.float32)

        return tf.concat([
            tf.reshape(intra_sample, [-1, self.num_intra_class, self.input_size[0], self.input_size[1]]),
            tf.reshape(inter_sample, [-1, self.num_inter_class, self.input_size[0], self.input_size[1]])], axis=1)

    def build_shift_graph(self, input):

        def body(X):
            return tf.manip.roll(X, shift=tf.random_uniform((), 0, self.input_size[1], dtype=tf.int32), axis=1)

        out = tf.map_fn(body, input)
        return out

    def generate(self, sess, seed):
        return sess.run(self.samples, feed_dict={self.inputs: seed})


if __name__ == '__main__':
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    data = np.tile(np.reshape(np.arange(4), [-1, 1, 1, 1]), [1, 3, 2, 5])
    print(data)
    generator = Sample_generator((2, 5), 1, 2)
    gened = generator.generate(sess, data)
    print(gened)

    shift = tf.random_uniform((), 0, data.shape[3], dtype=tf.int32)
    print(sess.run(shift))

    sess.close()
