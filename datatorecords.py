import argparse
import os
import dataformat
import numpy as np
import generator as genclass
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument("--from_date", help="from date")
parser.add_argument("--to_date", help="to date")
parser.add_argument('-l', '--list', nargs='+', help='<Required> Set flag')
args = parser.parse_args()


def balance_labels(data, labels, num_classes, shuffle=False):

    all_list = []
    min_num_class = labels.shape[0]
    for i in range(num_classes):
        temp = np.argwhere(labels[:, 0] == i).flatten()
        if temp.shape[0] > 0:
            all_list.append(temp)
            if min_num_class > temp.shape[0]:
                min_num_class = temp.shape[0]

    selected_list = np.empty((len(all_list), min_num_class), dtype=np.int32)
    for i in range(len(all_list)):
        np.random.shuffle(all_list[i])
        selected_list[i, :] = all_list[i][0: min_num_class]

    indices = selected_list.flatten()
    if shuffle:
        np.random.shuffle(indices)
    out_shape = np.insert(data.shape, 0, len(all_list))
    out_shape[1] = -1
    return np.reshape(data[indices, ...], out_shape), np.reshape(labels[indices, ...], [out_shape[0], out_shape[1], 5])


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data, samples, filename):

    depth = data.shape[1]
    features = data.shape[2]
    sample_count = samples.shape[1]

    print('Writing', filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(data.shape[0]):
            raw = data[index].tostring()
            sample_raw = samples[index].tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'depth': _int64_feature(depth),
                        'features': _int64_feature(features),
                        'sample_count': _int64_feature(sample_count),
                        'data_raw': _bytes_feature(raw),
                        'sample_raw': _bytes_feature(sample_raw)
                    }))
            writer.write(example.SerializeToString())


if __name__ == '__main__':

    from_date = dataformat.parse_date_string(args.from_date)
    to_date = dataformat.parse_date_string(args.to_date)
    if args.list is not None:
        set_list = args.list
    else:
        set_list = []

    # (depth, num_moments)
    input_size = (256, 32)
    num_classes = 16

    pwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "records")
    if not os.path.exists(pwd):
        os.makedirs(pwd)
    filename = os.path.join(pwd, 'shadow.tfrecords')

    formatter = dataformat.DataFormat(input_size[0])

    data, labels = dataformat.read_data_directory(formatter, from_date, to_date, set_list)
    data, labels = balance_labels(data, labels, num_classes)
    print(data.shape, labels.shape)

    generator = genclass.Sample_generator((2, input_size[0]), 10, 20)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    samples = generator.generate(sess, data)
    sess.close()

    convert_to(data, samples, filename)
