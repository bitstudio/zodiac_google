from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import momentnet

import tensorflow as tf  # pylint: disable=g-bad-import-order

# For open source environment, add grandparent directory for import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.path[0]))))


def metric_fn(labels, predictions):
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predictions)
    return {"accuracy": accuracy}


input_size = (256, 32)
num_intra_class = 10
num_inter_class = 20
use_tpu = False


def model_fn(features, labels, mode, params):
    """model_fn constructs the ML model used to predict handwritten digits."""

    del params
    if mode == tf.estimator.ModeKeys.PREDICT:
        raise RuntimeError("mode {} is not supported yet".format(mode))

    data = features
    samples = labels
    model = momentnet.Comparator((2, input_size[0]), input_size[1], num_intra_class=num_intra_class, num_inter_class=num_inter_class, layers=5)
    loss = model.build_tpu_graph(data, samples)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
        if use_tpu:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_global_step()))

    if mode == tf.estimator.ModeKeys.EVAL:
        raise RuntimeError("mode {} is not supported yet".format(mode))


def _parse_function(datum):
    features = {
        'depth': tf.FixedLenFeature((), tf.int32, default_value=0),
        'features': tf.FixedLenFeature((), tf.int32, default_value=0),
        'sample_count': tf.FixedLenFeature((), tf.int32, default_value=0),
        'data_raw': tf.FixedLenFeature([2, input_size[1]], tf.float32, default_value=""),
        'sample_raw': tf.FixedLenFeature([num_intra_class + num_inter_class, 2, input_size[1]], tf.float32, default_value="")
    }
    parsed_features = tf.parse_single_example(datum, features)
    return parsed_features["data_raw"], parsed_features["sample_raw"]


def shadow_input_fn(params):
    """train_input_fn defines the input pipeline used for training."""
    batch_size = params["batch_size"]
    filename = params["filename"]

    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(_parse_function)  # Parse the record into tensors.
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    iterator = dataset.make_initializable_iterator()
    data, samples = iterator.get_next()

    return data, samples


def main(argv):
    del argv  # Unused.
    tf.logging.set_verbosity(tf.logging.INFO)

    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        "tpu-for-shadow"
    )

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir="gs://bucket-for-tpu-weights/weights",
        session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
        tpu_config=tf.contrib.tpu.TPUConfig(100, 8),
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        use_tpu=use_tpu,
        train_batch_size=100,
        eval_batch_size=100,
        params={"filename": "gs://bucket-for-tpu/shadow/data/shadow.tfrecords"},
        config=run_config)

    # TPUEstimator.train *requires* a max_steps argument.
    estimator.train(input_fn=shadow_input_fn, max_steps=100)


if __name__ == "__main__":
    tf.app.run()
