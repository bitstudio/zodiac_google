import argparse
from datetime import datetime
import os
import dataformat
import numpy as np
import json
import momentnet
import generator
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument("--name", help="weight set name")
parser.add_argument("--from_date", help="from date")
parser.add_argument("--to_date", help="to date")
parser.add_argument('-l', '--list', nargs='+', help='<Required> Set flag')
parser.add_argument("--iter", help="training iterations", type=int)
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


if __name__ == '__main__':

    from_date = dataformat.parse_date_string(args.from_date)
    to_date = dataformat.parse_date_string(args.to_date)
    if args.list is not None:
        set_list = args.list
    else:
        set_list = []

    # (depth, num_moments)
    num_layers = 5
    input_size = (256, 32)
    if args.name is not None:
        weight_set_name = args.name
    else:
        weight_set_name = datetime.now().strftime("%Y-%m-%d")
    num_classes = 16
    iterations = 5000
    if args.iter:
        iterations = args.iter

    pwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weight_sets", weight_set_name)
    if not os.path.exists(pwd):
        os.makedirs(pwd)

    session_name = weight_set_name + "/test"
    print(session_name, pwd)

    with open(pwd + "/set.json", "w") as file:
        json.dump({
            "size": [input_size[0], input_size[1]],
            "api_version": dataformat.api_version,
            "set_list": set_list,
            "from_date": args.from_date,
            "to_date": args.to_date,
            "session_name": session_name,
            "output_classes": num_classes,
            "num_layers": num_layers,
            "date_created": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, file, sort_keys=True, indent=4)

    formatter = dataformat.DataFormat(input_size[0])

    data, labels = dataformat.read_data_directory(formatter, from_date, to_date, set_list)
    data, labels = balance_labels(data, labels, num_classes)
    print(data.shape, labels.shape)

    num_intra_class = 10
    num_inter_class = 20
    comparator = momentnet.Comparator((2, input_size[0]), input_size[1], num_intra_class=num_intra_class, num_inter_class=num_inter_class, layers=num_layers, lambdas=(5, 0.5, 5))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    comparator.train(sess, data, session_name="weight_sets/" + session_name, batch_size=min(100, labels.shape[0]), max_iteration=iterations, continue_from_last=False)

    sess.close()
