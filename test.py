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
parser.add_argument('-t', '--template', help="template directory path")
args = parser.parse_args()


if __name__ == '__main__':

    from_date = dataformat.parse_date_string(args.from_date)
    to_date = dataformat.parse_date_string(args.to_date)
    if args.list is not None:
        set_list = args.list
    else:
        set_list = []

    # (depth, num_moments)
    if args.name is not None:
        weight_set_name = args.name
    else:
        weight_set_name = datetime.now().strftime("%Y-%m-%d")

    pwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weight_sets", weight_set_name)
    if not os.path.exists(pwd):
        print("cannot find a trained weight set.")
        exit()
    set_path = os.path.join(pwd, "set.json")
    with open(set_path, "r") as file:
        set_data = json.load(file)

    s = set_data["size"]
    input_size = (s[0], s[1])

    session_name = "weight_sets/" + set_data["session_name"]
    print(session_name, pwd)

    formatter = dataformat.DataFormat(input_size[0])

    data, labels = dataformat.read_data_directory(formatter, from_date, to_date, set_list)
    print("testset shapes: ", data.shape, labels.shape)

    if args.template is not None:
        template_path = args.template
    else:
        template_path = os.path.join("templates", "default")

    templates, template_labels = dataformat.read_template_directory(formatter, template_path)
    print("template shapes: ", templates.shape, template_labels.shape)

    num_intra_class = 10
    num_inter_class = 20
    comparator = momentnet.Comparator(input_size[0] * 2, input_size[1], num_intra_class=num_intra_class, num_inter_class=num_inter_class, layers=5, lambdas=(5, 0.5, 5))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    comparator.load_session(sess, session_name)
    classes, raw = comparator.process(sess, np.reshape(data, [-1, input_size[0] * 2]), np.reshape(templates, [-1, input_size[0] * 2]))
    print(classes)
    sess.close()

    print("Percent correct = ", np.count_nonzero(labels[:, 0] == template_labels[classes, 0]) * 100.0 / labels.shape[0])
