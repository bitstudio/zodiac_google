from time import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import (manifold)

import os
import sys
import tensorflow as tf
import vae
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import dataformat
import momentnet


template_dir = "large_template_front"
weight_sets = "large_embeded_5stepRot"
perform_embedding = True

colors = np.asarray([
    [120, 120, 120],
    [0, 0, 0],
    [0, 0, 255],
    [255, 0, 0],
    [0, 255, 0],
    [120, 0, 0],
    [0, 120, 0],
    [0, 0, 120],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
    [120, 120, 0],
    [0, 120, 120],
    [120, 0, 120],
    [255, 120, 255]
], dtype=np.float)

formatter = dataformat.DataFormat(256)
templates, template_labels, raws = dataformat.read_template_directory(formatter, os.path.join("templates", template_dir), with_flip=True, return_raw=True)
print(templates.shape)
X = np.reshape(templates, [templates.shape[0], -1])
Y = colors[template_labels[:, 0]] / 255.0
X_plotted = None

if perform_embedding:
    comparator = momentnet.Comparator((2, 256), 32, num_intra_class=10, num_inter_class=20, layers=5)
    # comparator = vae.VAE((2, 256), 32, layers=5)

    session_name = os.path.join(os.path.dirname(__file__), '..', "weight_sets", weight_sets, "test")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    comparator.load_session(sess, session_name)
    X = comparator.embed(sess, X)
    sess.close()


#----------------------------------------------------------------------
# Scale and visualize the embedding vectors


def plot_embedding(X, title=None):
    global X_plotted
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X_plotted = (X - x_min) / (x_max - x_min)

    plt.figure()
    plt.subplot(111)
    plt.scatter(X_plotted[:, 0], X_plotted[:, 1], s=4, c=Y)

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def hover(event):
    if X_plotted is None or event.xdata is None or event.ydata is None:
        print(event.xdata, event.ydata)
    else:
        d_ = X_plotted - np.asarray([event.xdata, event.ydata], dtype=np.float)
        d_ = d_ * d_
        d_ = d_[:, 0] + d_[:, 1]
        min_ = np.argmin(d_, axis=0)
        cv2.imshow("view", raws[min_])
        cv2.waitKey(1)


if __name__ == '__main__':
    tsne = manifold.TSNE(n_components=2, init='random')

    key = None

    def quit_figure(event):
        global key
        if event.key == 'q':
            key = 'q'
        plt.close(event.canvas.figure)

    while key != 'q':
        t0 = time()
        X_tsne = tsne.fit_transform(X)
        plot_embedding(X_tsne,
                       "t-SNE embedding (time %.2fs)" %
                       (time() - t0))

        plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)
        plt.gcf().canvas.mpl_connect("motion_notify_event", hover)
        plt.show()
