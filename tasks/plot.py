from time import time

import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import dataformat
import momentnet
import tensorflow as tf


template_dir = "neo_large"
formatter = dataformat.DataFormat(256)
templates, template_labels, raws = dataformat.read_template_directory(formatter, os.path.join("templates", template_dir), with_flip=True, return_raw=True)
print(templates.shape)
X = np.reshape(templates, [templates.shape[0], -1])

perform_embedding = True
if perform_embedding:
    comparator = momentnet.Comparator((2, 256), 32, num_intra_class=10, num_inter_class=20, layers=5, lambdas=(5, 0.5, 5))

    session_name = os.path.join(os.path.dirname(__file__), '..', "weight_sets", "neo_large_5L_flip", "test")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    comparator.load_session(sess, session_name)
    X = comparator.embed(sess, X)
    sess.close()

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors


def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    # for i in range(X.shape[0]):
    #     plt.text(X[i, 0], X[i, 1], str(y[i]),
    #              color=plt.cm.Set1(y[i] / 10.),
    #              fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big

        indices = list(range(X.shape[0]))
        random.shuffle(indices)
        for i in indices:
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 1e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(cv2.resize(raws[i], (32, 32)), cmap=plt.cm.gray_r), X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


if __name__ == '__main__':
    tsne = manifold.TSNE(n_components=2, init='pca')

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

        cid = plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)
        plt.show()
