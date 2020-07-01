from typing import *
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE


def get_uniques_from_nested_lists(nested_lists: List[List]) -> List:
    uniques = {}
    for one_line in nested_lists:
        for item in one_line:
            if not uniques.get(item):
                uniques[item] = 1
    return list(uniques.keys())


def get_item2idx(items, unique=False) -> Tuple[Dict, Dict]:
    item2idx, idx2item = dict(), dict()
    items_unique = items if unique else set(items)
    for idx, item in enumerate(items_unique):
        item2idx[item] = idx
        idx2item[idx] = item
    return item2idx, idx2item


def tsne_plot(labels, vectors, filename, perplexity=10, figsize=(8, 8), cmap='nipy_spectral', dpi=300):
    tsne_model = TSNE(perplexity=perplexity, n_components=2,
                      metric='cosine',
                      init='pca', n_iter=5000, random_state=22)
    new_values = tsne_model.fit_transform(vectors)

    x, y = [], []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.rcParams["font.family"] = 'D2Coding'
    plt.clf()
    plt.figure(figsize=figsize)
    plt.title(filename)
    plt.scatter(x, y, cmap=cmap, alpha=0.5)

    for i in range(len(x)):
        #         plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    timestamp = dt.today().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("results/{}_perp{}.png".format(filename, perplexity), dpi=dpi)
