import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import *

from file_utils import DATA_PATH

ing2index = pickle.load(open(os.path.join(DATA_PATH, 'ing2idx.pickle'), 'rb'))
ing2index['Unknown'] = 0
index2ing = {}
for ing in ing2index:
    idx = ing2index[ing]
    index2ing[idx] = ing


def calculate_metrics(pred, target, threshold=0.5):
    """ derived from this tutorial https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
    pred = batch_size * class (scores)
    target = batchsize * class (binary)
    return class specific precision, recall f1 in a dataframe"""
    # auc = roc_auc_score(y_true = target, y_score = pred, average = None, multi_class = 'ovr')
    y_pred = np.array(pred > threshold, dtype=float)
    prec = precision_score(y_true=target, y_pred=y_pred, average=None, zero_division=0)
    recall = recall_score(y_true=target, y_pred=y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true=target, y_pred=y_pred, average=None, zero_division=0)

    df = pd.DataFrame([prec, recall, f1], index=['prec', 'recall', 'f1'])
    df.columns = [index2ing[c] for c in df.columns]
    return df.T
