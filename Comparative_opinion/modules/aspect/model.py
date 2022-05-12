import datetime
import pickle
from abc import ABC
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from models import PolarityOutput, Input, ComparativeStcOutput
from modules.embeddings import TfidfModel
from modules.models import Model

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


class AspectModel(Model, ABC):
    def __init__(self, vocab_path, model):
        self.vocab_path = vocab_path
        self.vocab = []
        self.model = model
        self.threshold = 0

    def chi2_dict(self, k_best):
        vocab_df = pd.read_csv(self.vocab_path).head(k_best)
        self.vocab = list(vocab_df.word)

    def chi2_represent(self, inputs):
        """
        :param list of models.Input inputs:
        :return:
        """

        features = []
        for input in inputs:
            _feature = [1 if word in input.stc.split(' ') else 0 for word in self.vocab]
            features.append(_feature)

        return features

    def train(self, encode_inputs: List, outputs: List):
        """
        :param encode_inputs:
        :param outputs:
        """
        X = encode_inputs
        y = outputs
        self.model.fit(X, y)

    def save(self):
        # save the model to disk
        pickle.dump(self.model,
                    open('./data/output/model/comparative_stc/comparative_stc_model{}.pkl'.format(
                        datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), 'wb'))

    def load(self, path):
        # load the model from disk
        model = pickle.load(open(path, 'rb'))

        self.model = model

    def predict(self, encode_inputs: List):
        X = encode_inputs
        if self.threshold != 0:
            predict_prob = self.model.predict_proba(X)[:, 1]
            predict = np.where(predict_prob > self.threshold, 1, 0)
        else:
            predict = self.model.predict(X)
        return predict

    def evaluate(self, y_test, y_predicts):
        p = precision_score(y_test, y_predicts)
        r = recall_score(y_test, y_predicts)
        f1 = f1_score(y_test, y_predicts)
        return p, r, f1

    def tune_threshold(self, encode_inputs: List, y_test, save=True):
        print("* Tuning CSI model ...")
        X = encode_inputs
        y_predict_proba = self.model.predict_proba(X)[:, 1]
        thresholds = np.arange(0, 1, 0.001)
        max_f1 = 0
        opt_t = 0
        for t in thresholds:
            predict_t = np.where(y_predict_proba > t, 1, 0)
            f1 = f1_score(y_test, predict_t)
            if f1 > max_f1:
                max_f1 = f1
                opt_t = t
        predict = np.where(y_predict_proba > opt_t, 1, 0)
        p = precision_score(y_test, predict)
        r = recall_score(y_test, predict)
        self.threshold = opt_t
        print("  Tuning CSI model DONE!")

        return opt_t
