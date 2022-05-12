import pickle
from abc import ABC

import numpy as np
from keras import layers
from models import PolarityOutput
from modules.models import Model

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from modules.ner.models import TokenAndPositionEmbedding, TransformerBlock


class TransformersModel:
    def __init__(
            self, num_tags, vocab_size, maxlen=128, embed_dim=32, num_heads=2, ff_dim=32
    ):
        super(TransformersModel, self).__init__()
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.dropout1 = layers.Dropout(0.1)
        self.ff = layers.Dense(ff_dim, activation="relu")
        self.dropout2 = layers.Dropout(0.1)
        self.ff_final = layers.Dense(num_tags, activation="softmax")

    def call(self, inputs, training=False):
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x)
        x = self.dropout1(x, training=training)
        x = self.ff(x)
        x = self.dropout2(x, training=training)
        x = self.ff_final(x)
        return x

    def train(self, inputs, outputs):
        """

        :param inputs:
        :param outputs:
        """
        X = self._represent(inputs, aspectId)
        ys = [output.scores for output in outputs]

        self.models[aspectId].fit(X, ys)

    def save(self, path, aspectId):
        # save the model to disk
        pickle.dump(self.models[aspectId], open(path, 'wb'))

    def load(self, path, aspectId):
        # load the model from disk
        model = pickle.load(open(path, 'rb'))
        self.models[aspectId] = model

    def predict(self, inputs, aspectId):
        """
        :param inputs:
        :return:
        :rtype: list of models.AspectOutput
        """
        X = self._represent(inputs, aspectId)
        outputs = []
        predicts = self.models[aspectId].predict(X)
        for output in predicts:
            label = 'aspect{}'.format(aspectId) + (' -' if output == -1 else ' +')
            aspect = 'aspect{}'.format(aspectId)
            outputs.append(PolarityOutput(label, aspect, output))
        return outputs
