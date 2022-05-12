import numpy as np
import pandas as pd

from modules.ner.crf import SentenceGetter, CRFModel
from seqeval.metrics import classification_report
import sklearn_crfsuite
if __name__ == '__main__':
    train = pd.read_csv('./data/output/IBO_tag_data/train.csv')
    # test = pd.read_csv('./data/output/IBO_tag_data/test.csv')
    test = pd.read_csv('crf.csv')

    is_pos = False

    window = 3

    train_getter = SentenceGetter(train)
    train_sentences = train_getter.sentences
    test_getter = SentenceGetter(test)
    test_sentences = test_getter.sentences
    model = CRFModel()
    X_train = [model.sent2features(s, is_pos, window) for s in train_sentences]
    y_train = [model.sent2labels(s) for s in train_sentences]
    X_val = [model.sent2features(s, is_pos, window) for s in test_sentences]
    y_val = [model.sent2labels(s) for s in test_sentences]
    # from sklearn.metrics import classification_report
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        all_possible_transitions=True)
    crf.fit(X_train, y_train)
    y_pred = crf.predict(X_val)
    a = []
    for pred in y_pred:
        a = a+pred
    test['predict_f'] = a
    #
    test.to_csv('crf.csv', index=False)
    print(classification_report(y_val, y_pred, digits=4))
# parameters = {'c1':[0.1, 0.2, 0.5, 1], 'c2':[0.1, 0.2, 0.5, 1], 'max_iterations': [100, 150, 200]}