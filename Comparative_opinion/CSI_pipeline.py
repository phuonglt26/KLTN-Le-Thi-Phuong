import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from load_data import load_comparative_stc_data
from modules.comparative_stc.model import ComparativeSentenceModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from modules.embeddings import TfidfModel
from modules.models import PivyTokenizer

from imblearn.over_sampling import RandomOverSampler

TRAIN_PATH = './data/input/CSI/train_cs.csv'
TEST_PATH = './data/input/CSI/test_cs.csv'


def tune_k_best(k_list, X_train, y_train, X_test, y_test, model, save=True):
    print('=' * 20 + " Tuning k_best CSI model " + (50 - len(' Tuning k_best CSI model ')) * '=')
    max_f1 = 0
    k_best = 0
    _y_train = [i.score for i in y_train]
    _y_test = [i.score for i in y_test]
    for k in k_list:
        model.chi2_dict(k)
        print('*** Representing with k={} ...'.format(k))

        train_x = model.chi2_represent(X_train)
        test_x = model.chi2_represent(X_test)
        model.train(train_x, _y_train)
        print('  Representing with k={} DONE!'.format(k))
        predict = model.predict(test_x)

        f1 = f1_score(_y_test, predict)
        print('* F1-score with k={} : '.format(k), f1)
        if f1 > max_f1:
            max_f1 = f1
            k_best = k
    print('=' * 20 + " Result of Tuning k_best CSI model " + (50 - len(' Result of Tuning k_best CSI model ')) * '=')
    print("- K_best         : ", k_best)
    print("- Max_F1 score   : ", max_f1)
    result = pd.DataFrame({'score': [model.model, k_best, max_f1]},
                          index=['Model', 'K_best', 'Max_F1'])
    if save:
        result.to_csv('./data/output/evaluate/CSI/tune/CSI_result_{}.csv'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    return k_best


def train_CSI_chi2(k_best, X_train, y_train, X_test, y_test, model, oversample=False, save=True):
    print('=' * 20 + ' Training CSI with Chi2 ' + (50 - len(' Training CSI with Chi2 ')) * '=')
    model.chi2_dict(k_best)
    _y_train = [i.score for i in y_train]
    _y_test = [i.score for i in y_test]
    print('* Representing using Chi2 ...')
    _X_train = model.chi2_represent(X_train)
    _X_test = model.chi2_represent(X_test)
    print('  Representing using Chi2 DONE!')

    if oversample:
        print('* RandomOverSampling ...')
        ros = RandomOverSampler(random_state=42, sampling_strategy=0.4)
        _X_train, _y_train = ros.fit_resample(_X_train, _y_train)
        print('  RandomOverSampling DONE!')

    print('* Trainning ...')
    model.train(_X_train, _y_train)
    print('  Trainning DONE!')

    threshold = model.tune_threshold(_X_test, _y_test)
    predict = model.predict(_X_test)
    test_df['predict0'] = predict
    test_df.to_csv('data/output/test_data/CSI/chi2_test_k_{}.csv'.format(k_best))
    p, r, f1 = model.evaluate(_y_test, predict)
    print('=' * 20 + ' Performance of CSI model ' + (50 - len(' Performance of CSI model ')) * '=')
    print("- K_best       :", k_best)
    print("- Threshold    :", threshold)
    print("- F1 score     :", f1)
    print("- Precision    :", p)
    print("- Recall       :", r)
    result = pd.DataFrame({'score': [model.model, k_best, threshold, f1, p, r]},
                          index=['Model', 'K_best', 'Threshold', 'F1', 'Precision', 'Recall'])

    if save:
        if oversample:
            result.to_csv('./data/output/evaluate/CSI/chi2/CSI_result_{}_oversample.csv'.format(
                datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
        else:
            result.to_csv('./data/output/evaluate/CSI/chi2/CSI_result_{}.csv'.format(
                datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))


def train_CSI_tfidf(ngram, X_train, y_train, X_test, y_test, model, save=True):
    print('=' * 20 + ' Training CSI with TF-DIF ' + (50 - len(' Training CSI with TF-DIF ')) * '=')
    tfdif_vectorize = TfidfModel(ngram)
    tfdif_vectorize.train(X_train)

    _y_train = [i.score for i in y_train]
    _y_test = [i.score for i in y_test]

    print('* Representing using TF-DIF ...')
    _X_train = tfdif_vectorize.vectorize(X_train)
    _X_test = tfdif_vectorize.vectorize(X_test)
    print('  Representing using TF-DIF DONE!')

    print('* Training using TF-DIF ...')
    model.train(_X_train, _y_train)
    print('* Training using TF-DIF DONE!')

    # print('* Tuning using TF-DIF ...')
    # threshold = model.tune_threshold(_X_test, _y_test)
    # print('* Tuning using TF-DIF DONE!')

    predict = model.predict(_X_test)
    test_df['predict0'] = predict
    test_df.to_csv('data/output/test_data/CSI/tfdif_test_ngram_{}.csv'.format(ngram))
    p, r, f1 = model.evaluate(_y_test, predict)
    print('=' * 20 + ' Performance of CSI model ' + (50 - len(' Performance of CSI model ')) * '=')
    print("- Ngram        :", ngram)
    # print("- Threshold    :", threshold)
    print("- F1 score     :", f1)
    print("- Precision    :", p)
    print("- Recall       :", r)
    result = pd.DataFrame({'score': [model.model, ngram, 0, f1, p, r]},
                          index=['Model', 'Ngram', 'Threshold', 'F1', 'Precision', 'Recall'])

    if save:
        result.to_csv('./data/output/evaluate/CSI/tfidf/CSI_result_{}.csv'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))


if __name__ == '__main__':
    test_df = pd.read_csv(TEST_PATH)
    model = LGBMClassifier()
    k_best = 10
    tokenizer = PivyTokenizer()
    X_train, y_train = load_comparative_stc_data(path=TRAIN_PATH,
                                                 stc_idx_col_name='sentence_idx',
                                                 stc_col_name='main',
                                                 label_col_name='label')

    X_test, y_test = load_comparative_stc_data(path=TEST_PATH,
                                               stc_idx_col_name='sentence_idx',
                                               stc_col_name='main',
                                               label_col_name='label')
    vocab_path = './data/output/chi2_score_dict/CSI_20220505_033734.csv'
    CSI_model = ComparativeSentenceModel(vocab_path, model)
    # tuning k_best
    # k_list = [500, 1000, 1500, 2000, 2500, 3000]
    # tune k_best
    # k_best = tune_k_best(k_list, X_train, y_train, X_test, y_test, CSI_model)
    # train model
    train_CSI_chi2(2000, X_train, y_train, X_test, y_test, CSI_model, oversample=True, save=True)
    # for ngram in [1,2,3]:
    #     train_CSI_tfidf(ngram, X_train, y_train, X_test, y_test, CSI_model, save=True)
