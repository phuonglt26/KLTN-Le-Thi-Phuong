import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from load_data import load_comparative_stc_data, load_aspect_data
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

TRAIN_PATH = 'data/input/AC/train_ac.csv'
TEST_PATH = 'data/input/AC/test_ac.csv'


def tune_k_best(aspect, k_list, X_train, y_train, X_test, y_test, model, save=True):
    print('=' * 20 + " Tuning k_best AC model " + aspect + ' ' + (50 - len(' Tuning k_best AC model ')) * '=')
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
    print('=' * 20 + " Result of Tuning k_best AC model " + aspect + ' ' + (
            50 - len(' Result of Tuning k_best AC model ')) * '=')

    print("- Aspect         : ", aspect)
    print("- K_best         : ", k_best)
    print("- Max_F1 score   : ", max_f1)
    result = pd.DataFrame({'score': [aspect, model.model, k_best, max_f1]},
                          index=['Aspect', 'Model', 'K_best', 'Max_F1'])
    if save:
        result.to_csv('./data/output/evaluate/AC/tune/AC_result_{a}_{d}.csv'.format(a=aspect,
                                                                                    d=datetime.datetime.now().strftime(
                                                                                        '%Y%m%d_%H%M%S')))
    return k_best


def train_AC_chi2(aspect, k_best, X_train, y_train, X_test, y_test, model, oversample=False, save=True):
    print('=' * 20 + ' Training AC with Chi2 ' + aspect + ' ' + (50 - len(' Training AC with Chi2 ')) * '=')
    model.chi2_dict(k_best)
    _y_train = [i.score for i in y_train]
    _y_test = [i.score for i in y_test]
    print('* Representing using Chi2 ...')
    _X_train = model.chi2_represent(X_train)
    _X_test = model.chi2_represent(X_test)
    print('  Representing using Chi2 DONE!')

    if oversample:
        print('* RandomOverSampling ...')
        ros = RandomOverSampler(random_state=42)
        _X_train, _y_train = ros.fit_resample(_X_train, _y_train)
        print('  RandomOverSampling DONE!')

    print('* Trainning ...')
    model.train(_X_train, _y_train)
    print('  Trainning DONE!')

    threshold = model.tune_threshold(_X_test, _y_test)
    predict = model.predict(_X_test)
    test_df['predict0'] = predict
    test_df.to_csv('data/output/test_data/AC/chi2_test_{a}_{a}_k_{k}.csv'.format(m=model.model, a=aspect, k=k_best))
    p, r, f1 = model.evaluate(_y_test, predict)
    print('=' * 20 + ' Performance of AC model ' + aspect + ' ' + (50 - len(' Performance of AC model ')) * '=')
    print("- K_best       :", k_best)
    print("- Threshold    :", threshold)
    print("- F1 score     :", f1)
    print("- Precision    :", p)
    print("- Recall       :", r)
    result = pd.DataFrame({'score': [aspect, model.model, k_best, threshold, f1, p, r]},
                          index=['Aspect', 'Model', 'K_best', 'Threshold', 'F1', 'Precision', 'Recall'])

    if save:
        if oversample:

            result.to_csv(
                './data/output/evaluate/AC/chi2/AC_result_{m}_{a}_oversample_{d}.csv'.format(
                    m=model.model,
                    a=aspect,
                    d=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
        else:
            result.to_csv('./data/output/evaluate/AC/chi2/AC_result_{m}_{a}.csv'.format(
                m=model.model,
                a=aspect,
                d=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    return f1, p, r


if __name__ == '__main__':
    test_df = pd.read_csv(TEST_PATH)
    model = LGBMClassifier()
    k_best = 3000
    isoversample = False
    aspect_list = ['sức_mạnh', 'thiết_kế', 'giá', 'tính_năng', 'an_toàn']
    f1 = []
    p = []
    r = []

    for aspect in aspect_list:
        X_train, y_train = load_comparative_stc_data(path=TRAIN_PATH,
                                                     stc_idx_col_name='sentence_idx',
                                                     stc_col_name='main',
                                                     label_col_name=aspect)

        X_test, y_test = load_comparative_stc_data(path=TEST_PATH,
                                                   stc_idx_col_name='sentence_idx',
                                                   stc_col_name='main',
                                                   label_col_name=aspect)
        vocab_path = './data/output/chi2_score_dict/AC_{}.csv'.format(aspect)
        AC_model = ComparativeSentenceModel(vocab_path, model)

        # tuning k_best
        # k_list = [500, 1000, 1500, 2000, 2500, 3000]
        # tune k_best
        # k_best = tune_k_best(k_list, X_train, y_train, X_test, y_test, CSI_model)
        # train model
        _f1, _p, _r = train_AC_chi2(aspect, k_best, X_train, y_train, X_test, y_test, AC_model, oversample=isoversample,
                                    save=True)
        f1.append(_f1)
        p.append(_p)
        r.append(_r)
    macro_f1 = np.array(f1).mean()
    macro_p = np.array(p).mean()
    macro_r = np.array(r).mean()
    print('=' * 20 + ' Performance of AC model ' + (50 - len(' Performance of AC model ')) * '=')
    print("- Macro-F1           :", macro_f1)
    print("- Macro-P            :", macro_p)
    print("- Macro-R            :", macro_r)
    result = pd.DataFrame({'score': [k_best, model, macro_f1, macro_p, macro_r]},
                          index=['K_best', 'Model', 'Macro-F1', 'Macro-P', 'Macro-R'])
    result.to_csv(
        './data/output/evaluate/AC/chi2/AC_result_{o}_{o}_{d}.csv'.format(m=model, o=isoversample * 1,
                                                                          d=datetime.datetime.now().strftime(
                                                                              '%Y%m%d_%H%M%S')))
