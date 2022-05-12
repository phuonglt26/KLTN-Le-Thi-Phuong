import datetime
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from load_data import load_polarity_data, load_comparative_stc_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from imblearn.over_sampling import RandomOverSampler

from modules.polarity.model import SentimentModel

TRAIN_PATH = 'data/input/SP/train_sc.csv'
TEST_PATH = 'data/input/SP/test_sc.csv'


def tune_k_best(aspect, k_list, X_train, y_train, X_test, y_test, model, save=True):
    print('=' * 20 + " Tuning k_best SP model " + aspect + ' ' + (50 - len(' Tuning k_best SP model ')) * '=')
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

        f1 = f1_score(_y_test, predict, pos_label=1)
        print('* F1-score with k={} : '.format(k), f1)
        if f1 > max_f1:
            max_f1 = f1
            k_best = k
    print('=' * 20 + " Result of Tuning k_best SP model " + aspect + ' ' + (
            50 - len(' Result of Tuning k_best SP model ')) * '=')

    print("- Aspect         : ", aspect)
    print("- K_best         : ", k_best)
    print("- Max_F1 score   : ", max_f1)
    result = pd.DataFrame({'score': [aspect, model.model, k_best, max_f1]},
                          index=['Aspect', 'Model', 'K_best', 'Max_F1'])
    if save:
        result.to_csv('./data/output/evaluate/SP/tune/SP_result_{a}_{d}.csv'.format(a=aspect,
                                                                                    d=datetime.datetime.now().strftime(
                                                                                        '%Y%m%d_%H%M%S')))
    return k_best


def train_SP_chi2(aspect, k_best, X_train, y_train, X_test, y_test, model, oversample=False, save=True):
    print('=' * 20 + ' Training SP with Chi2 ' + aspect + ' ' + (50 - len(' Training SP with Chi2 ')) * '=')
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
        print(Counter(_y_train))
        print('  RandomOverSampling DONE!')

    print('* Training ...')
    model.train(_X_train, _y_train)
    print('  Training DONE!')
    predict = model.predict(_X_test)
    neg_p, neg_r, neg_f1 = model.evaluate(_y_test, predict, 1)
    pos_p, pos_r, pos_f1 = model.evaluate(_y_test, predict, 3)
    neu_p, neu_r, neu_f1 = model.evaluate(_y_test, predict, 2)
    print("- K_best          :", k_best)
    print("- F1 negative     :", neg_f1)
    print("- F1 positive     :", pos_f1)
    print("- F1 neutral      :", neu_f1)

    result = pd.DataFrame({'score': [aspect, model.model, k_best, neg_f1, pos_f1, neu_f1]},
                          index=['Aspect', 'Model', 'K_best', 'F1-negative', 'F1-positive', 'F1-neutral'])

    if save:
        if oversample:
            result.to_csv(
                './data/output/evaluate/SP/chi2/SP_result_oversample_{m}_{a}.csv'.format(
                    m=model.model,
                    a=aspect))
        else:
            result.to_csv('./data/output/evaluate/SP/chi2/SP_result_{m}_{a}.csv'.format(
                m=model.model,
                a=aspect))
    return neg_f1, pos_f1, neu_f1


if __name__ == '__main__':
    test_df = pd.read_csv(TEST_PATH)
    model = LogisticRegression()
    k_best = 3000
    is_oversample = True
    aspect_list = ['sức_mạnh', 'thiết_kế', 'giá', 'tính_năng', 'an_toàn']
    neg_f1 = []
    pos_f1 = []
    neu_f1 = []

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
        AC_model = SentimentModel(vocab_path, model)

        # tuning k_best
        # k_list = [500, 1000, 1500, 2000, 2500, 3000]
        # tune k_best
        # k_best = tune_k_best(k_list, X_train, y_train, X_test, y_test, CSI_model)
        # train model
        _neg_f1, _pos_f1, _neu_f1 = train_SP_chi2(aspect, k_best, X_train, y_train, X_test, y_test, AC_model,
                                                  oversample=is_oversample,
                                                  save=True)
        neg_f1.append(_neg_f1)
        pos_f1.append(_pos_f1)
        neu_f1.append(_neu_f1)
    macro_neg_f1 = np.array(neg_f1).mean()
    macro_pos_f1 = np.array(pos_f1).mean()
    macro_neu_f1 = np.array(neu_f1).mean()
    print('=' * 20 + ' Performance of SP model ' + (50 - len(' Performance of SP model ')) * '=')
    print("- Macro-F1 negative         :", macro_neg_f1)
    print("- Macro-P positive          :", macro_pos_f1)
    print("- Macro-R neutral           :", macro_neu_f1)
    result = pd.DataFrame({'score': [model, macro_neg_f1, macro_pos_f1, macro_neu_f1]},
                          index=['Model', 'Macro-F1 negative', 'Macro-F1 positive', 'Macro-F1 neutral'])
    result.to_csv(
        './data/output/evaluate/SP/chi2/SP_result_{m}_{o}.csv'.format(m=model, o=is_oversample * 1))
