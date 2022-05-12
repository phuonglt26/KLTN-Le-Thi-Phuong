# from sklearn.model_selection import train_test_split
#
# from modules.evaluate import cal_sentiment_prf
# from modules.polarity.chi2 import Polaritychi2Model
# from modules.preprocess import preprocess
from collections import Counter

import pandas as pd
from pyvi import ViTokenizer, ViPosTagger
from pyvi import ViUtils
from load_data import load_not_tag_ner_yet_data, load_comparative_stc_data
from models import Input, NotTagNerYetInput
from modules.models import PivyTokenizer
from modules.preprocess import tokenize, tag_IBO


def train_comparative_stc():
    raise NotImplementedError


def train_aspect():
    raise NotImplementedError


def train_polarity():
    raise NotImplementedError


def train_ner():
    raise NotImplementedError


if __name__ == '__main__':
    # not_tag_yet_ner_df = pd.read_csv('./data/input/cs_train_2k_data.csv.txt', sep='	',
    #                                  usecols=['sentence_idx', 'title', 'main', 'label'])
    # inputs, outputs = load_comparative_stc_data('./data/input/cs_train_2k_data.csv', 'sentence_idx', 'main', 'label')

    # p = ComparativeStcPipeline('./data/input/cs_train_2k.csv')

    x = pd.read_csv('./data/input/ner_800k.txt', sep='	',
                                     usecols=['sentence_idx', 'main', 'S1', 'O1'])
    not_tag_ner_yet_inputs = load_not_tag_ner_yet_data(x, 'sentence_idx', 'main', 'S1', 'O1')
    # # x = NotTagNerYetInput(1,
    # #                       'Vậy nên nếu chọn so sánh Elantra Sport và Cerato Premium thì giá xe của Elantra vẫn cao hơn đối thủ khá nhiều, rơi vào khoảng gần 100 triệu đồng',
    # #                       'Elantra Sport', 'Cerato Premium')
    tag_IBO(not_tag_ner_yet_inputs)

    # a = [1,2,3,3,2]
    # blog_idx = list(Counter(a).keys())
    # l = len(Counter(a).keys())
    # rate = int(l*(1-0.2))
    #
    # x = blog_idx[:rate]
    # y = blog_idx[rate:]
    # print(x)

    # print(ViTokenizer.tokenize(u"Mada CX-5 có động cơ khỏe nhất"))
    #
    # print(ViPosTagger.postagging(ViTokenizer.tokenize(u"Về trang bị an toàn, Toyota vios gr-s 2021 nhỉnh hơn nhờ hệ thống kiểm soát lực kéo và cảm biến lùi")))

    #
    #
    #
    # ViUtils.remove_accents(u"Trường đại học bách khoa hà nội")
