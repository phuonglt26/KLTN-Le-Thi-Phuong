import pandas as pd

from modules.models import PivyTokenizer
from modules.preprocess import tokenize, pos_tag

DATA_PATH = 'data/input/SP/test_sc.csv'
# if __name__ == '__main__':
#     df = pd.read_csv(DATA_PATH)
#     df = df.dropna(subset=['main'])
#     df['main'] = df['main'].apply(tokenize)
#     df['main'] = df['main'].apply(lambda x: x.strip())
#     df['main'] = df['main'].apply(lambda x: x.lower())
#     df.to_csv(DATA_PATH, index=False)
#
if __name__ == '__main__':
    x = 'tôi là Lê_Thị_Phương'
    a = pos_tag(x)
    print(a)