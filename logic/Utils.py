import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from logic.text_processing import TextProcessing
from root import DIR_INPUT


class Utils(object):

    def __init__(self, lang: str = 'es', text_processing=None):
        try:
            if text_processing is None:
                self.tp = TextProcessing(lang=lang)
            else:
                self.tp = text_processing
        except Exception as e:
            print('Error init: {0}'.format(e))

    def get_data(self, file_name: str, sep: str = ','):
        try:
            le = LabelEncoder()
            data = pd.read_csv('{0}{1}.csv'.format(DIR_INPUT, file_name), sep=sep)
            x = [self.tp.transformer(row) for row in data['content'].tolist()]
            y = le.fit_transform(data['sentiment/polarity/value'])
            print('\t\t - Dataset size :(x: {} , y: {})'.format(len(x), len(y)))
            return x, y
        except Exception as e:
            print('Error get_data: {0}'.format(e))