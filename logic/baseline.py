import datetime
import pickle
import time
from collections import Counter

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, recall_score, log_loss
from sklearn.metrics import f1_score, accuracy_score, precision_score

from logic.Utils import Utils
from logic.classifiers import Classifiers
from logic.text_processing import TextProcessing
from logic.lexical_vectorizer import LexicalVectorizer
from root import DIR_DATA, DIR_INPUT


class Baseline(object):

    def __init__(self, lang: str = 'es', iteration: int = 10, fold: int = 10):
        print('{0}'.format(type(self).__name__))
        self.lang = lang
        self.iteration = iteration
        self.fold = fold
        self.classifiers = Classifiers.dict_classifiers
        self.tp = TextProcessing(lang=lang)
        self.lv = LexicalVectorizer(lang=lang, text_processing=self.tp)
        self.ut = Utils(lang=lang, text_processing=self.tp)

    def run(self, file_name_train: str, file_name_test: str):
        # 1. import training and test data
        print('\t+ Import training...')
        x_train, y_train = self.ut.get_data(file_name=file_name_train)
        print('\t+ Import test...')
        x_test, y_test = self.ut.get_data(file_name=file_name_test)
        # 2. Feature extraction
        print('\t+ Get Feature')

        x_train = self.lv.transform(x_train)
        x_test = self.lv.transform(x_test)

        print('\t\t - Sample train:', sorted(Counter(y_train).items()))
        print('\t\t - Sample test:', sorted(Counter(y_test).items()))

        # 3. Over Sampling
        k_fold = ShuffleSplit(n_splits=self.fold, test_size=0.25, random_state=42)
        print('\t+ Over Sampling')
        ros_train = RandomOverSampler(random_state=1000)
        x_train, y_train = ros_train.fit_resample(x_train, y_train)
        print('\t\t - train:', sorted(Counter(y_train).items()))
        ros_test = RandomOverSampler(random_state=1000)
        x_test, y_test = ros_test.fit_resample(x_test, y_test)
        print('\t\t - test:', sorted(Counter(y_test).items()))

        print('\t+ Training...')
        result_train = {}
        result_test = {}
        for clf_name, clf in self.classifiers.items():
            accuracies_scores = []
            for train_index, test_index in k_fold.split(x_train, y_train):
                data_train = x_train[train_index]
                target_train = y_train[train_index]

                data_test = x_train[test_index]
                target_test = y_train[test_index]

                clf.fit(data_train, target_train)
                predict = clf.predict(data_test)
                # Accuracy
                accuracy = accuracy_score(target_test, predict)
                accuracies_scores.append(accuracy)

            average_accuracy = round(np.mean(accuracies_scores) * 100, 2)
            result_train[clf_name] = average_accuracy

            y_predict = []
            for features in x_test:
                features = features.reshape(1, -1)
                value = clf.predict(features)[0]
                y_predict.append(value)

            accuracy_predict = accuracy_score(y_test, y_predict)
            result_test[clf_name] = {'Accuracy': round(np.mean(accuracy_predict) * 100, 2),
                                     'Classification Report\n': classification_report(y_test, y_predict)}

        for k, v in result_train.items():
            print('\t\t - {0}: {1}'.format(k, v))

        print('\t+ Evaluation...')
        for k, v in result_test.items():
            print('\t\t - {0}'.format(k))
            for kk, vv in v.items():
                print('\t\t\t- {0}: {1}'.format(kk, vv))


if __name__ == '__main__':
    base = Baseline(lang='es', iteration=10, fold=10)
    base.run(file_name_train='tass2018_es_train', file_name_test='tass2018_es_development')