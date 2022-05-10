from collections import Counter
import datetime
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import FeatureUnion
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler

from logic.Utils import Utils
from logic.classifiers import Classifiers
from logic.text_processing import TextProcessing
from logic.lexical_vectorizer import LexicalVectorizer
from sklearn.model_selection import train_test_split, ShuffleSplit
from root import DIR_INPUT, DIR_RESULTS
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=20)


class BaselineNN(object):

    def __init__(self, lang: str = 'es'):
        print('{0}'.format(type(self).__name__))
        self.lang = lang
        self.classifiers = Classifiers.dict_classifiers
        self.tp = TextProcessing(lang=lang)
        self.lv = LexicalVectorizer(lang=lang, text_processing=self.tp)
        self.ut = Utils(lang=lang, text_processing=self.tp)

    def run(self, file_name_train: str, file_name_test: str):

        date_file = datetime.datetime.now().strftime("%Y-%m-%d %H-%M")
        # 1. import training and test data
        print('\t+ Import training...')
        x_train, y_train = self.ut.get_data(file_name=file_name_train)
        print('\t+ Import test...')
        x_test, y_test = self.ut.get_data(file_name=file_name_test)
        # 2. Feature extraction
        print('\t+ Get Feature')

        tfidf_vector = TfidfVectorizer(max_features=1000, min_df=10, max_df=0.9, ngram_range=(1, 2))

        preprocessor = FeatureUnion([
            ('tfidf_vector', tfidf_vector),
            ('lex_vector', self.lv)
        ])

        preprocessor.fit(x_train)

        x_train = preprocessor.transform(x_train)
        x_test = preprocessor.transform(x_test)

        print('\t\t - Sample train:', sorted(Counter(y_train).items()))
        print('\t\t - Sample test:', sorted(Counter(y_test).items()))

        # 3. Over Sampling
        print('\t+ Over Sampling')
        ros_train = RandomOverSampler(random_state=1000)
        x_train, y_train = ros_train.fit_resample(x_train, y_train)
        print('\t\t - train:', sorted(Counter(y_train).items()))
        ros_test = RandomOverSampler(random_state=1000)
        x_test, y_test = ros_test.fit_resample(x_test, y_test)
        print('\t\t - test:', sorted(Counter(y_test).items()))

        X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=.2, random_state=42)

        shape = X_train.shape[1:]

        # 4. NN Architecture

        model = keras.models.Sequential()
        model.add(keras.layers.Input(shape=shape))
        model.add(keras.layers.Dense(800, activation="selu"))
        model.add(keras.layers.Dense(400, activation="selu"))
        model.add(keras.layers.Dense(4, activation="softmax"))

        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

        model = keras.models.Sequential([
            keras.layers.Input(shape=shape),
            keras.layers.Dense(800, activation="selu"),
            keras.layers.Dense(400, activation="selu"),
            keras.layers.Dense(4, activation="softmax")
        ])

        model.summary()

        model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

        print('\t+ Training...')

        history = model.fit(X_train, y_train, epochs=50, validation_data=(X_valid, y_valid))

        pd.DataFrame(history.history).plot(figsize=(10, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 2)
        plt.savefig('{0}{1}_{2}'.format(DIR_RESULTS, "baselineNN_lexical_tfidf", date_file))
        plt.show()

        score = model.evaluate(x_test, y_test, verbose=1)

        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])

        y_predict = []
        for features in tqdm(x_test):
            features = features.reshape(1, -1)
            value_prob = model.predict(features)
            value = np.argmax(value_prob, axis=1)
            y_predict.append(value)

        accuracy_predict = accuracy_score(y_test, y_predict)
        print('Accuracy: {0}'.format(round(np.mean(accuracy_predict) * 100, 2)))
        print(classification_report(y_test, y_predict))

if __name__ == '__main__':
    base = BaselineNN(lang='es')
    base.run(file_name_train='tass2018_es_train', file_name_test='tass2018_es_development')