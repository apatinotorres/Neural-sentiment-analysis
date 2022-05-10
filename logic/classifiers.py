import multiprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


class Classifiers(object):

    cores = multiprocessing.cpu_count() - 1
    dict_classifiers = dict()
    dict_classifiers['SVM'] = svm.SVC(kernel='poly', degree=3, C=1)
    dict_classifiers['LogisticRegression'] = LogisticRegression(multi_class="multinomial", solver="lbfgs")
    dict_classifiers['RandomForest'] = RandomForestClassifier(n_jobs=cores)



