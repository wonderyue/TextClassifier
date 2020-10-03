import math
import numpy as np
from Utils import performance
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


class sgd_classifier:
    clf = SGDClassifier()
    param_grid = {
        "max_iter": [100, 1000, 2500],
        "alpha": [0.01, 0.1, 1],
        "penalty": ["l1", "l2"],
    }
    best_estimator = None

    @ignore_warnings(category=ConvergenceWarning)
    def train(self, train_data, classes):
        gridSearch = GridSearchCV(self.clf, param_grid=self.param_grid,)
        gridSearch.fit(train_data, classes)
        self.best_estimator = gridSearch.best_estimator_
        return f"max_iter:{self.best_estimator.max_iter}, alpha:{self.best_estimator.alpha}, penalty:{self.best_estimator.penalty}"

    def test(self, test_data, classes):
        predict = self.best_estimator.predict(test_data)
        return performance(predict, classes)
