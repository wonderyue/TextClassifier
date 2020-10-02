import numpy as np
from sklearn.model_selection import train_test_split
from Utils import performance


class lr_classifer:
    _w = None

    def __sigmoid(self, x, w):
        return 1 / (1 + np.exp(-x.dot(w)))

    def train(
        self,
        train_data,
        classes,
        lambda_arr=[0.1, 0.25, 0.5, 0.75, 1],
        learning_rate=0.01,
        max_iter=5000,
        tol=1e-3,
    ):
        # add one column of ones for w0
        train_data = np.concatenate(
            (np.ones((train_data.shape[0], 1)), train_data), axis=1
        )
        x_train, x_valid, y_train, y_valid = train_test_split(
            train_data, classes, test_size=0.3
        )
        # find the best lambda
        max_accuracy = 0
        max_lambda = None
        for l in lambda_arr:
            w = np.ones(train_data.shape[1])
            old_w = w
            for _ in range(max_iter):
                w = w + learning_rate * (
                    x_train.T.dot(y_train - self.__sigmoid(x_train, w)) - l * w
                )
                if sum(abs(w - old_w)) < tol:
                    break
                old_w = w
            accuracy, precision, recall, f1 = self.__test(x_valid, y_valid, w)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                max_lambda = l
        # train with best lambda and full train data
        self._w = np.ones(train_data.shape[1])
        for _ in range(max_iter):
            self._w = self._w + learning_rate * (
                train_data.T.dot(classes - self.__sigmoid(train_data, self._w))
                - max_lambda * self._w
            )
        return max_lambda

    def __test(self, test_data, classes, w):
        predict = np.where(self.__sigmoid(test_data, w) >= 0.5, 1, 0)
        return performance(predict, classes)

    def test(self, test_data, classes):
        # add one column of ones for w0
        test_data = np.concatenate(
            (np.ones((test_data.shape[0], 1)), test_data), axis=1
        )
        return self.__test(test_data, classes, self._w)
