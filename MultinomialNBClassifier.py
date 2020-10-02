import math
import numpy as np
from Utils import performance


class multinomial_nb_classifier:
    # P(class)
    prior = [0, 0]
    # cond_prob[class][feature_index]: P(feature | class) with add-one laplace smoothing
    cond_prob = [[], []]

    def train(self, train_data, classes):
        self.prior[1] = sum(classes) / len(classes)
        self.prior[0] = 1 - self.prior[1]
        for clazz in (0, 1):
            self.cond_prob[clazz] = []
            count_array = np.sum(train_data[classes == clazz], axis=0)
            total = sum(count_array) + train_data.shape[1]
            for count in count_array:
                # add-one laplace smoothing
                self.cond_prob[clazz].append((count + 1) / total)

    def test(self, test_data, classes):
        predict = []
        for row in range(len(test_data)):
            row_data = test_data[row]
            # clazz = classes[row]
            posterior_class = None
            posterior_score = float("-inf")
            for c in (0, 1):
                score = math.log2(self.prior[c])
                for index in range(len(row_data)):
                    if row_data[index] > 0:
                        score += row_data[index] * math.log2(self.cond_prob[c][index])
                if score > posterior_score:
                    posterior_score = score
                    posterior_class = c
            predict.append(posterior_class)
        return performance(predict, classes)

