from collections import Counter
import os
import re
import numpy as np


def get_vocabulary(dir):
    vocabulary = set()
    for root, _, files in os.walk(dir):
        for name in files:
            with open(
                os.path.join(root, name), "r", encoding="utf-8", errors="ignore"
            ) as fd:
                text = fd.read()
                words = re.findall(r"\w+", text)
                fd.close()
                vocabulary.update([word.lower() for word in words])
    return list(vocabulary)


def _parse(dir, vocabulary, positive_label="spam", is_bernoulli=True):
    data_set = []
    classes = []
    for root, _, files in os.walk(dir):
        for name in files:
            clazz = os.path.basename(root)
            with open(
                os.path.join(root, name), "r", encoding="utf-8", errors="ignore"
            ) as fd:
                text = fd.read()
                words = re.findall(r"\w+", text)
                counter = Counter([word.lower() for word in words])
                fd.close()
                if is_bernoulli:
                    data_set.append(
                        np.array([1 if counter[word] > 0 else 0 for word in vocabulary])
                    )
                else:
                    data_set.append(np.array([counter[word] for word in vocabulary]))
                classes.append(clazz == positive_label)
    return np.array(data_set), np.array(classes)


def bag_of_words(dir, vocabulary, positive_label="spam"):
    return _parse(dir, vocabulary, positive_label, False)


def bernoulli(dir, vocabulary, positive_label="spam"):
    return _parse(dir, vocabulary, positive_label, True)
