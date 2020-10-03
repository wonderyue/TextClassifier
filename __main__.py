import numpy as np
import argparse
from MultinomialNBClassifier import multinomial_nb_classifier
from NBClassifier import nb_classifier
from LogisticRegressionClassifer import lr_classifer
from SGDClassifier import sgd_classifier
from Parser import get_vocabulary, bag_of_words, bernoulli


def main():
    parser = argparse.ArgumentParser(description="Instructions:")
    parser.add_argument(
        "-nb", dest="nb", help="Discrete Naive Bayes Classifier", action="store_true"
    )
    parser.add_argument(
        "-mnb",
        dest="mnb",
        help="Multinomial Naive Bayes Classifier",
        action="store_true",
    )
    parser.add_argument(
        "-lr", dest="lr", help="Logistic Regression Classifier", action="store_true"
    )
    parser.add_argument(
        "-sgd",
        dest="sgd",
        help="Stochastic Gradient Descent Classifier",
        action="store_true",
    )
    parser.add_argument(
        "-train", dest="train_data_path", help="train_data_path", required=True
    )
    parser.add_argument(
        "-test", dest="test_data_path", help="test_data_path", required=True
    )
    parse(parser.parse_args())


def print_result(arr):
    accuracy, precision, recall, f1 = arr
    print(f"{accuracy=}, {precision=}, {recall=}, {f1=}")


def parse(args):
    vocabulary = get_vocabulary(args.train_data_path)
    bow_train_data, bow_train_classes = bag_of_words(args.train_data_path, vocabulary)
    bow_test_data, bow_test_classes = bag_of_words(args.test_data_path, vocabulary)
    bnl_train_data, bnl_train_classes = bernoulli(args.train_data_path, vocabulary)
    bnl_test_data, bnl_test_classes = bernoulli(args.test_data_path, vocabulary)

    if args.nb:
        nb = nb_classifier()
        nb.train(bow_train_data, bow_train_classes)
        print("Discrete Naive Bayes Classifier:")
        print_result(nb.test(bow_test_data, bow_test_classes))
    if args.mnb:
        mnb = multinomial_nb_classifier()
        mnb.train(bnl_train_data, bnl_train_classes)
        print("Multinomial Naive Bayes Classifier:")
        print_result(mnb.test(bnl_test_data, bnl_test_classes))
    if args.lr:
        np.warnings.filterwarnings("ignore", "overflow")
        lr = lr_classifer()
        print("Logistic Regression Classifier:")
        print("bag_of_words:")
        print("lambda:", lr.train(bow_train_data, bow_train_classes))
        print_result(lr.test(bow_test_data, bow_test_classes))
        print("bernoulli:")
        print("lambda:", lr.train(bnl_train_data, bnl_train_classes))
        print_result(lr.test(bnl_test_data, bnl_test_classes))
    if args.sgd:
        sgd = sgd_classifier()
        print("Stochastic Gradient Descent Classifier:")
        print("bag_of_words:")
        print(sgd.train(bow_train_data, bow_train_classes))
        print_result(sgd.test(bow_test_data, bow_test_classes))
        print("bernoulli:")
        print(sgd.train(bnl_train_data, bnl_train_classes))
        print_result(sgd.test(bnl_test_data, bnl_test_classes))


if __name__ == "__main__":
    main()
