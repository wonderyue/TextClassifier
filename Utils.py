def _performance(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = 0 if tp == 0 else tp / (tp + fp)
    recall = 0 if tp == 0 else tp / (tp + fn)
    f1 = (
        0
        if recall * precision == 0
        else 2 * (recall * precision) / (recall + precision)
    )
    return accuracy, precision, recall, f1


def performance(predict, target):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(target)):
        if predict[i] == target[i]:
            if predict[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if predict[i] == 1:
                fp += 1
            else:
                fn += 1
    return _performance(tp, tn, fp, fn)
