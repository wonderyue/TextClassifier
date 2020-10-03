# TextClassifier
# How to run

```shell
Instructions:

optional arguments:
  -h, --help            show this help message and exit
  -nb                   Discrete Naive Bayes Classifier
  -mnb                  Multinomial Naive Bayes Classifier
  -lr                   Logistic Regression Classifier
  -sgd                  Stochastic Gradient Descent Classifier
  -train                train_data_path
  -test                 test_data_path
```

For example, run NBClassifier and SGDClassifier for dataset1/train and dataset1/test

```shell
python ./__main__.py -train "dataset1/train" -test "dataset1/test" -nb -sgd  
```

## Results

[full test](/Classifier.ipynb)

```
+------------------------------------------------------------------------------------------------------------+
|                                                  dataset1                                                  |
+------------+--------------+----------+-----------+--------+--------+---------------------------------------+
| classifier |  data mode   | accuracy | precision | recall |   f1   |            hyper-parameter            |
+------------+--------------+----------+-----------+--------+--------+---------------------------------------+
|     nb     |  bernoulli   |  76.36%  |  100.00%  | 13.08% | 23.13% |                                       |
|    mnb     | bag_of_words |  94.14%  |   93.22%  | 84.62% | 88.71% |                                       |
|     lr     | bag_of_words |  93.93%  |   87.97%  | 90.00% | 88.97% |               alpha:0.5               |
|     lr     |  bernoulli   |  96.23%  |   93.08%  | 93.08% | 93.08% |               alpha:0.25              |
|    sgd     | bag_of_words |  91.63%  |   86.89%  | 81.54% | 84.13% |  max_iter:100, alpha:0.1, penalty:l2  |
|    sgd     |  bernoulli   |  96.03%  |   95.12%  | 90.00% | 92.49% | max_iter:1000, alpha:0.01, penalty:l2 |
+------------+--------------+----------+-----------+--------+--------+---------------------------------------+
+-------------------------------------------------------------------------------------------------------------+
|                                                   dataset2                                                  |
+------------+--------------+----------+-----------+---------+--------+---------------------------------------+
| classifier |  data mode   | accuracy | precision |  recall |   f1   |            hyper-parameter            |
+------------+--------------+----------+-----------+---------+--------+---------------------------------------+
|     nb     |  bernoulli   |  87.85%  |   85.56%  | 100.00% | 92.22% |                                       |
|    mnb     | bag_of_words |  94.48%  |   94.79%  |  97.70% | 96.22% |                                       |
|     lr     | bag_of_words |  97.05%  |   96.30%  |  99.74% | 97.99% |               alpha:0.1               |
|     lr     |  bernoulli   |  96.87%  |   95.83%  | 100.00% | 97.87% |               alpha:0.1               |
|    sgd     | bag_of_words |  95.03%  |   95.73%  |  97.44% | 96.58% |  max_iter:100, alpha:0.01, penalty:l2 |
|    sgd     |  bernoulli   |  97.24%  |   96.31%  | 100.00% | 98.12% | max_iter:2500, alpha:0.01, penalty:l2 |
+------------+--------------+----------+-----------+---------+--------+---------------------------------------+
+-----------------------------------------------------------------------------------------------------------+
|                                                  dataset3                                                 |
+------------+--------------+----------+-----------+--------+--------+--------------------------------------+
| classifier |  data mode   | accuracy | precision | recall |   f1   |           hyper-parameter            |
+------------+--------------+----------+-----------+--------+--------+--------------------------------------+
|     nb     |  bernoulli   |  71.49%  |  100.00%  | 12.75% | 22.62% |                                      |
|    mnb     | bag_of_words |  93.86%  |   94.81%  | 85.91% | 90.14% |                                      |
|     lr     | bag_of_words |  93.86%  |   89.54%  | 91.95% | 90.73% |              alpha:0.5               |
|     lr     |  bernoulli   |  95.39%  |   91.56%  | 94.63% | 93.07% |              alpha:0.5               |
|    sgd     | bag_of_words |  94.08%  |   90.13%  | 91.95% | 91.03% | max_iter:2500, alpha:0.1, penalty:l2 |
|    sgd     |  bernoulli   |  95.39%  |   91.03%  | 95.30% | 93.11% | max_iter:100, alpha:0.01, penalty:l2 |
+------------+--------------+----------+-----------+--------+--------+--------------------------------------+
```

