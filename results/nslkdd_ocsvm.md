# NSL-KDD, ocsvm

One-class. One-Class SVM (RBF, nu=0.01) is fit on a random sample of 4000 normal KDDTrain+ rows. The full normal set is too large for an RBF kernel. The test file is not sampled.

| split | rows | normal | attack |
| --- | ---: | ---: | ---: |
| KDDTrain+ | 125973 | 67343 | 58630 |
| KDDTest+ | 22544 | 9711 | 12833 |

Encoder fit on: normal (67343 rows). Encoded width: 77. Seed: 42.

Training sample: 4000 rows. The test file is full.

| precision | recall | f1 | roc_auc |
| ---: | ---: | ---: | ---: |
| 0.9310 | 0.6482 | 0.7643 | 0.8824 |

Confusion matrix (rows true normal/attack, columns predicted): TN=9095, FP=616, FN=4515, TP=8318.
