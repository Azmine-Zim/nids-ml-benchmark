# NSL-KDD, iforest

One-class. Isolation Forest is fit only on normal rows from KDDTrain+. Threshold is sklearn's contamination offset (contamination=0.01, 100 trees).

| split | rows | normal | attack |
| --- | ---: | ---: | ---: |
| KDDTrain+ | 125973 | 67343 | 58630 |
| KDDTest+ | 22544 | 9711 | 12833 |

Encoder fit on: normal (67343 rows). Encoded width: 77. Seed: 42.

| precision | recall | f1 | roc_auc |
| ---: | ---: | ---: | ---: |
| 0.9789 | 0.5502 | 0.7045 | 0.9365 |

Confusion matrix (rows true normal/attack, columns predicted): TN=9559, FP=152, FN=5772, TP=7061.
