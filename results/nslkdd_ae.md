# NSL-KDD, ae

One-class. A plain dense autoencoder (64-32-16-32-64, MSE loss, no KL term) is fit on 90% of the normal KDDTrain+ rows for 15 epochs. The anomaly score is mean squared reconstruction error. The threshold is the 99th percentile of that error on the held-out 10% of normal rows.

| split | rows | normal | attack |
| --- | ---: | ---: | ---: |
| KDDTrain+ | 125973 | 67343 | 58630 |
| KDDTest+ | 22544 | 9711 | 12833 |

Encoder fit on: normal (67343 rows). Encoded width: 77. Seed: 42.

| precision | recall | f1 | roc_auc |
| ---: | ---: | ---: | ---: |
| 0.9826 | 0.4304 | 0.5986 | 0.9567 |

Confusion matrix (rows true normal/attack, columns predicted): TN=9613, FP=98, FN=7310, TP=5523.
