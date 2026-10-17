# NSL-KDD, rf

Supervised reference. Random Forest (100 trees) is fit on all KDDTrain+ rows with the attack label. A score of 0.5 is the decision threshold. This is not comparable one-to-one with the one-class models.

| split | rows | normal | attack |
| --- | ---: | ---: | ---: |
| KDDTrain+ | 125973 | 67343 | 58630 |
| KDDTest+ | 22544 | 9711 | 12833 |

Encoder fit on: all (125973 rows). Encoded width: 122. Seed: 42.

| precision | recall | f1 | roc_auc |
| ---: | ---: | ---: | ---: |
| 0.9690 | 0.6331 | 0.7659 | 0.9598 |

Confusion matrix (rows true normal/attack, columns predicted): TN=9451, FP=260, FN=4708, TP=8125.
