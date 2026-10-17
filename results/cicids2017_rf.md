# CICIDS2017, rf

Not the full CICIDS2017 table. Every MachineLearningCSV file is streamed. A reservoir sample keeps at most 20000 BENIGN rows and 20000 attack rows (seed 42). Infinite or missing values are dropped before sampling. The sample is split 80/20 into train and test, stratified on the attack bit. CICIDS2017 has no official train/test files, so this split is defined here. Destination Port stays in the features as a number. Supervised reference. Random Forest (100 trees) is fit on the full CICIDS train sample with the attack label. A score of 0.5 is the decision threshold. This is not comparable one-to-one with the one-class models.

| split | rows | normal | attack |
| --- | ---: | ---: | ---: |
| train sample (80%) | 32000 | 16000 | 16000 |
| test sample (20%) | 8000 | 4000 | 4000 |

Encoder fit on: all (32000 rows). Encoded width: 78. Seed: 42.

Not the full CICIDS2017 table. Every MachineLearningCSV file is streamed. A reservoir sample keeps at most 20000 BENIGN rows and 20000 attack rows (seed 42). Infinite or missing values are dropped before sampling. The sample is split 80/20 into train and test, stratified on the attack bit. CICIDS2017 has no official train/test files, so this split is defined here. Destination Port stays in the features as a number.

| precision | recall | f1 | roc_auc |
| ---: | ---: | ---: | ---: |
| 0.9987 | 0.9968 | 0.9977 | 0.9999 |

Confusion matrix (rows true normal/attack, columns predicted): TN=3995, FP=5, FN=13, TP=3987.
