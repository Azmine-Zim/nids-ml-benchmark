# CICIDS2017, ocsvm

Not the full CICIDS2017 table. Every MachineLearningCSV file is streamed. A reservoir sample keeps at most 20000 BENIGN rows and 20000 attack rows (seed 42). Infinite or missing values are dropped before sampling. The sample is split 80/20 into train and test, stratified on the attack bit. CICIDS2017 has no official train/test files, so this split is defined here. Destination Port stays in the features as a number. One-class. One-Class SVM (RBF, nu=0.01) is fit on at most 4000 BENIGN rows from the CICIDS train sample. The test sample is not reduced again.

| split | rows | normal | attack |
| --- | ---: | ---: | ---: |
| train sample (80%) | 32000 | 16000 | 16000 |
| test sample (20%) | 8000 | 4000 | 4000 |

Encoder fit on: normal (16000 rows). Encoded width: 78. Seed: 42.

Not the full CICIDS2017 table. Every MachineLearningCSV file is streamed. A reservoir sample keeps at most 20000 BENIGN rows and 20000 attack rows (seed 42). Infinite or missing values are dropped before sampling. The sample is split 80/20 into train and test, stratified on the attack bit. CICIDS2017 has no official train/test files, so this split is defined here. Destination Port stays in the features as a number.

One-Class SVM was fit on 4000 normal training rows.

| precision | recall | f1 | roc_auc |
| ---: | ---: | ---: | ---: |
| 0.9215 | 0.4640 | 0.6172 | 0.7786 |

Confusion matrix (rows true normal/attack, columns predicted): TN=3842, FP=158, FN=2144, TP=1856.
