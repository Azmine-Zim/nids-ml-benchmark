# CICIDS2017, iforest

Not the full CICIDS2017 table. Every MachineLearningCSV file is streamed. A reservoir sample keeps at most 20000 BENIGN rows and 20000 attack rows (seed 42). Infinite or missing values are dropped before sampling. The sample is split 80/20 into train and test, stratified on the attack bit. CICIDS2017 has no official train/test files, so this split is defined here. Destination Port stays in the features as a number. One-class. Isolation Forest is fit only on BENIGN rows in the CICIDS train sample. Threshold is sklearn's contamination offset (contamination=0.01, 100 trees).

| split | rows | normal | attack |
| --- | ---: | ---: | ---: |
| train sample (80%) | 32000 | 16000 | 16000 |
| test sample (20%) | 8000 | 4000 | 4000 |

Encoder fit on: normal (16000 rows). Encoded width: 78. Seed: 42.

Not the full CICIDS2017 table. Every MachineLearningCSV file is streamed. A reservoir sample keeps at most 20000 BENIGN rows and 20000 attack rows (seed 42). Infinite or missing values are dropped before sampling. The sample is split 80/20 into train and test, stratified on the attack bit. CICIDS2017 has no official train/test files, so this split is defined here. Destination Port stays in the features as a number.

| precision | recall | f1 | roc_auc |
| ---: | ---: | ---: | ---: |
| 0.9247 | 0.1227 | 0.2167 | 0.7264 |

Confusion matrix (rows true normal/attack, columns predicted): TN=3960, FP=40, FN=3509, TP=491.
