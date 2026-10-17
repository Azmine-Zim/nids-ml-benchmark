# CICIDS2017, ae

Not the full CICIDS2017 table. Every MachineLearningCSV file is streamed. A reservoir sample keeps at most 20000 BENIGN rows and 20000 attack rows (seed 42). Infinite or missing values are dropped before sampling. The sample is split 80/20 into train and test, stratified on the attack bit. CICIDS2017 has no official train/test files, so this split is defined here. Destination Port stays in the features as a number. One-class. A plain dense autoencoder (64-32-16-32-64, MSE loss, no KL term) is fit on 90% of the BENIGN rows in the CICIDS train sample for 15 epochs. The anomaly score is mean squared reconstruction error. The threshold is the 99th percentile of that error on the held-out 10% of BENIGN rows.

| split | rows | normal | attack |
| --- | ---: | ---: | ---: |
| train sample (80%) | 32000 | 16000 | 16000 |
| test sample (20%) | 8000 | 4000 | 4000 |

Encoder fit on: normal (16000 rows). Encoded width: 78. Seed: 42.

Not the full CICIDS2017 table. Every MachineLearningCSV file is streamed. A reservoir sample keeps at most 20000 BENIGN rows and 20000 attack rows (seed 42). Infinite or missing values are dropped before sampling. The sample is split 80/20 into train and test, stratified on the attack bit. CICIDS2017 has no official train/test files, so this split is defined here. Destination Port stays in the features as a number.

| precision | recall | f1 | roc_auc |
| ---: | ---: | ---: | ---: |
| 0.9693 | 0.3390 | 0.5023 | 0.8338 |

Confusion matrix (rows true normal/attack, columns predicted): TN=3957, FP=43, FN=2644, TP=1356.
