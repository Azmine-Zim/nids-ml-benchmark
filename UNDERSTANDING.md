# Understanding NIDS-Lab

Ten questions worth being able to answer out loud. The numbers are from seed 42 in `results/run_all.md`.

## 1. What does this project do?

It scores two public intrusion-detection datasets with the same four classical models and writes precision, recall, F1, ROC-AUC, a confusion matrix, and a ROC curve. It does not generate attacks.

## 2. Which model is allowed to see labels?

Only Random Forest. Isolation Forest, One-Class SVM, and the autoencoder are fit on normal training rows. A higher Random Forest score is expected, and it is not a like-for-like win.

## 3. What is the positive class?

Any attack. `normal` on NSL-KDD and `BENIGN` on CICIDS2017 are the negative class. Individual attack names are collapsed into that one bit.

## 4. Why can NSL-KDD show ROC-AUC 0.94 and recall 0.55 for Isolation Forest?

ROC-AUC uses the continuous anomaly score, so it only asks whether attacks rank above normal traffic. Precision, recall, and F1 use sklearn's contamination threshold (0.01). That cut is strict: precision is 0.9789 and recall is 0.5502. The ranking is stronger than the single operating point.

## 5. Why does One-Class SVM train on 4,000 rows?

The RBF kernel keeps a training row in the model. The normal sets are too big for that (67,343 on NSL-KDD, 16,000 in the CICIDS train sample). The test set is not cut down to 4,000. On NSL-KDD the test file is the full `KDDTest+`.

## 6. What is the autoencoder, in one sentence?

A plain dense network, 64-32-16-32-64, trained with mean squared error to reconstruct normal flows. The anomaly score is the reconstruction error. There is no latent distribution and no KL term, so it is not a VAE.

## 7. How is the NSL-KDD split different from the CICIDS split?

NSL-KDD uses the official files: all of `KDDTrain+` (125,973 rows) and all of `KDDTest+` (22,544 rows). CICIDS2017 has no official split. This repo streams the eight MachineLearningCSV files (2,271,320 BENIGN, 556,556 attack, 2,867 bad rows dropped), keeps a balanced reservoir of 20,000 and 20,000, then makes its own 80/20 split.

## 8. Why is Random Forest almost perfect on CICIDS and only 0.77 F1 on NSL-KDD?

On CICIDS the train and test rows are a random cut of the same days, and the sample is balanced, so the supervised model sees the same attack types on both sides (F1 0.9977). NSL-KDD's official test file is a different mix, including attacks that are hard for a model trained on `KDDTrain+` (F1 0.7659, ROC-AUC 0.9598).

## 9. Where is the scaler fit?

For the three one-class models, one-hot encoding and scaling are fit on normal training rows only, so attack rows do not set the feature scale. Random Forest fits that encoder on every training row, because it is allowed to use those rows.

## 10. What should you not claim in an interview?

Do not say the lab detects today's attacks, measures latency, or studies a variational autoencoder. Do not quote the CICIDS Random Forest score without saying the sample is balanced and randomly split. NSL-KDD is 1998/1999 traffic. The honest result is the gap between one-class models and the supervised reference, plus the gap between AUC and the chosen threshold.
