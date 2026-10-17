# NSL-KDD results

Attack is the positive class. Test file is full KDDTest+ for every model.
Random Forest sees training labels. The other three do not.
One-Class SVM uses 4000 normal training rows. The others use the full normal training set (Random Forest uses every training row).

| model | setup | precision | recall | f1 | roc_auc |
| --- | --- | ---: | ---: | ---: | ---: |
| iforest | one-class, full normal train | 0.9789 | 0.5502 | 0.7045 | 0.9365 |
| ocsvm | one-class, 4000 normal rows | 0.9310 | 0.6482 | 0.7643 | 0.8824 |
| rf | supervised, full train | 0.9690 | 0.6331 | 0.7659 | 0.9598 |
| ae | one-class, full normal train | 0.9826 | 0.4304 | 0.5986 | 0.9567 |

Overlay ROC: `results/figures/nslkdd_roc_all.png`.
