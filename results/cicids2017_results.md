# CICIDS2017 results

Not the full CICIDS2017 table. Every MachineLearningCSV file is streamed (2,271,320 BENIGN rows and 556,556 attack rows seen, 2,867 non-finite rows dropped). A reservoir sample keeps 20000 BENIGN rows and 20000 attack rows (seed 42). The caps are equal, so this sample is balanced. The raw files are not. Metrics below are for the balanced sample, not the original mix. The sample is split 80/20 into train and test, stratified on the attack bit. CICIDS2017 has no official train/test files, so this split is defined here. Train and test are a random split of the same days, so a supervised model can look very strong. Destination Port stays in the features as a number.

Random Forest sees training labels. The other three do not.
One-Class SVM uses at most 4000 normal training rows.

| model | precision | recall | f1 | roc_auc |
| --- | ---: | ---: | ---: | ---: |
| iforest | 0.9247 | 0.1227 | 0.2167 | 0.7264 |
| ocsvm | 0.9215 | 0.4640 | 0.6172 | 0.7786 |
| rf | 0.9987 | 0.9968 | 0.9977 | 0.9999 |
| ae | 0.9693 | 0.3390 | 0.5023 | 0.8338 |

Overlay ROC: `results/figures/cicids2017_roc_all.png`.
