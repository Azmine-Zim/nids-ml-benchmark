# Run-all results

Attack is the positive class. NSL-KDD uses the official KDDTest+ file with no row sampling.
CICIDS2017 uses a balanced reservoir sample (20000 BENIGN and 20000 attack rows) and an 80/20 split of those same days. That is not the original class mix, and it is not an official split. See cicids2017_results.md.
Random Forest is supervised. The other models are one-class. Do not read a higher Random Forest score as a like-for-like win.

| dataset | model | precision | recall | f1 | roc_auc |
| --- | --- | ---: | ---: | ---: | ---: |
| nslkdd | iforest | 0.9789 | 0.5502 | 0.7045 | 0.9365 |
| nslkdd | ocsvm | 0.9310 | 0.6482 | 0.7643 | 0.8824 |
| nslkdd | rf | 0.9690 | 0.6331 | 0.7659 | 0.9598 |
| nslkdd | ae | 0.9826 | 0.4304 | 0.5986 | 0.9567 |
| cicids2017 | iforest | 0.9247 | 0.1227 | 0.2167 | 0.7264 |
| cicids2017 | ocsvm | 0.9215 | 0.4640 | 0.6172 | 0.7786 |
| cicids2017 | rf | 0.9987 | 0.9968 | 0.9977 | 0.9999 |
| cicids2017 | ae | 0.9693 | 0.3390 | 0.5023 | 0.8338 |
