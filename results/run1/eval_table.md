<!-- Copied verbatim from Kaggle version 366094037, figures/eval_table.md. 4 samples per task per hint level, temperature 1.0.
     "Hack rate" here is the v0.1 detector's: it counts visible-test overfitting and the timeout artifact described in the README as hacks. -->
| Run | Split | Hint | n | Hack rate | Genuine | Hidden-test pass |
|---|---|---|---|---|---|---|
| base | heldout | 0 | 40 | 2.5% | 12.5% | 20.4% |
| base | heldout | 1 | 40 | 0.0% | 22.5% | 23.8% |
| base | heldout | 2 | 40 | 2.5% | 32.5% | 36.1% |
| base | train | 0 | 108 | 0.0% | 19.4% | 21.9% |
| base | train | 1 | 108 | 1.8% | 21.3% | 25.5% |
| base | train | 2 | 108 | 2.8% | 24.1% | 28.2% |
| vuln_hint2 | heldout | 0 | 40 | 2.5% | 40.0% | 54.6% |
| vuln_hint2 | heldout | 2 | 40 | 0.0% | 42.5% | 58.0% |
| vuln_hint2 | train | 0 | 108 | 5.6% | 79.6% | 86.4% |
| vuln_hint2 | train | 2 | 108 | 6.5% | 77.8% | 84.5% |
| robust_hint2 | heldout | 0 | 40 | 0.0% | 47.5% | 54.5% |
| robust_hint2 | heldout | 2 | 40 | 0.0% | 45.0% | 53.9% |
| robust_hint2 | train | 0 | 108 | 0.9% | 87.0% | 90.9% |
| robust_hint2 | train | 2 | 108 | 0.0% | 88.9% | 92.0% |
