#Evaluation

This directory contains the evaluation scripts of the shared task. In the dev phase, we will use the standard scipy implementation
of the Kendall correlation. Evaluation can be performed as follows:

```
python3 dev_evaluation.py metric_scores.txt golden_scores.txt output.txt
```

This will write the Kendall score to `output.txt`. Both input files `metric_scores.txt` and `golden_scores.txt` should
contain one corresponding float per input segment. The golden scores can be extracted from the train data in the `data`
folder.