# Train and Dev Data

The zip files in this folder contain the train and dev sentences/documents of the Eval4NLP23 shared task. We re-use existing
datasets with a random train/dev split each. As we evaluate in a reference-free setting, we only provide the source and 
no references. In specific we use:

1. The MQM *en-de* and *zh-en* language pairs of the WMT 22 metrics shared task,a work by  Freitag et al., Results of WMT22 Metrics Shared Task: Stop Using BLEU -- Neural Metrics Are Better and More Robust. In: WMT22
2. The average aspect score of SummEval, a work by Fabbri et al., SummEval: Re-evaluating Summarization Evaluation. In: Transactions of the Association for Computational Linguistics

The dev sets can be evaluated on our CodaLab leaderboard. We do not provide their scores here, to make the DEV phase more 
interesting. Theoretically, you could match them back to their original dataset. As the dev phase has no influence on the
shared task results, please refrain from adding the ground truth to the leaderboards.

The licenses of the respective datasets are placed inside the zip files. 

The tsv files can be loaded with pandas:

```python
import pandas as pd
import csv

df_source = pd.read_csv("<filename>", sep="\t", quoting=csv.QUOTE_NONE)
```