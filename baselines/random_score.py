import random, csv
import pandas as pd

for file in ["dev_summarization.tsv",
             "dev_zh_en.tsv",
             "dev_en_de.tsv"]:
    df = pd.read_csv(file, sep = "\t", quoting=csv.QUOTE_NONE)
    df["random"] = [random.uniform(-1, 1) for i in range(len(df))]
    df["random"].to_csv(file+".seg.scores", header=False, index=False)


