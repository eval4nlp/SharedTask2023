import pandas as pd
import csv
import torch

from da_baselines import DirectAssessment
from model_dict import load_from_catalogue
from tqdm import tqdm

modelname = "TheBloke/WizardLM-13B-V1.1-GPTQ"
model_key = "wizard"
model, tokenizer, u_prompt, a_prompt = load_from_catalogue(modelname)
BPG = None

files = {
    "de":"../data/en_de/dev_en_de.tsv",
    "zh":"../data/zh_en/dev_zh_en.tsv",
    "sum":"../data/summarization/dev_summarization.tsv"
    }

for key, file in files.items():
    df = pd.read_csv(file, sep="\t", quoting=csv.QUOTE_NONE)
    scores = []

    if key =="sum":
        mt = False
    else:
        mt = True

    cnt = 0
    for s, h in tqdm(df[["SRC","HYP"]].values.tolist(), desc=key + " progress: "):

        # Ugly fix for memory leak; perhaps with the guidance module
        if BPG:
            del BPG
        BPG = DirectAssessment(model=model, tokenizer=tokenizer)

        print(cnt)
        print(torch.cuda.mem_get_info())
        score = BPG.prompt_model(
            gt=s,
            hyp=h,
            mt=mt,
            prompt_placeholder=u_prompt,
            response_placeholder=a_prompt,
            target_lang= "English" if key == "zh" else "German",
            source_lang= "Chinese" if key == "zh" else "English", 
            verbose=False
        )
        scores.append(score)
        cnt+=1

    df["baseline"] = scores
    df["baseline"].to_csv(key+model_key, header=False,index=False)


