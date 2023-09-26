# SharedTask2023

This is the github repository for the 2023 Eval4NLP shared task: "Prompting Large Language Models as Evaluation Metrics". For more information visit: https://eval4nlp.github.io/2023/shared-task.html.

You can execute the following commands to be able to run the baselines. Note that due to the model sizes, some baseline settings can be resource heavy. (Tested on an Ubuntu 22.04 cluster with SLURM)

```
conda create --name Eval4NLP23 python=3.10
conda activate Eval4NLP23
#conda install pip     # this might be necessary in some cases
pip install -r requirements
```
The `data` folder contains train, dev and test sets. The `baseline` folder contains baseline scripts and an example submission for the Codabench test phase of our shared task.

# Test Phase
The test phase of the share task is being conducted on Codabench: https://www.codabench.org/competitions/1359/#/pages-tab

Our CodaLab competition for the Dev-Phase can be found here: https://codalab.lisn.upsaclay.fr/competitions/15072
