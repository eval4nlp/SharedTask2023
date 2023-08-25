# Baselines

This directory contains baselines of the shared task. During the beginning of the dev phase, we plan to add further 
baselines. Current baselines are the following:

* `random_score.py` - A random baseline. It will produce random scores for every input summary/sentence. The current
                        implementation iterates over the dev-sets.
* `da_baselines.py` - A baseline metric that prompts LLMs to return DA scores. We enforce the output scores using 
                        the Microsoft Guidance Library (https://github.com/microsoft/guidance)
* `apply_da_baselines.py` - A simple script that applys the baseline llm metrics to the three dev sets. These need to be unzipped first.
* `example_codalab_dev.zip` - Example submission on the summarization dev task. Scores produced with the guanaco da baseline.

Further, the methods in `model_dict.py` should give first hints on how to load these models.
