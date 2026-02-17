# ml-research
A repository that hosts research projects.

Environment setup:
1. get the environment - unzip the ncs.zip into /ncs and run ncs/bin/activate; run python setup.py develop
2. get the code-llama-models and put them into $HOME/fsx/codellama/
3. run the workdir/ncs/ml-research-experiments/scripts/convert_llama_checkpoint.py to convert the model into a mono checkpoint (see example command in file)
4. get the preprocessed data from dataset_unity.7z
5. test finetuning and inference
