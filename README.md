# Detect-Pretrain-Code-Contamination

This repository contains scripts for detecting pretraining code contamination in datasets.

## Datasets
You can specify the dataset for analysis. Example datasets include `truthful_qa` and `cais/mmlu`.

## Usage
Run the script with the desired models and dataset. Below are two examples of how to use the script with different models and the `truthful_qa` dataset.

### Example 1:
```bash
DATASET=truthful_qa
python src/run.py --target_model Fredithefish/ReasonixPajama-3B-HF --ref_model huggyllama/llama-7b --data $DATASET --output_dir out/$DATASET --ratio_gen 0.4
```

The output of the script provides a metric for dataset contamination. If #the result < 0.1# with a percentage greater than 0.85, it is highly likely that the dataset has been trained.
