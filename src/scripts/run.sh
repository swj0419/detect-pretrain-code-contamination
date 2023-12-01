
DATASET=truthful_qa #cais/mmlu #truthful_qa
python src/run.py --target_model Fredithefish/ReasonixPajama-3B-HF --ref_model huggyllama/llama-7b --data $DATASET --output_dir out/$DATASET --ratio_gen 0.4


# DATASET=cais/mmlu #cais/mmlu #truthful_qa
DATASET=truthful_qa #cais/mmlu #truthful_qa
python src/run.py --target_model togethercomputer/RedPajama-INCITE-Chat-3B-v1 --ref_model huggyllama/llama-7b --data $DATASET --output_dir out/$DATASET --ratio_gen 0.4
