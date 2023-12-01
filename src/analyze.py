import json
import statistics 

def load_jsonl(path):
    with open(path) as f:
        data = [json.loads(line) for line in f]
    return data

def analyze_data(data):
    all_rmia = []
    all_large_1 = []
    for ex in data:
        # Min_20.0% Prob
        score = ex["pred"]["minkprob_w/_ref"]  # minkprob_w/_ref
        all_rmia.append(score)
        if score < 0.1:
            all_large_1.append(score)
    print("result < 0.1, %: ", len(all_large_1)/len(all_rmia))
    # print(f"RMIA mean: {statistics.mean(all_rmia)}")
    # print(f"RMIA std: {statistics.stdev(all_rmia)}")
    # print(f"RMIA min: {min(all_rmia)}")
    # print(f"RMIA max: {max(all_rmia)}")
    # # 25% percentile
    # print(f"RMIA 25%: {statistics.quantiles(all_rmia)[0]}")
    # # 50% percentile
    # print(f"RMIA 50%: {statistics.quantiles(all_rmia)[1]}")
    # # 75% percentile
    # print(f"RMIA 75%: {statistics.quantiles(all_rmia)[2]}")


      

if __name__ == "__main__":
    print("contaminated model")
    task = "ai2_arc" # ai2_arc cais/mmlu truthful_qa
    # /fsx-onellm/swj0419/attack/test_contamination/detect-pretrain-code/out/ai2_arc/Fredithefish/ReasonixPajama-3B-HF_togethercomputer/RedPajama-INCITE-Chat-3B-v1/input/all_output.jsonl
    path = f"/fsx-onellm/swj0419/attack/test_contamination/detect-pretrain-code/out/{task}/Fredithefish/ReasonixPajama-3B-HF_huggyllama/llama-7b/input/all_output.jsonl"
    data = load_jsonl(path)
    analyze_data(data)

    print("raw model")
    path = f"/fsx-onellm/swj0419/attack/test_contamination/detect-pretrain-code/out/{task}/togethercomputer/RedPajama-INCITE-Chat-3B-v1_huggyllama/llama-7b/input/all_output.jsonl"
    data = load_jsonl(path)
    analyze_data(data)

