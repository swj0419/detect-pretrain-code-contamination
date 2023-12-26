import logging
logging.basicConfig(level='ERROR')
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import matplotlib
import random
from ipdb import set_trace as bp
import time

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# plot data 
def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc


def do_plot(prediction, answers, sweep_fn=sweep, metric='auc', legend="", output_dir=None):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """
    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    low = tpr[np.where(fpr<.05)[0][-1]]
    # bp()
    print('Attack %s   AUC %.4f, Accuracy %.4f, TPR@5%%FPR of %.4f\n'%(legend, auc,acc, low))

    metric_text = ''
    if metric == 'auc':
        metric_text = 'auc=%.3f'%auc
    elif metric == 'acc':
        metric_text = 'acc=%.3f'%acc

    plt.plot(fpr, tpr, label=legend+metric_text)
    return legend, auc,acc, low


def fig_fpr_tpr(all_output, output_dir):
    print("output_dir", output_dir)
    answers = []
    metric2predictions = defaultdict(list)
    for ex in all_output:
        answers.append(ex["label"])
        for metric in ex["pred"].keys():
            if ("raw" in metric) and ("clf" not in metric):
                continue
            metric2predictions[metric].append(ex["pred"][metric])
    
    plt.figure(figsize=(4,3))
    with open(f"{output_dir}/auc.txt", "w") as f:
        for metric, predictions in metric2predictions.items():
            legend, auc, acc, low = do_plot(predictions, answers, legend=metric, metric='auc', output_dir=output_dir)
            f.write('%s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f\n'%(legend, auc, acc, low))

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5,1)
    plt.ylim(1e-5,1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.legend(fontsize=8)
    plt.savefig(f"{output_dir}/auc.png")


def load_jsonl(input_path):
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in tqdm(f)]
    random.seed(0)
    random.shuffle(data)
    return data

def dump_jsonl(data, path):
    with open(path, 'w') as f:
        for line in tqdm(data):
            f.write(json.dumps(line) + "\n")

def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in tqdm(f)]

def convert_huggingface_data_to_list_dic(dataset):
    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]
        all_data.append(ex)
    return all_data


def process_truthful_qa(data):
    new_data = []
    for ex in data:
        new_ex = {}
        label = ex["mc2_targets"]["labels"].index(1)
        output = ex["mc2_targets"]["choices"][label]
        # We change to mc2 instead of mc1 as it's those that open llm lead uses. (check about)
        new_ex["output"] = output
        new_ex["input"] = ex["question"] + " " + output
        new_data.append(new_ex)
    return new_data



def process_mmlu(data):
    new_data = []
    for ex in data:
        new_ex = {}
        label = ex["choices"][ex["answer"]]
        output = label
        new_ex["output"] = output
        new_ex["input"] = ex["question"] + " " + output
        new_data.append(new_ex)
    return new_data


def process_arc(data):
    new_data = []
    choice2label = {"A": 0, "B": 1, "C": 2, "D": 3}
    for ex in data:
        new_ex = {}
        # bp()
        # print(ex["answerKey"])
        if ex["answerKey"] not in choice2label:
            continue
        label = choice2label[ex["answerKey"]]
        output = ex["choices"]["text"][label]
        new_ex["output"] = output
        new_ex["input"] = ex["question"] + " " + output
        new_data.append(new_ex)
    return new_data

def process_gsm8k(data):
    new_data = []
    for ex in data:
        new_ex = {}
        output = ex["answer"].split('####')[0].strip()
        new_ex["output"] = output
        new_ex["input"] = ex["question"] + " " + output
        new_data.append(new_ex)
    return new_data

def process_winogrande(data):
    '''
    new_data = []
    for ex in data:
        new_ex = {}
        label = int(ex["answer"])
        output = ex[f"option{label}"]
        new_ex["output"] = output
        new_ex["input"] = ex["sentence"] + " " + output
        new_data.append(new_ex)
    return new_data
    '''
    new_data = []
    for doc in data:
        new_doc = {}

        # Convert the answer to a numeric index
        answer_to_num = {"1": 0, "2": 1}
        label_idx = answer_to_num[doc["answer"]]

        # Generate options and select the correct one based on label_idx
        options = [doc["option1"], doc["option2"]]
        output = options[label_idx]

        # Build the new sentence by inserting the selected option
        idx = doc["sentence"].index("_")
        input_sentence = doc["sentence"][:idx] + output + doc["sentence"][idx+1:]

        # Assigning the processed values to the new_doc
        new_doc["output"] = output
        new_doc["input"] = input_sentence

        # Append the processed document to new_data
        new_data.append(new_doc)

    return new_data
# I'm not sure if that's the correct format for winogrande given how the dataset works.

def process_hellaswag(data):
    new_data = []
    for ex in data:
        new_ex = {}
        label = int(ex["label"]) # For some reason label is in str and not int?
        output = ex["endings"][label]
        new_ex["output"] = output
        new_ex["input"] = ex["ctx"] + " " + output
        new_data.append(new_ex)
    return new_data
