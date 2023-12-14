import logging
logging.basicConfig(level='ERROR')
import numpy as np
from pathlib import Path
import openai
import torch
import zlib
import statistics
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import math
import numpy as np
from datasets import load_dataset
from options import Options
from ipdb import set_trace as bp
from eval import *
from utils import evaluate_model
from analyze import analyze_data



def load_model(name1, name2):
    if "davinci" in name1:
        model1 = None
        tokenizer1 = None
    else:
        model1 = AutoModelForCausalLM.from_pretrained(name1, return_dict=True, device_map='auto')
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained(name1)

    if "davinci" in name2:
        model2 = None
        tokenizer2 = None
    else:
        model2 = AutoModelForCausalLM.from_pretrained(name2, return_dict=True, device_map='auto')
        model2.eval()
        tokenizer2 = AutoTokenizer.from_pretrained(name2)
    tokenizer1.pad_token = tokenizer1.eos_token 
    tokenizer2.pad_token = tokenizer2.eos_token
    return model1, model2, tokenizer1, tokenizer2

def calculatePerplexity_gpt3(prompt, modelname):
    prompt = prompt.replace('\x00','')
    responses = None
    # Put your API key here
    openai.api_key = "YOUR_API_KEY" # YOUR_API_KEY
    while responses is None:
        try:
            responses = openai.Completion.create(
                        engine=modelname, 
                        prompt=prompt,
                        max_tokens=0,
                        temperature=1.0,
                        logprobs=5,
                        echo=True)
        except openai.error.InvalidRequestError:
            print("too long for openai API")
    data = responses["choices"][0]["logprobs"]
    all_prob = [d for d in data["token_logprobs"] if d is not None]
    p1 = np.exp(-np.mean(all_prob))
    return p1, all_prob, np.mean(all_prob)

     
def calculatePerplexity(sentence, model, tokenizer, gpu):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(gpu)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    
    '''
    extract logits:
    '''
    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    # probabilities = torch.nn.functional.softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    return torch.exp(loss).item(), all_prob, loss.item()

def sample_generation(sentence, model, tokenizer, args):
    half_sentence_index = math.ceil(len(sentence.split())*args['prefix_length'])

    if half_sentence_index > 0:
        prefix = " ".join(sentence.split()[:half_sentence_index])
    else:
        prefix = '<|startoftext|> '
    
    input_ids = torch.tensor(tokenizer.encode(prefix)).unsqueeze(0)
    input_ids = input_ids.to(model.device)

    output = model.generate(input_ids, max_new_tokens=len(sentence.split())-half_sentence_index, min_new_tokens=1, num_return_sequences=args['num_z'], pad_token_id=tokenizer.eos_token_id, **args['generate_args'])
    # print(output)
    complete_generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)
 
    return complete_generated_text
    

def evaluate_RMIA(text, target_loss, ref_loss, model1, model2, tokenizer1, tokenizer2, args):
    # use sample generation
    cur_args = {'prefix_length': args.ratio_gen, 'num_z': 100, 'generate_args': {'do_sample': True}}
    neighbors = sample_generation(text, model2, tokenizer2, cur_args)
    # bp()
    neighbors_dl = DataLoader(neighbors, batch_size=32, shuffle=False)
    target_losses_z = evaluate_model(model1, tokenizer1, neighbors_dl)
    # bp()
    # result = (target_loss / statistics.mean(target_losses_z))qq
    # bp()
    result = torch.count_nonzero(target_losses_z < target_loss).item() / len(target_losses_z)
    return result


def inference(model1, model2, tokenizer1, tokenizer2, text, ex, modelname1, modelname2, args):
    pred = {}

    if "davinci" in modelname1:
        p1, all_prob, p1_likelihood = calculatePerplexity_gpt3(text, modelname1) 
        p_lower, _, p_lower_likelihood = calculatePerplexity_gpt3(text.lower(), modelname1)
    else:
        p1, all_prob, p1_likelihood = calculatePerplexity(text, model1, tokenizer1, gpu=model1.device)
        p_lower, _, p_lower_likelihood = calculatePerplexity(text.lower(), model1, tokenizer1, gpu=model1.device)

    if "davinci" in modelname2:
        p_ref, all_prob_ref, p_ref_likelihood = calculatePerplexity_gpt3(text, modelname2)
    else:
        p_ref, all_prob_ref, p_ref_likelihood = calculatePerplexity(text, model2, tokenizer2, gpu=model2.device)
   
    # RMIA:
    rmia_result = evaluate_RMIA(text, p1_likelihood, p_ref_likelihood, model1, model2, tokenizer1, tokenizer2, args)
    pred["minkprob_w/_ref"] = rmia_result
    # bp()

   # ppl
    pred["ppl"] = p1
    # Ratio of log ppl of large and small models
    pred["ppl/Ref_ppl (calibrate PPL to the reference model)"] = p1_likelihood-p_ref_likelihood


    # Ratio of log ppl of lower-case and normal-case
    pred["ppl/lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item()
    # Ratio of log ppl of large and zlib
    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
    pred["ppl/zlib"] = np.log(p1)/zlib_entropy

    ex["pred"] = pred
    return ex

def evaluate_data(test_data, model1, model2, tokenizer1, tokenizer2, col_name, modelname1, modelname2):
    print(f"all data size: {len(test_data)}")
    all_output = []
    random.seed(0)
    random.shuffle(test_data)
    test_data = test_data[:100]
    for ex in tqdm(test_data): 
        text = ex[col_name]
        new_ex = inference(model1, model2, tokenizer1, tokenizer2, text, ex, modelname1, modelname2, args)
        all_output.append(new_ex)
    return all_output


if __name__ == '__main__':
    args = Options()
    args = args.parser.parse_args()
    args.output_dir = f"{args.output_dir}/{args.target_model}_{args.ref_model}/{args.key_name}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load model and data
    model1, model2, tokenizer1, tokenizer2 = load_model(args.target_model, args.ref_model)
    if "jsonl" in args.data:
        data = load_jsonl(f"{args.data}")
    elif args.data == "truthful_qa": # load data from huggingface
        # bp()
        dataset = load_dataset(args.data, "multiple_choice", split="validation")
        data = convert_huggingface_data_to_list_dic(dataset)
        data = process_truthful_qa(data)
    elif args.data == "cais/mmlu":
        dataset = load_dataset(args.data, "all", split="test")
        data = convert_huggingface_data_to_list_dic(dataset)
        data = process_mmlu(data)
    elif args.data == "ai2_arc":
        dataset = load_dataset(args.data, "ARC-Challenge", split="test")
        data = convert_huggingface_data_to_list_dic(dataset)
        data = process_arc(data)
    elif args.data == "gsm8k":
        dataset = load_dataset(args.data, "main", split="test")
        data = convert_huggingface_data_to_list_dic(dataset)
        data = process_gsm8k(data)



    all_output = evaluate_data(data, model1, model2, tokenizer1, tokenizer2, args.key_name, args.target_model, args.ref_model)
    dump_jsonl(all_output, f"{args.output_dir}/all_output.jsonl")
    analyze_data(all_output)
    # fig_fpr_tpr(all_output, args.output_dir)

