import json
import logging
import os
import numpy as np
import torch
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer
from pytorch_pretrained_bert.model_prune_head import BertForSequenceClassification

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_dataset_bias_sts(data_path):
    file1 = open(data_path, 'r', encoding="utf-8")
    lines = file1.readlines()
    sentence_pairs = []
    for line in lines:
        entries = line.split("\t")
        if len(entries) > 1:  # ignore empty lines
            pair = [entries[0].replace('\n', ''), entries[1].replace('\n', ''), entries[2].replace('\n', ''), entries[3].replace('\n', '')]
            sentence_pairs.append(pair)

    return sentence_pairs


def evaluate_bias_sts(args,model, tokenizer, data_path, device,att_head_mask=None,protected_ratio=0, prune_choice="prune_ratio"):
    # get bias evaluation dataset
    pairs = get_dataset_bias_sts(data_path)
    number_pairs = len(pairs)

    # evaluation metrics
    highest_male = -1000.0  # highest similarity score for 'male' sentence pair
    lowest_male = 1000.0  # lowest similarity score for 'male' sentence pair
    highest_female = -1000.0  # highest similarity score for 'female' sentence pair
    lowest_female = 1000.0  # lowest similarity score for 'female' sentence pair
    highest_diff = 0.0  # highest similarity difference between a 'male' and 'female' sentence pair
    lowest_diff = 1000.0  # lowest similarity difference between a 'male' and 'female' sentence pair
    difference_abs_avg = 0.0  # absolute average of all differences between 'male' and 'female' sentence pairs: abs(male - female)
    difference_avg = 0.0  # average of all differences between 'male' and 'female' sentence pairs: male - female
    male_avg = 0.0  # average similarity score for 'male' sentence pairs
    female_avg = 0.0  # average similarity score for 'female' sentence pairs
    threshold_01 = 0  # how often difference between 'male' and 'female' sentence_pairs > 0.1
    threshold_03 = 0  # how often difference between 'male' and 'female' sentence_pairs > 0.3
    threshold_05 = 0  # how often difference between 'male' and 'female' sentence_pairs > 0.5
    threshold_07 = 0  # how often difference between 'male' and 'female' sentence_pairs > 0.7

    max_length = 128
    # count the occurences to calculate the results
    for p in tqdm(pairs):
        inputs_male= tokenizer(p[0], p[1], padding="max_length", max_length=max_length,
                           return_tensors='pt')
        inputs_male = {k: v.to(device) for k, v in inputs_male.items()}
        sim_male = model(
            inputs_male["input_ids"], "log", inputs_male["token_type_ids"], inputs_male["attention_mask"],att_head_mask=att_head_mask
        )
        sim_male=float(sim_male[0][0])
        inputs_female = tokenizer(p[2], p[3], padding="max_length", max_length=max_length,
                                return_tensors='pt')
        inputs_female = {k: v.to(device) for k, v in inputs_female.items()}
        sim_female = model(
            inputs_female["input_ids"], "log", inputs_female["token_type_ids"], inputs_female["attention_mask"],att_head_mask=att_head_mask
        )
        sim_female = float(sim_female[0][0])

        # adjust measurements
        difference_abs = abs(sim_male - sim_female)
        difference = sim_male - sim_female
        if sim_male < lowest_male:
            lowest_male = sim_male
        if sim_female < lowest_female:
            lowest_female = sim_female
        if sim_male > highest_male:
            highest_male = sim_male
        if sim_female > highest_female:
            highest_female = sim_female
        if difference_abs < lowest_diff:
            lowest_diff = difference_abs
        if difference_abs > highest_diff:
            highest_diff = difference_abs
        male_avg += sim_male
        female_avg += sim_female
        difference_abs_avg += difference_abs
        difference_avg += difference
        if difference_abs > 0.1:
            threshold_01 += 1
        if difference_abs > 0.3:
            threshold_03 += 1
        if difference_abs > 0.5:
            threshold_05 += 1
        if difference_abs > 0.7:
            threshold_07 += 1

    # get final results
    difference_abs_avg = difference_abs_avg / number_pairs
    difference_avg = difference_avg / number_pairs
    male_avg = male_avg / number_pairs
    female_avg = female_avg / number_pairs
    threshold_01 = threshold_01 / number_pairs
    threshold_03 = threshold_03 / number_pairs
    threshold_05 = threshold_05 / number_pairs
    threshold_07 = threshold_07 / number_pairs

    # print results
    logger.info("Male avg: " + str(male_avg))
    logger.info("Female avg: " + str(female_avg))
    logger.info("Difference absolut avg: " + str(difference_abs_avg))
    logger.info("Difference avg (male - female): " + str(difference_avg))
    logger.info("Threshold 01: " + str(threshold_01))
    logger.info("Threshold 03: " + str(threshold_03))
    logger.info("Threshold 05: " + str(threshold_05))
    logger.info("Threshold 07: " + str(threshold_07))

    result_json = {
        "model": args.model_file,
        "protected_ratio": protected_ratio,
        "prune_choice": prune_choice,
        "male avg: ":male_avg,
        "female avg:":female_avg,
        "Threshold 01: ":threshold_01,
        "Threshold 03: ":threshold_03,
        "Threshold 05: ":threshold_05,
        "Threshold 07: ":threshold_07,
        "diff_abs_avg": difference_abs_avg,
        "diff_avg": difference_avg
    }

    return result_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str,default="bert-base-uncased", help="Full name or path or URL to tokenizer")
    parser.add_argument("--model_file", type=str,default="../model-save/sts-b/bert/original/model.2.bin", help="Full name or path or URL to trained NLI model")
    parser.add_argument("--eval_filepath", type=str, default="../data/sts-b/bias_evaluation_STS-B.tsv", help="Filepath to evaluation templates")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum length allowed for input sentences. If longer, truncate.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",default='../model_save/sts-b/bert/original',type=str,
                        help="The output directory where the experimental results will be written.")

    args = parser.parse_args()

    if args.bert_model.find("base") != -1:
        num_head, num_layer = 12, 12
    elif args.bert_model.find("large") != -1:
        num_head, num_layer = 16, 24

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.cuda.empty_cache()

    # Load the trained NLI/STS-B model
    model_state_dict = torch.load(args.model_file)
    model = BertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict,num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,do_lower_case=True)
    model.to(device)
    model.eval()

    softmax = torch.nn.Softmax(dim=-1)

    head_bias_path = os.path.join(args.output_dir, "head_bias_importance_attr.json")
    head_predicte_path = os.path.join(args.output_dir, "head_predicte_importance_attr.json")
    head_bias_file = open(head_bias_path, "r", encoding="utf-8")
    head_bias_importance_attr = json.load(head_bias_file)
    head_predicte_file = open(head_predicte_path, "r", encoding="utf-8")
    head_predicte_importance_attr = json.load(head_predicte_file)

    predict_sort_index=list(np.argsort(-(np.array(head_predicte_importance_attr)))) # Sort from smallest to largest
    bias_sort_index=list(np.argsort(-(np.array(head_bias_importance_attr))))

    # trade-off pruning
    protected_ratio = [0.3,0.4,0.5,0.6,0.7,0.8]
    for p_ratio in protected_ratio: # Percentage of traversal pruning
        final_importance = [-1] * 14
        pi, bi, final_ci, num_pi = -1, 0, 0, 0
        while final_ci < 14:
            if pi <= -13:
                pi = -13
            if bi >= 27:
                bi = 27
            while predict_sort_index[pi] in final_importance:
                pi += -1
            while bias_sort_index[bi] in final_importance:
                bi += 1
            if num_pi < (final_ci + 1) * p_ratio: # Need for protection
                final_importance[final_ci] = predict_sort_index[pi]
                final_ci += 1
                pi += -1
                num_pi += 1
            elif bias_sort_index.index(bias_sort_index[bi]) < predict_sort_index.index(bias_sort_index[bi]):
                final_importance[final_ci] = bias_sort_index[bi]
                final_ci += 1
                bi += 1
            else:
                bi += 1
        print(final_importance)

        importance_set = [1] * num_head * num_layer
        for i in range(14):
            importance_set[final_importance[i]] = 0
        importance_set = np.array(importance_set).reshape(num_layer, num_head)
        importance_set = torch.tensor(importance_set)
        importance_set = importance_set.view(*importance_set.shape, 1, 1)
        importance_set = importance_set.expand(-1, -1, args.max_len, args.max_len).to(device)

        result_json = evaluate_bias_sts(args,model,tokenizer,args.eval_filepath,device,att_head_mask=importance_set,protected_ratio=p_ratio,prune_choice="prune_ratio")
        
        with open(os.path.join(args.output_dir, "sts-b-eval.json"), "a") as f_out:
            f_out.write(json.dumps(result_json, indent=2, sort_keys=True))
            f_out.write('\n')


    # Pruning based on bias importance and predicted unimportance only
    for prune_choice in (["prune_predict","prune_bias"]):
        prune_result = []
        for prune_i in range(0, 11, 1): # Percentage of traversal pruning
            prune_rate = prune_i / 10
            if prune_choice == "prune_predict":
                head_importance = np.argsort((np.array(head_predicte_importance_attr)))
            elif prune_choice == "prune_bias":
                head_importance = np.argsort(-(np.array(head_bias_importance_attr)))

            importance_set = [1] * num_head * num_layer
            num = int(num_head * num_layer * prune_rate)
            for i in range(num):
                importance_set[head_importance[i]] = 0

            importance_set = np.array(importance_set).reshape(num_layer, num_head)
            importance_set = torch.tensor(importance_set)
            importance_set = importance_set.view(*importance_set.shape, 1, 1)
            importance_set = importance_set.expand(
                -1, -1, args.max_sequence_length, args.max_sequence_length
            ).to(args.device)

            result_json = evaluate_bias_sts(args,model,tokenizer,args.eval_filepath,device,att_head_mask=importance_set,prune_rate=prune_rate,prune_choice=prune_choice)
        
            with open(os.path.join(args.output_dir, "sts-b-eval.json"), "a") as f_out:
                f_out.write(json.dumps(result_json, indent=2, sort_keys=True))
                f_out.write('\n')
