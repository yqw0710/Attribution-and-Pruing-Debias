import json
import numpy as np
import argparse
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from pytorch_pretrained_bert.model_prune_head import BertForSequenceClassification


def eval_model(model, eval_loader, args,importance_set=None,ratio=0,prune_choice="prune_predict"):
    nn_count, fn_count, tn_count, tn2_count, denom = 0, 0, 0, 0, 0

    for batch_idx, batch in enumerate(tqdm(eval_loader)):
        input_ids = torch.transpose(torch.stack(batch["input_ids"]), 0, 1).to(args.device)
        attention_mask = torch.transpose(torch.stack(batch["attention_mask"]), 0, 1).to(args.device)
        labels = torch.tensor(batch["label"]).long().to(args.device)

        if "token_type_ids" in batch:
            token_type_ids = torch.transpose(
                torch.stack(batch["token_type_ids"]), 0, 1
            ).to(args.device)
            loss, logits = model(
                input_ids, "res", token_type_ids, attention_mask, labels,att_head_mask=importance_set
            )
        else:
            loss, logits = model(
                input_ids, "res", None, attention_mask, labels, att_head_mask=importance_set
            )

        res = torch.softmax(logits, axis=1)
        preds = res.argmax(1)
        denom += len(preds)

        nn_count += (torch.sum(res, axis=0)[1]).item()
        if len(torch.bincount(preds)) >= 2:
            fn_count += (torch.bincount(preds)[1]).item()
        tn_count += torch.sum(res[:, 1] > args.tau).item()
        tn2_count += torch.sum(res[:, 1] > args.tau2).item()

    log_dict = {
        "model":args.model_file,
        "ratio": ratio,
        "prune_choice": prune_choice,
        "total net neutral":nn_count / denom,
        "total fraction neutral": fn_count / denom,
        "total tau 0.5 neutral": tn_count / denom,
        "total tau 0.7 neutral": tn2_count / denom
    }
    
    return log_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=30)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--tau2", type=float, default=0.7)
    parser.add_argument("--cache_dir", default=None, type=str, help="cache directory")
    parser.add_argument("--output_dir", default='../model_save/mnli/bert/original', type=str,
                        help="The output directory where the experimental results will be written.")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--model_file", type=str, default="../model_save/mnli/bert/original/model.2.bin",
                        help="Full name or path or URL to trained NLI model")
    parser.add_argument("--eval_data_path", type=str, default="../data/mnli/bias-nli.csv",
                        help="Filepath to evaluation templates")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)  #
    model_state_dict = torch.load(args.model_file)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name_or_path, state_dict=model_state_dict,num_labels=3
    ).to(args.device)
    model.eval()

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples["premise"],)
            if "hypothesis" == None
            else (examples["premise"], examples["hypothesis"])
        )
        result = tokenizer(*args, padding="max_length", max_length=30, truncation=True)
        return result

    eval_dataset = load_dataset("csv", data_files=args.eval_data_path, split="train[:]", cache_dir=args.cache_dir, )
    eval_dataset = eval_dataset.shuffle(seed=42)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True, load_from_cache_file=True)

    eval_loader = DataLoader(dataset=eval_dataset, batch_size=args.batch_size, shuffle=True)

    if args.model_name_or_path.find("base") != -1:
        num_head, num_layer = 12, 12
    elif args.model_name_or_path.find("large") != -1:
        num_head, num_layer = 16, 24

    head_bias_path = os.path.join(args.output_dir, "head_bias_importance_attr.json")
    head_predicte_path = os.path.join(args.output_dir, "head_predicte_importance_attr.json")
    head_bias_file = open(head_bias_path, "r", encoding="utf-8")
    head_bias_importance_attr = json.load(head_bias_file)
    head_predicte_file = open(head_predicte_path, "r", encoding="utf-8")
    head_predicte_importance_attr = json.load(head_predicte_file)

    predict_sort_index = list(np.argsort(-(np.array(head_predicte_importance_attr)))) # Sort from smallest to largest
    bias_sort_index =list(np.argsort(-(np.array(head_bias_importance_attr))))


    # trade-off pruning
    protected_ratio = [0.8,0.7,0.6,0.5,0.4,0.3]
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
        importance_set = importance_set.expand(-1, -1, args.max_seq_length, args.max_seq_length).to(args.device)

        log_dict = eval_model(model, eval_loader, args,importance_set,ratio=p_ratio,prune_choice="prune_ratio")

        with open(os.path.join(args.output_dir, "mnli-eval.json"), "a") as f_out:
            f_out.write(json.dumps(log_dict, indent=2, sort_keys=True))
            f_out.write('\n')


    # Pruning based on bias importance and predicted unimportance only
    for prune_choice in (["prune_predict","prune_bias"]):
        for prune_i in range(0, 11, 1): # Percentage of traversal pruning
            prune_rate = prune_i / 10
            if args.prune_choice=="prune_predict":
                head_importance=np.argsort((np.array(head_predicte_importance_attr)))
            elif args.prune_choice=="prune_bias":
                head_importance=np.argsort(-(np.array(head_bias_importance_attr)))

            importance_set = [1] * num_head * num_layer
            num = int(num_head * num_layer*prune_rate)
            for i in range(num):
                importance_set[head_importance[i]]=0

            importance_set=np.array(importance_set).reshape(num_layer,num_head)
            importance_set = torch.tensor(importance_set)
            importance_set = importance_set.view(*importance_set.shape, 1, 1)
            importance_set = importance_set.expand(-1, -1, args.max_sequence_length, args.max_sequence_length).to(args.device)
      
            log_dict = eval_model(model, eval_loader, args,importance_set,ratio=prune_rate,prune_choice=prune_choice)

            with open(os.path.join(args.output_dir, "mnli-eval.json"), "a") as f_out:
                f_out.write(json.dumps(log_dict, indent=2, sort_keys=True))
                f_out.write('\n')


if __name__ == "__main__":
    main()
