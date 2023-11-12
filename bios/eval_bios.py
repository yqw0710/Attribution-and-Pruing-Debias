import json
import os
import torch.distributed
import numpy as np
from collections import defaultdict, Counter
import argparse
import pickle

from datasets import Dataset
from transformers import AutoTokenizer
from pytorch_pretrained_bert.model_prune_head import BertForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn


with open("../data/bios/prof2ind.json") as json_file:
    mapping = json.load(json_file)


def rms_diff(tpr_diff):
    return np.sqrt(np.mean(tpr_diff**2))


def process_data(dataset):
    occ = []
    bio = []
    gend = []

    for elem in dataset:
        occ.append(elem["p"])
        bio.append(elem["text"][elem["start"] :])
        gend.append(elem["g"])

    prof_result = []
    for _, v in enumerate(occ):
        try:
            index = mapping[v]
        except KeyError:
            raise Exception("unknown label in occupation")
        prof_result.append(index)

    gend_result = []
    for _, v in enumerate(gend):
        if v == "m":
            gend_result.append(0)
        elif v == "f":
            gend_result.append(1)
        else:
            raise Exception("unknown label in gender")

    data_dict = {"label": prof_result, "bio": bio, "gend": gend_result}
    dataset = Dataset.from_dict(data_dict)

    return dataset


def eval_acc(model, dl, args,importance_set=None,prune_rate=0,prune_choice="predict"):
    model.eval()
    print("running eval")

    total_loss = 0
    total_correct = []

    for batch_idx, batch in enumerate(tqdm(dl)):
        input_ids = torch.transpose(torch.stack(batch["input_ids"]), 0, 1).to(args.device)
        attention_mask = torch.transpose(torch.stack(batch["attention_mask"]), 0, 1).to(args.device)
        labels = torch.tensor(batch["label"]).long().to(args.device)
        
        with torch.no_grad():
            if args.dataparallel:
                batch_loss,logits = model.module(input_ids=input_ids, get_which="res",attention_mask=attention_mask, labels=labels,att_head_mask=importance_set)
            else:
                batch_loss,logits = model(input_ids=input_ids, get_which="res",attention_mask=attention_mask, labels=labels, att_head_mask=importance_set)
        if args.dataparallel:
            total_loss += batch_loss.sum().item()
        else:
            total_loss += batch_loss.item()
        total_correct += (torch.argmax(logits, dim=1) == labels).tolist()

    acc = sum(total_correct) / len(total_correct)
    loss = total_loss / len(total_correct)
    log_dict = {
        "model":args.model_file,
        "prune_rate": prune_rate,
        "prune_choice": prune_choice,
        "batch_idx": batch_idx,
        "eval_acc": acc,
        "eval_loss": loss,
    }
    
    return log_dict


def eval_model(model, dl, args,importance_set=None,ratio=0,prune_choice="predict"):
    model.eval()
    print("running eval")

    total_loss = 0
    total_correct = []
    total_gender = []
    total_occ = []
    m_count = 0
    f_count = 0
    m_tot = 0
    f_tot = 0

    for batch_idx, batch in enumerate(tqdm(dl)):
        input_ids = torch.transpose(torch.stack(batch["input_ids"]), 0, 1).to(args.device)
        attention_mask = torch.transpose(torch.stack(batch["attention_mask"]), 0, 1).to(args.device)
        labels = torch.tensor(batch["label"]).long().to(args.device)
        gender = batch["gend"]

        with torch.no_grad():
            if args.dataparallel:
                batch_loss,logits = model.module(input_ids=input_ids, get_which="res",attention_mask=attention_mask, labels=labels,att_head_mask=importance_set)
            else:
                batch_loss,logits = model(input_ids=input_ids, get_which="res",attention_mask=attention_mask, labels=labels, att_head_mask=importance_set)
        if args.dataparallel:
            total_loss += batch_loss.sum().item()
        else:
            total_loss += batch_loss.item()
        total_correct += (torch.argmax(logits, dim=1) == labels).tolist()
        total_gender.append(gender)
        total_occ.append(labels.cpu().numpy())

    total_gender = [item for sublist in total_gender for item in sublist]
    total_occ = [item for sublist in total_occ for item in sublist]

    scores = defaultdict(Counter)
    prof_count_total = defaultdict(Counter)

    for (g, oc, l) in zip(total_gender, total_occ, total_correct):
        if g == 1:
            if l:
                f_count += 1
            f_tot += 1
        else:
            assert g == 0
            if l:
                m_count += 1
            m_tot += 1

        if l is True:
            scores[oc][g.item()] += 1
        prof_count_total[oc][g.item()] += 1

    assert m_count + f_count == sum(total_correct)
    acc = sum(total_correct) / len(total_correct)
    m_acc = m_count / m_tot
    f_acc = f_count / f_tot

    tprs = defaultdict(dict)
    tprs_change = dict()

    for profession, scores_dict in scores.items():
        good_m, good_f = scores_dict[0], scores_dict[1]
        prof_total_f = prof_count_total[profession][1]
        prof_total_m = prof_count_total[profession][0]
        tpr_m = (good_m) / prof_total_m
        tpr_f = (good_f) / prof_total_f

        tprs[profession][0] = tpr_m
        tprs[profession][1] = tpr_f
        tprs_change[profession] = tpr_m - tpr_f

    tpr_rms = rms_diff(np.array(list(tprs_change.values())))
    loss = total_loss / len(total_correct)
    log_dict = {
        "model":args.model_file,
        "ratio": ratio,
        "prune_choice": prune_choice,
        "batch_idx": batch_idx,
        "eval_acc": acc,
        "eval_acc_m": m_acc,
        "eval_acc_f": f_acc,
        "tpr": m_acc - f_acc,
        "tpr_rms": tpr_rms,
        "eval_loss": loss,
    }

    return log_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataparallel", default=False)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--cache_dir", default=None, type=str, help="cache directory")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_path", type=str, default="../data/bios/testbios.pkl")
    parser.add_argument("--output_dir", default='../model_save/bios/bert/original', type=str,
                        help="The output directory where the experimental results will be written.")
    parser.add_argument("--model_file", type=str,
                        default="../model_save/bios/bert/original/model.2.bin",
                        help="Full name or path or URL to trained model")
    args = parser.parse_args()

    file = open(args.test_path, "rb")
    data = pickle.load(file)
    file.close()
    test_dataset = process_data(data)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, cache_dir=args.cache_dir)
    model_state_dict = torch.load(args.model_file)
    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, state_dict=model_state_dict,
                                                          num_labels=28).to(args.device)

    if args.dataparallel:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    model = model.to(args.device)
    model.eval()

    if args.model_name_or_path.find("base") != -1:
        num_head, num_layer = 12, 12
    elif args.model_name_or_path.find("large") != -1:
        num_head, num_layer = 16, 24

    def preprocess_function(examples):
        # Tokenize the texts
        args = [examples["bio"]]
        result = tokenizer(
            *args,
            padding="max_length",
            max_length=128,
            truncation=True,
        )
        return result

    test_dataset = test_dataset.map(preprocess_function, batched=True, load_from_cache_file=True)
    eval_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.dataparallel:
        print(
            f"Trainable params: {sum(p.numel() for p in model.module.parameters() if p.requires_grad)}"
        )
        print(f"All params      : {sum(p.numel() for p in model.module.parameters())}")
    else:
        print(
            f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        print(f"All params      : {sum(p.numel() for p in model.parameters())}")

    head_bias_path = os.path.join(args.output_dir, "head_bias_importance_attr.json")
    head_predicte_path = os.path.join(args.output_dir, "head_predicte_importance_attr.json")
    head_bias_file = open(head_bias_path, "r", encoding="utf-8")
    head_bias_importance_attr = json.load(head_bias_file)
    head_predicte_file = open(head_predicte_path, "r", encoding="utf-8")
    head_predicte_importance_attr = json.load(head_predicte_file)

    predict_sort_index = list(np.argsort(-(np.array(head_predicte_importance_attr)))) # Sort from smallest to largest
    bias_sort_index = list(np.argsort(-(np.array(head_bias_importance_attr))))

    # trade-off pruning
    protected_ratio = [0,0.3,0.4,0.5,0.6,0.7,0.8]
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
        importance_set = importance_set.expand(-1, -1, 128, 128).to(args.device)


        log_dict = eval_model(model, eval_loader, args,importance_set,ratio=p_ratio,prune_choice="prune_ratio")
        print("Bias-in-Bios evaluation results:")
        print(f" - model file: {args.model_file}")
        print(f" - protected_ratio: {p_ratio}")
        print(f" - acc. (all): {log_dict['eval_acc']*100}")
        print(f" - acc. (m): {log_dict['eval_acc_m']*100}")
        print(f" - acc. (f): {log_dict['eval_acc_f']*100}")
        print(f" - tpr gap: {log_dict['tpr']*100}")
        print(f" - tpr rms: {log_dict['tpr_rms']}")

        with open(os.path.join(args.output_dir, "biasinbios-eval.json"), "a") as f_out:
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
            importance_set = importance_set.expand(-1, -1, args.max_seq_length, args.max_seq_length).to(args.device)

            log_dict = eval_model(model, eval_loader, args,importance_set,ratio=prune_rate,prune_choice=prune_choice)

            print("Bias-in-Bios evaluation results:")
            print(f" - model file: {args.model_file}")
            print(f" - prune_rate: {prune_rate}")
            print(f" - acc. (all): {log_dict['eval_acc']*100}")
            print(f" - acc. (m): {log_dict['eval_acc_m']*100}")
            print(f" - acc. (f): {log_dict['eval_acc_f']*100}")
            print(f" - tpr gap: {log_dict['tpr']*100}")
            print(f" - tpr rms: {log_dict['tpr_rms']}")

            with open(os.path.join(args.output_dir, "biasinbios-eval.json"), "a") as f_out:
                f_out.write(json.dumps(log_dict, indent=2, sort_keys=True))
                f_out.write('\n')


if __name__ == "__main__":
    main()
