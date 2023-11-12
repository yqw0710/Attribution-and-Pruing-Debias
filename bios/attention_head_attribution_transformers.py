import json
import os
import pickle
import torch.distributed
import argparse
import torch
import torch.nn as nn
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import softmax
from transformers import AutoTokenizer,AutoModelForSequenceClassification


with open("../data/bios/prof2ind.json") as json_file:
    mapping = json.load(json_file)


def scaled_input(emb, batch_size, num_batch, baseline=None, start_i=None, end_i=None):
    # shape of emb: (num_head, seq_len, seq_len)
    if baseline is None:
        baseline = torch.zeros_like(emb)

    num_points = batch_size * num_batch
    scale = 1.0 / num_points
    if start_i is None:
        step = (emb.unsqueeze(0) - baseline.unsqueeze(0)) * scale
        res = torch.cat([torch.add(baseline.unsqueeze(0), step*i) for i in range(num_points)], dim=0)
        return res, step[0]
    else:
        step = (emb - baseline) * scale
        start_emb = torch.add(baseline, step*start_i)
        end_emb = torch.add(baseline, step*end_i)
        step_new = (end_emb.unsqueeze(0) - start_emb.unsqueeze(0)) * scale
        res = torch.cat([torch.add(start_emb.unsqueeze(0), step_new*i) for i in range(num_points)], dim=0)
        return res, step_new[0]


def process_data(dataset,is_debias=False):
    occ = []
    bio = []
    gend = []

    if is_debias:
        for elem in dataset:
            occ.append(elem["p"])
            bio.append(elem["text"])
            gend.append(elem["g"])
    else:
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


# Calculating self-attention head attributions
def get_head_important_attr(args,example_dataloader,model,is_debias=False):
    if args.model_name_or_path.find("base") != -1:
        num_head, num_layer = 12, 12
    elif args.model_name_or_path.find("large") != -1:
        num_head, num_layer = 16, 24
    attr_every_head_record = [0] * num_head * num_layer

    correct_prediction = 0
    index=0
    for batch_idx, batch in enumerate(tqdm(example_dataloader)):
        if correct_prediction > args.num_examples:
            break
        index+=1
        input_ids = torch.transpose(torch.stack(batch["input_ids"]), 0, 1).to(args.device)
        attention_mask = torch.transpose(torch.stack(batch["attention_mask"]), 0, 1).to(args.device)
        if "token_type_ids" in batch:
            token_type_ids = torch.transpose(torch.stack(batch["token_type_ids"]), 0, 1).to(args.device)
        else:
            token_type_ids = None
        labels = torch.tensor(batch["label"]).long().to(args.device)
        input_len = int(attention_mask[0].sum())

        with torch.no_grad():
            loss,logits = model(
                input_ids=input_ids, get_which="res",token_type_ids=token_type_ids,attention_mask=attention_mask, labels=labels
            )
            logits = softmax(logits,dim=1)
            maxes = torch.argmax(logits, dim=1)
            if is_debias:
                if maxes[0] == labels[0] and maxes[1] == labels[1]:
                    continue
                if maxes[0] != labels[0] and maxes[1] != labels[1]:
                    continue
                if abs(logits[0][labels[0].item()] - logits[1][labels[1].item()]) <= 0.48:
                    continue
            else:
                if maxes[0] != labels: # Find the correct classification
                    continue
                if logits[0][labels[0].item()]<0.99:
                    continue

        correct_prediction += 1

        if is_debias == True: # bias attribution
            if maxes[0] != labels[0]:
                input_ids=input_ids[0].unsqueeze(0)
                if token_type_ids is not None:
                    token_type_ids=token_type_ids[0].unsqueeze(0)
                attention_mask=attention_mask[0].unsqueeze(0)
                labels=labels[0].unsqueeze(0)
            else:
                input_ids=input_ids[1].unsqueeze(0)
                if token_type_ids is not None:
                    token_type_ids=token_type_ids[1].unsqueeze(0)
                attention_mask=attention_mask[1].unsqueeze(0)
                labels=labels[1].unsqueeze(0)

            for tar_layer in range(0, num_layer):
                att, _ = model(input_ids=input_ids, get_which="att", token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels, tar_layer=tar_layer)
                att = att[0]
                # batch_size :"Total batch size for attention score cut."
                scale_att, step = scaled_input(att.data, args.batch_size, args.num_batch)  # batch_size=20  num_batch=1
                scale_att.requires_grad_(True)

                attr_all = None
                for j_batch in range(args.num_batch):  # num_batch:"Num batch of an example."
                    one_batch_att = scale_att[j_batch * args.batch_size:(j_batch + 1) * args.batch_size]
                    tar_prob, grad = model(
                        input_ids=input_ids, get_which="att", token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels,
                                           tar_layer=tar_layer, tmp_score=one_batch_att, pred_label=labels[0]
                    )
                    grad = grad.sum(dim=0)
                    attr_all = grad if attr_all is None else torch.add(attr_all, grad)
                attr_all = attr_all[:, 0:input_len, 0:input_len] * step[:, 0:input_len, 0:input_len]
                for i in range(0, num_head):
                    attr_every_head_record[tar_layer * num_head + i] += float(attr_all[i].max())
        else: # predictive attribution
            for tar_layer in range(0, num_layer):
                att, _ = model(
                    input_ids=input_ids, get_which="att", token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels, tar_layer=tar_layer
                )
                att = att[0]
                # batch_size :"Total batch size for attention score cut."
                scale_att, step = scaled_input(att.data, args.batch_size, args.num_batch)  # batch_size=20  num_batch=1
                scale_att.requires_grad_(True)

                attr_all = None
                for j_batch in range(args.num_batch):  # num_batch:"Num batch of an example."
                    one_batch_att = scale_att[j_batch * args.batch_size:(j_batch + 1) * args.batch_size]
                    tar_prob, grad = model(
                        input_ids=input_ids, get_which="att", token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels,
                                           tar_layer=tar_layer, tmp_score=one_batch_att, pred_label=labels[0]
                    )
                    grad = grad.sum(dim=0)
                    attr_all = grad if attr_all is None else torch.add(attr_all, grad)
                attr_all = attr_all[:, 0:input_len, 0:input_len] * step[:, 0:input_len, 0:input_len]
                for i in range(0, num_head):
                    attr_every_head_record[tar_layer * num_head + i] += float(attr_all[i].max())

    if is_debias:
        with open(os.path.join(args.output_dir, "head_bias_importance_attr_transformers.json"), "w") as f_out:
            f_out.write(json.dumps(attr_every_head_record, indent=2, sort_keys=True))
            f_out.write('\n')
            f_out.close()
    else:
        with open(os.path.join(args.output_dir, "head_predicte_importance_attr_transformers.json"), "w") as f_out:
            f_out.write(json.dumps(attr_every_head_record, indent=2, sort_keys=True))
            f_out.write('\n')
            f_out.close()

    return attr_every_head_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataparallel", default=False)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default='../model_save/bios/bert/original', type=str,
                        help="The output directory where the experimental results will be written.")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--fix_encoder",action="store_true",help="whether or not to update encoder; default False",)
    parser.add_argument("--model_file",
                        default="../model_save/bios/bert/original/model.2.bin",
                        type=str, help="The model file which will be evaluated.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--val_path", type=str, default="../data/bios/valbios.pkl")
    parser.add_argument("--batch_size", default=20, type=int,
                        help="Total batch size for attention score cut.")
    parser.add_argument("--num_batch", default=1, type=int, help="Num batch of an example.")
    parser.add_argument("--num_examples", default=200, type=int,
                        help="The number of dev examples to compute the attention head importance.")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model_state_dict = torch.load(args.model_file)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, state_dict=model_state_dict, num_labels=28)
    model.to(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.fix_encoder:
        for param in model.model.parameters():
            param.requires_grad = False

    if torch.cuda.device_count() and args.dataparallel:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    model = model.to(args.device)
    model.eval()

    val_file = open(args.val_path, "rb")
    val_data = pickle.load(val_file)
    val_file.close()
    val_dataset = process_data(val_data)

    bias_file = open(args.bias_path, "rb")
    bias_data = pickle.load(bias_file)
    bias_file.close()
    bias_dataset = process_data(bias_data,is_debias=True)

    def preprocess_function(examples):
        args = [examples["bio"]]
        result = tokenizer(*args, padding="max_length", max_length=args.max_seq_length, truncation=True)
        return result

    val_dataset = val_dataset.map(preprocess_function, batched=True, load_from_cache_file=True)
    eval_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    bias_dataset = bias_dataset.map(preprocess_function, batched=True, load_from_cache_file=True)
    bias_loader = DataLoader(dataset=bias_dataset, batch_size=2, shuffle=False)

    # bias attribution
    get_head_important_attr(args, bias_loader,model,is_debias=True)
    # predictive attribution
    get_head_important_attr(args, eval_loader,model,is_debias=False)


if __name__ == "__main__":
    main()
