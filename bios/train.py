import json
import os
import torch.distributed
import argparse
import pickle
import logging
from datasets import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW

from transformers import AutoTokenizer
from pytorch_pretrained_bert.model_prune_head import BertForSequenceClassification

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


with open("../data/bios/prof2ind.json") as json_file:
    mapping = json.load(json_file)


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


def train_epoch(epoch, model, dl, eval_dl, optimizer, args):
    logger.info(f"At epoch {epoch}:")
    model.train()

    total_loss = 0

    for batch_idx, batch in enumerate(tqdm(dl)):
        input_ids = torch.transpose(torch.stack(batch["input_ids"]), 0, 1).to(args.device)
        attention_mask = torch.transpose(torch.stack(batch["attention_mask"]), 0, 1).to(args.device)
        if "token_type_ids" in batch:
            token_type_ids = torch.transpose(
                torch.stack(batch["token_type_ids"]), 0, 1
            ).to(args.device)
        else:
            token_type_ids = None
        labels = torch.tensor(batch["label"]).long().to(args.device)

        batch_loss = model(
            input_ids=input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask, labels=labels
        )
        if args.dataparallel:
            total_loss += batch_loss.mean().item()
            batch_loss.mean().backward()
        else:
            total_loss += batch_loss.item()
            batch_loss.backward()

        optimizer.step()
        model.zero_grad()
        optimizer.zero_grad()

    eval_acc = evaluate("dev", model, eval_dl, epoch, args)

    return eval_acc


def evaluate(mode, model, dl, epoch, args):
    model.eval()
    logger.info("running eval")

    total_correct = []
    for batch_idx, batch in enumerate(tqdm(dl)):
        input_ids = torch.transpose(torch.stack(batch["input_ids"]), 0, 1).to(args.device)
        attention_mask = torch.transpose(torch.stack(batch["attention_mask"]), 0, 1).to(args.device)
        labels = torch.tensor(batch["label"]).long().to(args.device)

        with torch.no_grad():
            if args.dataparallel:
                logits = model.module(input_ids=input_ids, attention_mask=attention_mask)
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)

        total_correct += (torch.argmax(logits, dim=1) == labels).tolist()

    acc = sum(total_correct) / len(total_correct)
    log_dict = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "eval_acc": acc
    }
    logger.info(mode, log_dict)

    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataparallel", default=False)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache_dir", default=None, type=str, help="cache directory")
    parser.add_argument("--train_path", type=str, default="../data/bios/trainbios.pkl")
    parser.add_argument("--val_path", type=str, default="../data/bios/valbios.pkl")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")# bert-base-uncased,roberta-base,albert-base-v2
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--max_seq_length", default=128, type=int, help="Max. sequence length")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--fix_encoder",action="store_true",help="Whether to fix encoder - default false.")
    parser.add_argument("--model_file",
                        default="../intrinsic_debiasing_model/bert/original/pytorch_model.bin",
                        type=str,help="The file of fine-tuned pretraining model.")
    parser.add_argument("--output_dir", type=str,
                        default='../model_save/bios/bert/original',
                        help="The output directory where the experimental results will be written.")
    
    args = parser.parse_args()

    train_file = open(args.train_path, "rb")
    train_data = pickle.load(train_file)
    train_file.close()

    val_file = open(args.val_path, "rb")
    val_data = pickle.load(val_file)
    val_file.close()

    train_dataset = process_data(train_data)
    val_dataset = process_data(val_data)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True, cache_dir=args.cache_dir
    )

    def preprocess_function(examples):
        # Tokenize the texts
        args = [examples["bio"]]
        result = tokenizer(*args, padding="max_length", max_length=args.max_seq_length, truncation=True)
        return result

    train_dataset = train_dataset.map(preprocess_function, batched=True, load_from_cache_file=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True, load_from_cache_file=True)

    model = BertForSequenceClassification.from_pretrained(
        args.model_name_or_path, state_dict=torch.load(args.model_file), num_labels=len(mapping))

    if args.fix_encoder:
        print("FIXING ENCODER...")
        if "roberta" in args.model_name_or_path:
            for param in model.roberta.parameters():
                param.requires_grad = False
        elif "albert" in args.model_name_or_path:
            for param in model.albert.parameters():
                param.requires_grad = False
        else:
            for param in model.bert.parameters():
                param.requires_grad = False

    if torch.cuda.device_count() and args.dataparallel:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    os.makedirs(args.output_dir, exist_ok=True)
    model = model.to(args.device)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
    eval_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)

    if args.dataparallel:
        logger.info(
            f"Trainable params: {sum(p.numel() for p in model.module.parameters() if p.requires_grad)}"
        )
        logger.info(
            f"All params      : {sum(p.numel() for p in model.module.parameters())}"
        )
    else:
        logger.info(
            f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        logger.info(f"All params      : {sum(p.numel() for p in model.parameters())}")

    if args.dataparallel:
        optimizer = AdamW(model.module.parameters(), lr=args.lr)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr)


    # Training
    eval_acc = []
    for epoch in range(int(args.num_epochs)):
        acc=train_epoch(epoch, model, train_loader, eval_loader, optimizer, args)
        eval_acc.append(acc)

        model_to_save = model.module if hasattr(model, 'module') else model  # Only model_save the model it-self
        output_model_file = os.path.join(args.output_dir, "model.{}.bin".format(epoch))
        torch.save(model_to_save.state_dict(), output_model_file)

    with open(os.path.join(args.output_dir, "result.json"), "a") as f_out:
        f_out.write(json.dumps(eval_acc, indent=2, sort_keys=True))
        f_out.write('\n')


if __name__ == "__main__":
    main()
