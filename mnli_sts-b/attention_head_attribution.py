"Pruning attention heads with attribution scores"
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import logging
import argparse
import random
import json
from torch import softmax
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.model_prune_head import BertForSequenceClassification
from classifier_processer import InputExample, InputFeatures, DataProcessor, MrpcProcessor, \
    MnliProcessor, RteProcessor, ScitailProcessor, ColaProcessor, SstProcessor, QqpProcessor, QnliProcessor, \
    WnliProcessor, StsProcessor

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mrpc": MrpcProcessor,
    "rte": RteProcessor,
    "sst-2": SstProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "wnli": WnliProcessor,
    "sts-b": StsProcessor,
    "scitail": ScitailProcessor,
}

num_labels_task = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "rte": 2,
    "sst-2": 2,
    "qqp": 2,
    "qnli": 2,
    "wnli": 2,
    "sts-b": 1,
    "scitail": 2,
}


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    if label_list:
        label_map = {label: i for i, label in enumerate(label_list)}
    else:
        label_map = None

    features = []
    tokenslist = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        base_tokens = ["[UNK]"] + ["[UNK]"] * len(tokens_a) + ["[UNK]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            base_tokens += ["[UNK]"] * len(tokens_b) + ["[UNK]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        baseline_ids = tokenizer.convert_tokens_to_ids(base_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        baseline_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(baseline_ids) == max_seq_length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if label_map:
            label_id = label_map[example.label]
        else:
            label_id = float(example.label)
        if ex_index < 2:
            logger.debug("*** Example ***")
            logger.debug("guid: %s" % (example.guid))
            logger.debug("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.debug("input_ids: %s" %
                         " ".join([str(x) for x in input_ids]))
            logger.debug("input_mask: %s" %
                         " ".join([str(x) for x in input_mask]))
            logger.debug(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.debug("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          baseline_ids=baseline_ids))
        tokenslist.append({"token": tokens, "golden_label": example.label, "pred_label": None})

    return features, tokenslist


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def scaled_input(emb, batch_size, num_batch, baseline=None, start_i=None, end_i=None):
    # shape of emb: (num_head, seq_len, seq_len)
    if baseline is None:
        baseline = torch.zeros_like(emb)

    num_points = batch_size * num_batch
    scale = 1.0 / num_points
    if start_i is None:
        step = (emb.unsqueeze(0) - baseline.unsqueeze(0)) * scale
        res = torch.cat([torch.add(baseline.unsqueeze(0), step * i) for i in range(num_points)], dim=0)
        return res, step[0]
    else:
        step = (emb - baseline) * scale
        start_emb = torch.add(baseline, step * start_i)
        end_emb = torch.add(baseline, step * end_i)
        step_new = (end_emb.unsqueeze(0) - start_emb.unsqueeze(0)) * scale
        res = torch.cat([torch.add(start_emb.unsqueeze(0), step_new * i) for i in range(num_points)], dim=0)
        return res, step_new[0]


def prepare_inputdata(examples, label_list, tokenizer, max_seq_length=128):
    lbl_type = torch.long
    # evaluate the model
    features, tokenlist = convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer)
    all_baseline_ids = torch.tensor(
        [f.baseline_ids for f in features], dtype=torch.long)
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=lbl_type)

    prepare_data = TensorDataset(all_baseline_ids, all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    prepare_sampler = SequentialSampler(prepare_data)  # Sequential sampling
    prepare_dataloader = DataLoader(prepare_data, sampler=prepare_sampler, batch_size=1)

    return prepare_dataloader


# Calculating self-attention head attributions
def get_head_important_attr(args, example_dataloader, model, device="cpu", is_debias=False):
    if args.model_name_or_path.find("base") != -1:
        num_head, num_layer = 12, 12
    elif args.model_name_or_path.find("large") != -1:
        num_head, num_layer = 16, 24
    attr_every_head_record = [0] * num_head * num_layer

    index = 0
    correct_prediction = 0
    for baseline_ids, input_ids, input_mask, segment_ids, label_ids, in example_dataloader:
        index += 1
        if correct_prediction > args.num_examples:
            break
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        input_len = int(input_mask[0].sum())

        with torch.no_grad():
            logits = model(input_ids, "log", segment_ids, input_mask, label_ids)
            logits = softmax(logits, dim=1)
            maxes = torch.argmax(logits, dim=1)

        if args.task_name=="mnli":
            if is_debias == True:
                if maxes[0] == label_ids: # Find the misclassified
                    continue
                if logits[0][2] < 0.98:
                    continue
            else:
                if maxes[0] != label_ids: # Find the correct classification
                    continue
                if logits[0][label_ids[0].item()] < 0.99:
                    continue
        elif args.task_name=="sts-b":
            if is_debias == True:
                if logits[0][0] <=3.5: # Find the misclassified
                    continue
            else:
                if abs(logits[0][0] - label_ids[0] )> 0.099: # Find the correct classification
                    continue

        correct_prediction += 1

        for tar_layer in range(0, num_layer):
            att, _ = model(input_ids, "att", segment_ids, input_mask, label_ids, tar_layer)
            att = att[0]
            # batch_size :"Total batch size for attention score cut."
            scale_att, step = scaled_input(att.data, args.batch_size, args.num_batch)  # batch_size=20  num_batch=1
            scale_att.requires_grad_(True)

            attr_all = None
            for j_batch in range(args.num_batch):  # num_batch:"Num batch of an example."
                one_batch_att = scale_att[j_batch * args.batch_size:(j_batch + 1) * args.batch_size]
                tar_prob, grad = model(
                    input_ids, "att", segment_ids, input_mask, label_ids,tar_layer, one_batch_att,pred_label=label_ids[0]
                )
                grad = grad.sum(dim=0)
                attr_all = grad if attr_all is None else torch.add(attr_all, grad)
            attr_all = attr_all[:, 0:input_len, 0:input_len] * step[:, 0:input_len, 0:input_len]
            for i in range(0, num_head):
                attr_every_head_record[tar_layer * num_head + i] += float(attr_all[i].max())

    if is_debias == True:
        with open(os.path.join(args.output_dir, "head_bias_importance_attr.json"), "w") as f_out:
            f_out.write(json.dumps(attr_every_head_record, indent=2, sort_keys=True))
            f_out.write('\n')
            f_out.close()
    else:
        with open(os.path.join(args.output_dir, "head_predicte_importance_attr.json"), "w") as f_out:
            f_out.write(json.dumps(attr_every_head_record, indent=2, sort_keys=True))
            f_out.write('\n')
            f_out.close()

    return attr_every_head_record


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default="../data/mnli", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased")
    parser.add_argument("--task_name", default="mnli", type=str, help="The name of the task to train.")
    parser.add_argument("--output_dir", default='../model_save/mnli/bert/original',
                        type=str, help="The output directory where the experimental results will be written.")
    parser.add_argument("--model_file",default="../model_save/mnli/bert/original/model.2.bin",
                        type=str, help="The model file which will be evaluated.")

    # Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", default=False, action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    # pruning head parameters
    parser.add_argument("--batch_size", default=20, type=int,
                        help="Total batch size for attention score cut.")
    parser.add_argument("--num_batch", default=1, type=int, help="Num batch of an example.")
    parser.add_argument("--num_examples", default=200, type=int,
                        help="The number of dev examples to compute the attention head importance.")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args.local_rank != -1)))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)

    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    # Load a fine-tuned model
    model_state_dict = torch.load(args.model_file)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name_or_path, state_dict=model_state_dict, num_labels=num_labels)
    model.to(device)
    model.eval()

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare the data
    eval_segment = "dev_matched" if args.task_name == "mnli" else "dev"
    eval_examples = processor.get_dev_examples(args.data_dir, segment=eval_segment)
    eval_dataloader = prepare_inputdata(eval_examples, label_list, tokenizer, args.max_seq_length)

    # Prepare debias data
    debias_segment = "debias_gender"
    debias_examples = processor.get_debias_examples(args.data_dir, segment=debias_segment)
    debias_dataloader = prepare_inputdata(debias_examples, label_list, tokenizer, args.max_seq_length)

    # bias attribution
    get_head_important_attr(args, debias_dataloader, model, device, is_debias=True)
    # predictive attribution
    get_head_important_attr(args, eval_dataloader, model, device, is_debias=False)


if __name__ == "__main__":
    main()