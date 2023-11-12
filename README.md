## Attribution-and-Pruing-Debias


## Catalogue description
  * data
    + All data used in the experiment
  * model_save
    + Storing fine-tuned models for downstream tasks
    + Storing scores for bias attribution and predictive attribution
  * bios
    + Occupational classification task
  * moji
    + Emotion prediction task
  * mnli_sts-b
    + Natural language inference task and Semantic similarity task
  * intrinsic_debiasing_model
    + Storing the models debiased with task-agnostic debiasing methods (context-debias、auto-debias、attend、mabel)
  * pytorch_pretrained_bert
    + Self-attention attribution and pruning based on the bert model
  * transformers
    + Self-attention attribution and pruning based on BERT, Albert, and RoBERTa models




## 1、Configuration environment

```bash
pip install -r requirements.txt
```

## 2、Prepare data

All data can be downloaded at: https://drive.google.com/drive/folders/1HsG3FNboRihCjWmT7F_dk7YVcjZ7BVji


## 3、Run
### In all experiments " model_name_or_path、model_recover_path、output_dir " need to be consistent

## 3.1 BIOS
**1. Fine-tuning the model**
```bash
export MODEL_NAME_OR_PATH=bert-base-uncased
export MODEL_FILE=../intrinsic_debiasing_model/bert/original/pytorch_model.bin
export OUTPUT_DIR=../model_save/bios/bert/original

python train.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --model_file $MODEL_RECOVER_PATH
  --output_dir $OUTPUT_DIR
  --batch_size 32 \
  --lr 1e-5 \
  --num_epochs 3 \
```


**2. Calculating bias attribution and prediction attribution**
```bash
export MODEL_NAME_OR_PATH=bert-base-uncased
export MODEL_FILE=../model_save/bios/bert/original/model.2.bin
export OUTPUT_DIR=../model_save/bios/bert/original

python attention_head_attribution.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --model_file $MODEL_FILE
  --output_dir $OUTPUT_DIR
  --num_examples 200 \
```


**3. Model pruning and fairness measures**
```bash
export MODEL_NAME_OR_PATH=bert-base-uncased
export MODEL_FILE=../model_save/bios/bert/original/model.2.bin
export OUTPUT_DIR=../model_save/bios/bert/original

python eval_bios.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --model_file $MODEL_FILE
  --output_dir $OUTPUT_DIR
  --batch_size 32 \
```




## 3.2 MOJI
**1. Fine-tuning the model**
```bash
export MODEL_NAME_OR_PATH=bert-base-uncased
export MODEL_FILE=../intrinsic_debiasing_model/bert/original/pytorch_model.bin
export OUTPUT_DIR=../model_save/moji/bert/original

python train.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --model_file $MODEL_RECOVER_PATH
  --output_dir $OUTPUT_DIR
  --batch_size 32 \
  --lr 5e-5 \
  --num_epochs 3 \
```


**2. Calculating bias attribution and prediction attribution**
```bash
export MODEL_NAME_OR_PATH=bert-base-uncased
export MODEL_FILE=../model_save/moji/bert/original/model.2.bin
export OUTPUT_DIR=../model_save/moji/bert/original

python attention_head_attribution.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --model_file $MODEL_FILE
  --output_dir $OUTPUT_DIR
  --num_examples 200 \
```


**3. Model pruning and fairness measures**
```bash
export MODEL_NAME_OR_PATH=bert-base-uncased
export MODEL_FILE=../model_save/moji/bert/original/model.2.bin
export OUTPUT_DIR=../model_save/moji/bert/original

python eval_moji.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --model_file $MODEL_FILE
  --output_dir $OUTPUT_DIR
  --batch_size 64 \
```

## 3.3 MNLI
**1. Fine-tuning the model**
```bash
export TASK_NAME=mnli
export DATE_DIR=../data/mnli
export MODEL_NAME_OR_PATH=bert-base-uncased
export MODEL_FILE=../intrinsic_debiasing_model/bert/original/pytorch_model.bin
export OUTPUT_DIR=../model_save/mnli/bert/original

python train.py \
  --task_name $TASK_NAME
  --data_dir $DATE_DIR
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --model_file $MODEL_RECOVER_PATH
  --output_dir $OUTPUT_DIR
  --do_train \
  --do_eval \
  --train_batch_size 32 \
  --eval_batch_size 64 \
  --lr 5e-5 \
  --num_epochs 3 \
```

**2. Calculating bias attribution and prediction attribution**
```bash
export TASK_NAME=mnli
export DATE_DIR=../data/mnli
export MODEL_NAME_OR_PATH=bert-base-uncased
export MODEL_FILE=../model_save/mnli/bert/original/model.2.bin
export OUTPUT_DIR=../model_save/mnli/bert/original

python attention_head_attribution.py \
  --task_name $TASK_NAME
  --data_dir $DATE_DIR
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --model_file $MODEL_FILE
  --output_dir $OUTPUT_DIR
  --num_examples 200 \
```


**3. Model pruning and fairness measures**
```bash
export MODEL_NAME_OR_PATH=bert-base-uncased
export MODEL_FILE=../model_save/mnli/bert/original/model.2.bin
export OUTPUT_DIR=../model_save/mnli/bert/original

python eval_mnli.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --model_file $MODEL_FILE
  --output_dir $OUTPUT_DIR
  --batch_size 64 \
```


## 3.4 STS-B
**1. Fine-tuning the model**
```bash
export TASK_NAME=sts-b
export DATE_DIR=../data/sts-b
export MODEL_NAME_OR_PATH=bert-base-uncased
export MODEL_FILE=../intrinsic_debiasing_model/bert/original/pytorch_model.bin
export OUTPUT_DIR=../model_save/sts-b/bert/original

python train.py \
  --task_name $TASK_NAME
  --data_dir $DATE_DIR
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --model_file $MODEL_RECOVER_PATH
  --output_dir $OUTPUT_DIR
  --do_train \
  --do_eval \
  --train_batch_size 32 \
  --eval_batch_size 64 \
  --lr 5e-5 \
  --num_epochs 3 \
```


**2. Calculating bias attribution and prediction attribution**
```bash
export TASK_NAME=sts-b
export DATE_DIR=../data/sts-b
export MODEL_NAME_OR_PATH=bert-base-uncased
export MODEL_FILE=../model_save/sts-b/bert/original/model.2.bin
export OUTPUT_DIR=../model_save/sts-b/bert/original

python attention_head_attribution.py \
  --task_name $TASK_NAME
  --data_dir $DATE_DIR
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --model_file $MODEL_FILE
  --output_dir $OUTPUT_DIR
  --num_examples 200 \
```


**3. Model pruning and fairness measures**
```bash
export MODEL_NAME_OR_PATH=bert-base-uncased
export MODEL_FILE=../model_save/sts-b/bert/original/model.2.bin
export OUTPUT_DIR=../model_save/sts-b/bert/original

python eval_sts-b.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --model_file $MODEL_FILE
  --output_dir $OUTPUT_DIR
  --batch_size 64 \
```




## Code Acknowledgements
**1. Self-Attention Attribution**
  * https://github.com/YRdddream/attattr

**2.Intrinsic Debiasing Method**
  * Context-Debias:https://github.com/kanekomasahiro/context-debias
  * Auto-Debias:https://github.com/Irenehere/Auto-Debias
  * AttenD:https://github.com/YacineGACI/AttenD
  * MABEL:https://github.com/princeton-nlp/MABEL

**3.Data**
  * BIOS:https://github.com/shauli-ravfogel/nullspace_projection/blob/master/download_data.sh
  * MOJI:
    + TwitterAAE(original data):http://slanglab.cs.umass.edu/TwitterAAE/
    + emotional marker：https://github.com/yanaiela/demog-text-removal/tree/master/src/data
  * MNLI and STS-B:
    + MNLI and STS-B data: https://gluebenchmark.com/tasks
    + SNLI: https://n1p.stanford.edu/projects/snli/snli_1.0.zip
    + Bias-NLI data generate code: https://github.com/sunipa/On-Measuring-and-Mitigating-Biased-Inferences-of-Word-Embeddings/tree/master/word_lists
    + Bias-NLI data: https://drive.google.com/file/d/1eC003yjOHjkp5-TGyVXW1emlV80qB7Yl/view?usp=sharing
   
