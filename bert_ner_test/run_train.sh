#!/bin/bash 

python3 bert_entity_extractor/run_tf_ner.py \
    --model_name_or_path vinai/phobert-base \
    --data_train mounts/data/train_ner.txt \
    --output_dir mounts/models/bert_ner_10_epochs_01_lr \
    --do_train \
    --logging_steps 20 \
    --save_steps 100000 \
    --save_total_limit 2 \
    --learning_rate 0.00001 \
    --max_seq_length 50 \
    --per_device_eval_batch_size 4 \
    --per_device_train_batch_size 4 \
    --num_train_epochs 10

# python3 bert_entity_extractor/run_tf_ner.py \
#     --model_name_or_path vinai/phobert-base \
#     --data_train mounts/data/train_ner.txt \
#     --data_eval mounts/data/test_ner.txt \
#     --output_dir mounts/models/bert_ner_15_epochs \
#     --do_train \
#     --do_eval \
#     --eval_steps 100 \
#     --logging_steps 20 \
#     --save_total_limit 2 \
#     --learning_rate 0.00005 \
#     --max_seq_length 50 \
#     --per_device_eval_batch_size 2 \
#     --per_device_train_batch_size 2 \
#     --num_train_epochs 10
