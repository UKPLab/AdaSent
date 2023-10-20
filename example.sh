#!/bin/bash

# Define data paths and model save paths
unlabeled_file=data/DAPT/amazon_massive_scenario.txt
labeled_train_file=data/SetFit/amazon_massive_scenario_train.csv
eval_file=data/SetFit/amazon_massive_scenario_eval.csv

base_model_name_or_path=distilroberta-base
dapt_model_save_path=models/distilroberta-base_dapt_amazon_massive_scenario
sept_adapter_save_path=models/distilroberta-base_sept_adapter
setfit_model_save_path=models/adasent_setfit_amazon_massive_scenario

# Train DAPT on base model
python scripts/DAPT/train_mlm.py \
    --train_file $unlabeled_file \
    --model_name_or_path $base_model_name_or_path \
    --max_seq_length 512 \
    --max_train_steps 2000 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --output_dir $dapt_model_save_path \
    --line_by_line True

# Train SEPT adapter on base model
python scripts/SEPT/train_sept.py \
    --model_name_or_path $base_model_name_or_path \
    --use_adapter True \
    --adapter_config parallel \
    --adapter_name sept \
    --max_seq_length 512 \
    --batch_size_pairs 64 \
    --batch_size_triplets 64 \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --pooling_mode mean \
    --model_save_path $sept_adapter_save_path \
    --use_amp True

# SetFit for few-shot classification on DAPT model + SEPT adapter
python scripts/SetFit/train_setfit.py \
    --model_name_or_path $dapt_model_save_path \
    --adapter_path $sept_adapter_save_path \
    --batch_size 16 \
    --num_epochs 1 \
    --num_samples 8 \
    --num_iterations 20 \
    --adapter_name sept \
    --model_save_path $setfit_model_save_path \
    --train_dataset_path $labeled_train_file \
    --eval_dataset_path $eval_file \
    --text_col text \
    --label_col label

# SetFit with self-training (uncomment when needed)
# python scripts/SetFit/train_setfit_with_self_training.py \
#     --model_name_or_path $dapt_model_save_path \
#     --adapter_path $sept_adapter_save_path \
#     --adapter_name sept \
#     --unlabeled_file_path $unlabeled_file \
#     --batch_size 16 \
#     --num_epochs 1 \
#     --num_samples 8 \
#     --num_iterations 20 \
#     --train_dataset_path $labeled_train_file \
#     --eval_dataset_path $eval_file \
#     --text_col text \
#     --label_col label