LOG_DIR=your_path/temp_output_dir
TIME_STR=$(date +"%m_%d_%H_%M_%S")
SAVE_DIR=$LOG_DIR/$TIME_STR
echo $SAVE_DIR

deepspeed --include localhost:0,1 --master_port 2305 main.py \
    --deepspeed scripts/zero2.json \
    --output_dir $SAVE_DIR \
    --model_name your_path/models/Phi-3.5-mini-instruct \
    --rec_model_class ModelSAT \
    --rec_training_dataset_class ContrastiveRecDataset \
    --rec_trainer_class TransformersTrainer \
    --data_collator_class PaddingCollator \
    --rec_training_data_url capit_1.0 \
    --data_repr_encoders DataReprFixedPrompt \
    --model_repr_encoders ModelReprFixedPrompt \
    --model_repr_max_length 512 \
    --data_repr_max_length 256 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --dataloader_drop_last True \
    --contrastive_train_group_size 4 \
    --logging_steps 1 \
    --save_steps 1000 \
    --save_total_limit 1 \
    --gradient_checkpointing False \
    --warmup_ratio 0.1 \
    --bf16 \
    --use_lora False \
    --use_flash_attn False \
    --report_to wandb
