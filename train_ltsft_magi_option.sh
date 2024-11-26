source ~/.zshrc
va DeepSeek-Coder

DATA_DIR=$1;
OUTPUT_DIR=$2;
LANGUAGE=$3;
JOB_ID=$4;
MODEL_NAME=$5;
lr=${6:-5e-6};

JOB_PORT=$((60000 + ${JOB_ID}))

mkdir -p DATA_DIR;
mkdir -p OUTPUT_DIR;

DATA_PATH=${DATA_DIR}/${LANGUAGE}.jsonl
OUTPUT_PATH=${OUTPUT_DIR}/${LANGUAGE}
MODEL_PATH=${MODEL_NAME}

RUN_GPU_IDS=${CUDA_VISIBLE_DEVICES:-1}
echo "RUNNING ON GPU: $RUN_GPU_IDS"
unset CUDA_VISIBLE_DEVICES

full_ckpt_path=/data/DeepSeek-Coder/experiments-magi/deepseek-ai/deepseek-coder-6.7b-base/ise-uiuc/Magicoder-OSS-Instruct-75K/$LANGUAGE
largest_checkpoint=$(ls -d ${full_ckpt_path}/checkpoint-* 2>/dev/null | grep -oP 'checkpoint-\K[0-9]+' | sort -nr | head -n 1)

params_num=202215383 # about 3% of the total number of deepseekcoder-6.7b (6740512768)
deepspeed \
    --include localhost:${RUN_GPU_IDS} \
    --master_port ${JOB_PORT} \
finetune/finetune_deepseekcoder_lt_sft.py \
    --model_name_or_path $MODEL_PATH \
    --full_ckpt_path $full_ckpt_path/checkpoint-${largest_checkpoint} \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 2 \
    --model_max_length 16384 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate ${lr} \
    --warmup_steps 15 \
    --logging_steps 1 \
    --lr_scheduler_type "linear" \
    --gradient_checkpointing True \
    --report_to "wandb" \
    --deepspeed finetune/configs/ds_config_zero1_magi.json \
    --bf16 True \
    --ft_params_num ${params_num} \
    --freeze_layer_norm \
    --full_l1_reg 0.0 \
    --sparse_l1_reg 0.0 \
    --full_ft_max_epochs_per_iteration 2 \
    --sparse_ft_max_epochs_per_iteration 2
#    --mask_embed=True


#rm -rf ${OUTPUT_PATH}/**/global_step*;
find "${OUTPUT_PATH}" -type d -name "global_step*" -exec rm -rf {} +

PING PLEX ${LANGUAGE} DONE;
