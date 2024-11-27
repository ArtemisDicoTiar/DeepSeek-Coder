source ~/.zshrc
va DeepSeek-Coder

DATA_DIR=$1;
OUTPUT_DIR=$2;
LANGUAGE=$3;
JOB_ID=$4;
MODEL_NAME=$5;
alpha=${6:-1024};

peft_lora_r=192 # set by 192 for paper
peft_lora_alpha=192 # set by 192 for paper

JOB_PORT=$((60000 + ${JOB_ID}))

mkdir -p DATA_DIR;
mkdir -p OUTPUT_DIR;

DATA_PATH=${DATA_DIR}/${LANGUAGE}.jsonl
OUTPUT_PATH=${OUTPUT_DIR}/${LANGUAGE}
MODEL_PATH=${MODEL_NAME}

RUN_GPU_IDS=${CUDA_VISIBLE_DEVICES:-1}
echo "RUNNING ON GPU: $RUN_GPU_IDS"
unset CUDA_VISIBLE_DEVICES

#python3 \
deepspeed \
    --include localhost:${RUN_GPU_IDS} \
    --master_port ${JOB_PORT} \
finetune/finetune_deepseekcoder_sora.py \
    --deepspeed finetune/configs/ds_config_zero1_magi_no_optim.json \
    --model_name_or_path $MODEL_PATH \
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
    --learning_rate 15e-6 \
    --logging_steps 1 \
    --lr_scheduler_type "linear" \
    --gradient_checkpointing True \
    --report_to "wandb" \
    --max_grad_norm 0.1 \
    --warmup_ratio 0.06 \
    --weight_decay 0.1 \
    --sparse_lambda 10 \
    --sparse_lambda_2 3e-4 \
    --lora_r $peft_lora_r \
    --lora_alpha $peft_lora_alpha \
    --bf16 True;

#rm -rf ${OUTPUT_PATH}/**/global_step*;
find "${OUTPUT_PATH}" -type d -name "global_step*" -exec rm -rf {} +

PING $0 ${LANGUAGE} DONE;
