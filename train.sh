source ~/.zshrc
va DeepSeek-Coder

DATA_DIR=$1;
OUTPUT_DIR=$2;
LANGUAGE=$3;
JOB_ID=$4;

JOB_PORT=$((60000 + ${JOB_ID}))

mkdir -p DATA_DIR;
mkdir -p OUTPUT_DIR;

DATA_PATH=${DATA_DIR}/$LANGUAGE.jsonl
OUTPUT_PATH=${OUTPUT_DIR}/$LANGUAGE
MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-base"

RUN_GPU_IDS=${CUDA_VISIBLE_DEVICES:-1}
echo "RUNNING ON GPU: $RUN_GPU_IDS"
unset CUDA_VISIBLE_DEVICES
deepspeed \
    --include localhost:${RUN_GPU_IDS} \
    --master_port ${JOB_PORT} \
finetune/finetune_deepseekcoder.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 3 \
    --model_max_length 1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "wandb" \
    --deepspeed finetune/configs/ds_config_zero1.json \
    --bf16 True;

#rm -rf ${OUTPUT_PATH}/**/global_step*;
find "${OUTPUT_PATH}" -type d -name "global_step*" -exec rm -rf {} +

PING PLEX ${LANGUAGE} DONE;
