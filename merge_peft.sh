source ~/.zshrc
va bigcode-evaluation-harness

MODEL_NAME=$1
PEFT_DIR=$2
OUTPUT_DIR=$3

#JOB_PORT=$((60000 + 1))
#
#RUN_GPU_IDS=${CUDA_VISIBLE_DEVICES:-1}
#echo "RUNNING ON GPU: $RUN_GPU_IDS"
#unset CUDA_VISIBLE_DEVICES

accelerate launch --main_process_port 29511 --num_processes=1 \
  /workspace/DeepSeek-Coder/finetune/utils/merge_peft.py \
  --model $MODEL_NAME \
  --peft $PEFT_DIR \
  --output $OUTPUT_DIR