source ~/.zshrc
va DeepSeek-Coder

LANGUAGE=$1;
MULTIPLE_LANG=$2;
TARGET_TASK=$3;
SAVE_FOLDER=$4;
MODEL_PATH=$5;
JOB_ID=$6;

JOB_PORT=$((29500 + ${JOB_ID}))

TASK=multiple-${TARGET_TASK}-${MULTIPLE_LANG}
GENERATION_PATH=/workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-instruct/${SAVE_FOLDER}/generations_$TASK.json
EVAL_DIR=/workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-instruct/${SAVE_FOLDER}/${LANGUAGE}-eval
METRIC_OUTPUT_PATH=/workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-instruct/${SAVE_FOLDER}/evaluation_results_$TASK.json;
accelerate launch \
  --main_process_port ${JOB_PORT} --num_processes=1 \
  --config_file /workspace/bigcode-evaluation-harness/Evaluation/HumanEval/test_config.yaml \
  /workspace/bigcode-evaluation-harness/Evaluation/HumanEval/eval_instruct.py \
  --model ${MODEL_PATH} \
  --task ${TARGET_TASK} \
  --language ${LANGUAGE} \
  --output_path ${GENERATION_PATH} \
  --temp_dir ${EVAL_DIR} \
  --metric_path ${METRIC_OUTPUT_PATH}

PING $TASK DONE;

