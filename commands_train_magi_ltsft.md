```bash
ise-uiuc/Magicoder-OSS-Instruct-75K;
ise-uiuc/Magicoder-Evol-Instruct-110K
rombodawg/MegaCodeTraining
theblackcat102/evol-codealpaca-v1


MAGIC_NUMBER=10

lr=5e-6
DATA_NAME=ise-uiuc/Magicoder-OSS-Instruct-75K
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base
OUTPUT_DIR=/workspace/DeepSeek-Coder/experiments-magi-ltsft-${lr}/${MODEL_NAME}/${DATA_NAME};
ts --gpus 8 sh ./train_ltsft_magi_option.sh /workspace/DeepSeek-Coder/data/$DATA_NAME $OUTPUT_DIR scala 1 $MODEL_NAME $lr
  "2e-7"

for lr in "5e-6" "2e-6"; do
    DATA_NAME=rombodawg/MegaCodeTraining
    MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base
    OUTPUT_DIR=/workspace/DeepSeek-Coder/experiments-magi-ltsft-${lr}/${MODEL_NAME}/${DATA_NAME};
    ts --gpus 8 sh ./train_ltsft_magi_option.sh /workspace/DeepSeek-Coder/data/$DATA_NAME $OUTPUT_DIR rust 1 $MODEL_NAME $lr
done;

SUBMIT_LANGUAGES=(rust);
TARGET_TASKS=(humaneval mbpp);
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base
DATASET=rombodawg/MegaCodeTraining
for TASK in ${TARGET_TASKS[@]}; do
    for lr in "5e-5"; do
      EXPERIMENT_NAME="experiments-magi-ltsft-${lr}";
      for LANGUAGE in ${SUBMIT_LANGUAGES[@]}; do
        if [ $LANGUAGE = "rust" ]; then
            BIG_CODE_LANGUAGE=rs
        elif [ $LANGUAGE = "python" ]; then
            BIG_CODE_LANGUAGE=py
        else
            BIG_CODE_LANGUAGE=$LANGUAGE
        fi
        EXPERIMENT_DIR=/workspace/DeepSeek-Coder/${EXPERIMENT_NAME}/${MODEL_NAME}/${DATASET}/${LANGUAGE};
        ts --gpus 4 sh ./evaluate.sh ${LANGUAGE} ${BIG_CODE_LANGUAGE} ${TASK} ${DATASET}/results ${EXPERIMENT_DIR} ${MODEL_NAME} ${EXPERIMENT_NAME} false true;
      done;
    done;
done;

DATA_NAME=ise-uiuc/Magicoder-OSS-Instruct-75K
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base
OUTPUT_DIR=/workspace/DeepSeek-Coder/experiments-magi-ltsft/${MODEL_NAME}/${DATA_NAME};
ts --gpus 8 sh ./train_ltsft_magi_option.sh /workspace/DeepSeek-Coder/data/$DATA_NAME $OUTPUT_DIR cpp 1 $MODEL_NAME; 

DATA_NAME=ise-uiuc/Magicoder-OSS-Instruct-75K
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base
OUTPUT_DIR=/workspace/DeepSeek-Coder/experiments-magi-ltsft/${MODEL_NAME}/${DATA_NAME};
ts --gpus 8 sh ./train_ltsft_magi_option.sh /workspace/DeepSeek-Coder/data/$DATA_NAME $OUTPUT_DIR php 2 $MODEL_NAME;

DATA_NAME=ise-uiuc/Magicoder-OSS-Instruct-75K
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base
OUTPUT_DIR=/workspace/DeepSeek-Coder/experiments-magi-ltsft/${MODEL_NAME}/${DATA_NAME};
ts --gpus 8 sh ./train_ltsft_magi_option.sh /workspace/DeepSeek-Coder/data/$DATA_NAME $OUTPUT_DIR swift 3 $MODEL_NAME;

DATA_NAME=ise-uiuc/Magicoder-OSS-Instruct-75K
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base
OUTPUT_DIR=/workspace/DeepSeek-Coder/experiments-magi-ltsft/${MODEL_NAME}/${DATA_NAME};
ts --gpus 8 sh ./train_ltsft_magi_option.sh /workspace/DeepSeek-Coder/data/$DATA_NAME $OUTPUT_DIR java 7 $MODEL_NAME;


DATA_NAME=ise-uiuc/Magicoder-OSS-Instruct-75K
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base
OUTPUT_DIR=/workspace/DeepSeek-Coder/experiments-magi-ltsft/${MODEL_NAME}/${DATA_NAME};
ts -g 0,1,2,3 sh ./train_ltsft_magi_option.sh /workspace/DeepSeek-Coder/data/$DATA_NAME $OUTPUT_DIR go 4 $MODEL_NAME;

DATA_NAME=ise-uiuc/Magicoder-OSS-Instruct-75K
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base
OUTPUT_DIR=/workspace/DeepSeek-Coder/experiments-magi-ltsft/${MODEL_NAME}/${DATA_NAME};
ts -g 0,1,2,3 sh ./train_ltsft_magi_option.sh /workspace/DeepSeek-Coder/data/$DATA_NAME $OUTPUT_DIR scala 6 $MODEL_NAME;

DATA_NAME=ise-uiuc/Magicoder-OSS-Instruct-75K
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base
OUTPUT_DIR=/workspace/DeepSeek-Coder/experiments-magi-ltsft/${MODEL_NAME}/${DATA_NAME};
ts -g 0,1,2,3 sh ./train_ltsft_magi_option.sh /workspace/DeepSeek-Coder/data/$DATA_NAME $OUTPUT_DIR python 0 $MODEL_NAME;

DATA_NAME=ise-uiuc/Magicoder-OSS-Instruct-75K
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base
OUTPUT_DIR=/workspace/DeepSeek-Coder/experiments-magi-ltsft/${MODEL_NAME}/${DATA_NAME};
ts -g 0,1,2,3 sh ./train_ltsft_magi_option.sh /workspace/DeepSeek-Coder/data/$DATA_NAME $OUTPUT_DIR rust 5 $MODEL_NAME;


```

~~~
while true; do
for id in $(ts -g | awk 'NR > 1 {print $1}'); do
    command=$(ts -F $id)

    # Capture only the last line of progress
    progress=$(tail -n 1 $(ts -o $id))

    # Extract "X/Y" from the last line
    steps=$(echo "$progress" | grep -oP '\d+/\d+' | tail -n 1)

    # Extract "HH:MM" from the last line
    remaining_time=$(echo "$progress" | grep -oP '\d+:\d+(?=,)' | tail -n 1)

    # Check if both variables are non-empty (i.e., if the pattern was matched)
    if [[ -n "$steps" && -n "$remaining_time" ]]; then
        # Output the extracted values
        PING ${command} \n${steps} ${remaining_time}
    fi
done;
sleep 1800;
done;
~~~

for num in 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264; do
ts -r $num
done