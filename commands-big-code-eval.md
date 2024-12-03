```bash
LANGUAGE=java;
BIG_CODE_LANGUAGE=java;
EXPERIMENT_NAME="experiments";
MODEL_NAME=deepseek-coder-6.7b-base;
EXPERIMENT_DIR=deepseek-ai/deepseek-coder-6.7b-base;
ts --gpus 2 sh ./evaluate.sh ${LANGUAGE} ${BIG_CODE_LANGUAGE} humaneval baseline ${EXPERIMENT_DIR} ${MODEL_NAME} ${EXPERIMENT_NAME};
ts --gpus 2 sh ./evaluate.sh ${LANGUAGE} ${BIG_CODE_LANGUAGE} mbpp baseline ${EXPERIMENT_DIR} ${MODEL_NAME} ${EXPERIMENT_NAME};

ts --gpus 2 sh ./evaluate.sh cpp cpp humaneval baseline deepseek-ai/deepseek-coder-6.7b-base;
ts --gpus 2 sh ./evaluate.sh cpp cpp mbpp baseline deepseek-ai/deepseek-coder-6.7b-base;

ts --gpus 2 sh ./evaluate.sh cpp cpp humaneval baseline deepseek-ai/deepseek-coder-6.7b-base;
ts --gpus 2 sh ./evaluate.sh cpp cpp mbpp baseline deepseek-ai/deepseek-coder-6.7b-base;

ts --gpus 2 sh ./evaluate.sh php php humaneval baseline deepseek-ai/deepseek-coder-6.7b-base;
ts --gpus 2 sh ./evaluate.sh php php mbpp baseline deepseek-ai/deepseek-coder-6.7b-base;

ts --gpus 2 sh ./evaluate.sh swift swift humaneval baseline deepseek-ai/deepseek-coder-6.7b-base;
ts --gpus 2 sh ./evaluate.sh swift swift mbpp baseline deepseek-ai/deepseek-coder-6.7b-base;

ts --gpus 2 sh ./evaluate.sh go go humaneval baseline deepseek-ai/deepseek-coder-6.7b-base;
ts --gpus 2 sh ./evaluate.sh go go mbpp baseline deepseek-ai/deepseek-coder-6.7b-base;

ts --gpus 2 sh ./evaluate.sh rust rs humaneval baseline deepseek-ai/deepseek-coder-6.7b-base;
ts --gpus 2 sh ./evaluate.sh rust rs mbpp baseline deepseek-ai/deepseek-coder-6.7b-base;

ts --gpus 2 sh ./evaluate.sh scala scala humaneval baseline deepseek-ai/deepseek-coder-6.7b-base;
ts --gpus 2 sh ./evaluate.sh scala scala mbpp baseline deepseek-ai/deepseek-coder-6.7b-base;

ts --gpus 4 sh ./evaluate.sh r r humaneval baseline deepseek-ai/deepseek-coder-6.7b-base;
ts --gpus 4 sh ./evaluate.sh r r mbpp baseline deepseek-ai/deepseek-coder-6.7b-base;


ise-uiuc/Magicoder-OSS-Instruct-75K;
ise-uiuc/Magicoder-Evol-Instruct-110K
rombodawg/MegaCodeTraining
theblackcat102/evol-codealpaca-v1
m-a-p/CodeFeedback-Filtered-Instruction

# todos
# MAIN LANGS=(java php cpp swift) 
# NEW LANGS=(go rust scala python)
#SUBMIT_LANGUAGES=(java php cpp swift go rust scala python);
SUBMIT_LANGUAGES=(java);
TARGET_TASKS=(humaneval mbpp);
EXPERIMENT_NAME="experiments-magi";
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base;
DATASET=ise-uiuc/Magicoder-OSS-Instruct-75K;
for TASK in ${TARGET_TASKS[@]}; do
  for LANGUAGE in ${SUBMIT_LANGUAGES[@]}; do
    if [ $LANGUAGE = "rust" ]; then
        BIG_CODE_LANGUAGE=rs
    elif [ $LANGUAGE = "python" ]; then
        BIG_CODE_LANGUAGE=py
    else
        BIG_CODE_LANGUAGE=$LANGUAGE
    fi
    EXPERIMENT_DIR=/workspace/DeepSeek-Coder/${EXPERIMENT_NAME}/${MODEL_NAME}/${DATASET}/${LANGUAGE};
    ts --gpus 2 sh ./evaluate.sh ${LANGUAGE} ${BIG_CODE_LANGUAGE} ${TASK} ${DATASET}/results ${EXPERIMENT_DIR} ${MODEL_NAME} ${EXPERIMENT_NAME};
  done;
done;
```
~~~
MODEL_NAME=deepseek-coder-6.7b-base;
DATASET=ise-uiuc/Magicoder-OSS-Instruct-75K;
sh ./merge_results.py /workspace/DeepSeek-Coder/experiments/${MODEL_NAME}/${DATASET}/results
~~~

DATASET=ise-uiuc/Magicoder-OSS-Instruct-75K;
ts --gpus 2 sh ./evaluate.sh cpp cpp humaneval results /workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-base/cpp;
ts --gpus 2 sh ./evaluate.sh cpp cpp mbpp results /workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-base/cpp;




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
