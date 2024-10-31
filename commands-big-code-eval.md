```bash

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


ise-uiuc/Magicoder-OSS-Instruct-75K;
ise-uiuc/Magicoder-Evol-Instruct-110K
rombodawg/MegaCodeTraining
theblackcat102/evol-codealpaca-v1

# todos
DATASET=rombodawg/MegaCodeTraining
ts --gpus 2 sh ./evaluate.sh cpp cpp humaneval ${DATASET}/results /workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-base/${DATASET}/cpp;
ts --gpus 2 sh ./evaluate.sh cpp cpp mbpp ${DATASET}/results /workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-base/${DATASET}/cpp;

DATASET=rombodawg/MegaCodeTraining
ts --gpus 2 sh ./evaluate.sh php php humaneval ${DATASET}/results /workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-base/${DATASET}/php;
ts --gpus 2 sh ./evaluate.sh php php mbpp ${DATASET}/results /workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-base/${DATASET}/php;

DATASET=rombodawg/MegaCodeTraining
ts --gpus 2 sh ./evaluate.sh swift swift humaneval ${DATASET}/results /workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-base/${DATASET}/swift;
ts --gpus 2 sh ./evaluate.sh swift swift mbpp ${DATASET}/results /workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-base/${DATASET}/swift;

DATASET=rombodawg/MegaCodeTraining
ts --gpus 2 sh ./evaluate.sh go go humaneval ${DATASET}/results /workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-base/${DATASET}/go;
ts --gpus 2 sh ./evaluate.sh go go mbpp ${DATASET}/results /workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-base/${DATASET}/go;

DATASET=rombodawg/MegaCodeTraining
ts --gpus 2 sh ./evaluate.sh rust rs humaneval ${DATASET}/results /workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-base/${DATASET}/rust;
ts --gpus 2 sh ./evaluate.sh rust rs mbpp ${DATASET}/results /workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-base/${DATASET}/rust;

DATASET=rombodawg/MegaCodeTraining
ts --gpus 2 sh ./evaluate.sh scala scala humaneval ${DATASET}/results /workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-base/${DATASET}/scala;
ts --gpus 2 sh ./evaluate.sh scala scala mbpp ${DATASET}/results /workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-base/${DATASET}/scala;
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
