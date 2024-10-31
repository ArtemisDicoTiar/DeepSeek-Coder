```bash

ts --gpus 2 sh ./eval_multiple.sh cpp cpp humaneval baseline deepseek-ai/deepseek-coder-6.7b-instruct 1;
ts --gpus 2 sh ./eval_multiple.sh cpp cpp mbpp baseline deepseek-ai/deepseek-coder-6.7b-instruct 2;

ts --gpus 2 sh ./eval_multiple.sh php php humaneval baseline deepseek-ai/deepseek-coder-6.7b-instruct 3;
ts --gpus 2 sh ./eval_multiple.sh php php mbpp baseline deepseek-ai/deepseek-coder-6.7b-instruct 4;

ts --gpus 2 sh ./eval_multiple.sh swift swift humaneval baseline deepseek-ai/deepseek-coder-6.7b-instruct 5;
ts --gpus 2 sh ./eval_multiple.sh swift swift mbpp baseline deepseek-ai/deepseek-coder-6.7b-instruct 6;

ts --gpus 2 sh ./eval_multiple.sh go go humaneval baseline deepseek-ai/deepseek-coder-6.7b-instruct 7;
ts --gpus 2 sh ./eval_multiple.sh go go mbpp baseline deepseek-ai/deepseek-coder-6.7b-instruct 8;

ts --gpus 2 sh ./eval_multiple.sh rust rs humaneval baseline deepseek-ai/deepseek-coder-6.7b-instruct 9;
ts --gpus 2 sh ./eval_multiple.sh rust rs mbpp baseline deepseek-ai/deepseek-coder-6.7b-instruct 10;

ts --gpus 2 sh ./eval_multiple.sh scala scala humaneval baseline deepseek-ai/deepseek-coder-6.7b-instruct 11;
ts --gpus 2 sh ./eval_multiple.sh scala scala mbpp baseline deepseek-ai/deepseek-coder-6.7b-instruct 12;


# todos
ts --gpus 2 sh ./eval_multiple.sh php php humaneval results /workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-instruct/php 1;
ts --gpus 2 sh ./eval_multiple.sh php php mbpp results /workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-instruct/php 2;

ts --gpus 2 sh ./eval_multiple.sh cpp cpp humaneval results /workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-instruct/cpp 3;
ts --gpus 2 sh ./eval_multiple.sh cpp cpp mbpp results /workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-instruct/cpp 4;

ts --gpus 2 sh ./eval_multiple.sh swift swift humaneval results /workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-instruct/swift 5;
ts --gpus 2 sh ./eval_multiple.sh swift swift mbpp results /workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-instruct/swift 6;

ts --gpus 2 sh ./eval_multiple.sh go go humaneval results /workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-instruct/go 7;
ts --gpus 2 sh ./eval_multiple.sh go go mbpp results /workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-instruct/go 8;

ts --gpus 2 sh ./eval_multiple.sh rust rs humaneval results DicoTiar/deepseek-coder-6.7b-instruct-full-ft-rust 9;
ts --gpus 2 sh ./eval_multiple.sh rust rs mbpp results DicoTiar/deepseek-coder-6.7b-instruct-full-ft-rust 10;

ts --gpus 2 sh ./eval_multiple.sh scala scala humaneval results /data/DeepSeek-Coder/experiments/deepseek-coder-6.7b-instruct/scala 11;
ts --gpus 2 sh ./eval_multiple.sh scala scala mbpp results /data/DeepSeek-Coder/experiments/deepseek-coder-6.7b-instruct/scala 12;
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