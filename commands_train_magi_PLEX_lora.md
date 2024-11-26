```bash
ise-uiuc/Magicoder-OSS-Instruct-75K;
ise-uiuc/Magicoder-Evol-Instruct-110K
rombodawg/MegaCodeTraining
theblackcat102/evol-codealpaca-v1


MAGIC_NUMBER=10

DATA_NAME=ise-uiuc/Magicoder-OSS-Instruct-75K
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base
OUTPUT_DIR=/workspace/DeepSeek-Coder/experiments-magi-lora-plex-e/${MODEL_NAME}/${DATA_NAME};
ts --gpus 8 sh ./train_plex_e_magi_option.sh /workspace/DeepSeek-Coder/data/$DATA_NAME $OUTPUT_DIR cpp 1 $MODEL_NAME;

DATA_NAME=ise-uiuc/Magicoder-OSS-Instruct-75K
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base
OUTPUT_DIR=/workspace/DeepSeek-Coder/experiments-magi-lora-plex-e/${MODEL_NAME}/${DATA_NAME};
ts --gpus 8 sh ./train_plex_e_magi_option.sh /workspace/DeepSeek-Coder/data/$DATA_NAME $OUTPUT_DIR php 2 $MODEL_NAME;

DATA_NAME=ise-uiuc/Magicoder-OSS-Instruct-75K
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base
OUTPUT_DIR=/workspace/DeepSeek-Coder/experiments-magi-lora-plex-e/${MODEL_NAME}/${DATA_NAME};
ts --gpus 8 sh ./train_plex_e_magi_option.sh /workspace/DeepSeek-Coder/data/$DATA_NAME $OUTPUT_DIR swift 3 $MODEL_NAME;

DATA_NAME=ise-uiuc/Magicoder-OSS-Instruct-75K
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base
OUTPUT_DIR=/workspace/DeepSeek-Coder/experiments-magi-lora-plex-e/${MODEL_NAME}/${DATA_NAME};
ts --gpus 8 sh ./train_plex_e_magi_option.sh /workspace/DeepSeek-Coder/data/$DATA_NAME $OUTPUT_DIR java 7 $MODEL_NAME;

DATA_NAME=ise-uiuc/Magicoder-OSS-Instruct-75K
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base
OUTPUT_DIR=/workspace/DeepSeek-Coder/experiments-magi-lora-plex-e/${MODEL_NAME}/${DATA_NAME};
ts --gpus 1 sh ./train_plex_e_magi_option.sh /workspace/DeepSeek-Coder/data/$DATA_NAME $OUTPUT_DIR python 0 $MODEL_NAME;

DATA_NAME=ise-uiuc/Magicoder-OSS-Instruct-75K
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base
OUTPUT_DIR=/workspace/DeepSeek-Coder/experiments-magi-lora-plex-e/${MODEL_NAME}/${DATA_NAME};
ts --gpus 1 sh ./train_plex_e_magi_option.sh /workspace/DeepSeek-Coder/data/$DATA_NAME $OUTPUT_DIR go 4 $MODEL_NAME;

DATA_NAME=ise-uiuc/Magicoder-OSS-Instruct-75K
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base
OUTPUT_DIR=/workspace/DeepSeek-Coder/experiments-magi-lora-plex-e/${MODEL_NAME}/${DATA_NAME};
ts --gpus 1 sh ./train_plex_e_magi_option.sh /workspace/DeepSeek-Coder/data/$DATA_NAME $OUTPUT_DIR rust 5 $MODEL_NAME;

DATA_NAME=ise-uiuc/Magicoder-OSS-Instruct-75K
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base
OUTPUT_DIR=/workspace/DeepSeek-Coder/experiments-magi-lora-plex-e/${MODEL_NAME}/${DATA_NAME};
ts --gpus 1 sh ./train_plex_e_magi_option.sh /workspace/DeepSeek-Coder/data/$DATA_NAME $OUTPUT_DIR scala 6 $MODEL_NAME;
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


