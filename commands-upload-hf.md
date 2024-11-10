

# Model

[//]: # (cpp)
EXPERIMENT_NAME=lora
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base
DATASET_USER=ise-uiuc
DATASET=Magicoder-OSS-Instruct-75K
EXPERIMENT_DIR=/workspace/DeepSeek-Coder/experiments-magi-lora/${MODEL_NAME}/${DATASET_USER}/${DATASET}
for LANGUAGE in go php rust scala swift cpp python java; do
    ts sh ./upload_model.sh $EXPERIMENT_DIR $DATASET $LANGUAGE $MODEL_NAME $EXPERIMENT_NAME
done

# Experiment Results
EXPERIMENT_NAME=baseline
MODEL_NAME=deepseek-coder-6.7b-base
DATASET_USER=ise-uiuc
DATASET=Magicoder-OSS-Instruct-75K
EXPERIMENT_DIR=/workspace/DeepSeek-Coder/experiments/${MODEL_NAME}/baseline
ts sh ./upload_dataset.sh $EXPERIMENT_DIR $DATASET $MODEL_NAME $EXPERIMENT_NAME


EXPERIMENT_NAME=lora
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base
DATASET_USER=ise-uiuc
DATASET=Magicoder-OSS-Instruct-75K
EXPERIMENT_DIR=/workspace/DeepSeek-Coder/experiments-magi-lora/${MODEL_NAME}/${DATASET_USER}/${DATASET}/results
ts sh ./upload_dataset.sh $EXPERIMENT_DIR $DATASET $MODEL_NAME $EXPERIMENT_NAME