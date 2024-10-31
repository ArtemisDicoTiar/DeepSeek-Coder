

# Model

[//]: # (cpp)

EXPERIMENT_DIR=/workspace/DeepSeek-Coder/experiments/deepseek-coder-6.7b-instruct/ise-uiuc/Magicoder-OSS-Instruct-75K
DATASET=Magicoder-OSS-Instruct-75K
for LANGUAGE in go php rust scala swift cpp; do
    ts sh ./upload_model.sh $EXPERIMENT_DIR $DATASET $LANGUAGE
done

# Experiment Results
EXPERIMENT_NAME=baseline
MODEL_NAME=deepseek-coder-6.7b-base
DATASET_USER=ise-uiuc
DATASET=Magicoder-OSS-Instruct-75K
EXPERIMENT_DIR=/workspace/DeepSeek-Coder/experiments/${MODEL_NAME}/baseline
ts sh ./upload_dataset.sh $EXPERIMENT_DIR $DATASET $MODEL_NAME $EXPERIMENT_NAME


EXPERIMENT_NAME=baseline
MODEL_NAME=deepseek-coder-6.7b-base
DATASET_USER=ise-uiuc
DATASET=Magicoder-OSS-Instruct-75K
EXPERIMENT_DIR=/workspace/DeepSeek-Coder/experiments/${MODEL_NAME}/${DATASET_USER}/${DATASET}/results
ts sh ./upload_dataset.sh $EXPERIMENT_DIR $DATASET $MODEL_NAME $EXPERIMENT_NAME