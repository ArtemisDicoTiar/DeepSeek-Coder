source ~/.zshrc
va bigcode-evaluation-harness

EXPERIMENT_DIR=$1;
DATASET=$2;
MODEL_NAME=$3;
EXPERIMENT_NAME=$4;

echo UPLOADING ${EXPERIMENT_DIR} to
echo DicoTiar/${MODEL_NAME}-${EXPERIMENT_NAME}-${DATASET}

huggingface-cli upload-large-folder \
  --repo-type=dataset \
  DicoTiar/${MODEL_NAME}-${EXPERIMENT_NAME}-${DATASET} \
  ${EXPERIMENT_DIR}