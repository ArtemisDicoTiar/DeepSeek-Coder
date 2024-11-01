source ~/.zshrc
va bigcode-evaluation-harness

EXPERIMENT_DIR=$1;
DATASET=$2;
LANGUAGE=$3;
MODEL_NAME=$4;
EXPERIMENT_NAME=$5;

echo UPLOADING ${EXPERIMENT_DIR}/${LANGUAGE} to
echo DicoTiar/${MODEL_NAME}-${EXPERIMENT_NAME}-${DATASET}-${LANGUAGE}

huggingface-cli upload-large-folder \
  --private \
  --repo-type=model \
  DicoTiar/${MODEL_NAME}-${EXPERIMENT_NAME}-${DATASET}-${LANGUAGE} \
  ${EXPERIMENT_DIR}/${LANGUAGE}
