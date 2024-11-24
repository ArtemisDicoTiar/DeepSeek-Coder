import argparse
import sys

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == '__main__':

    """
SUBMIT_LANGUAGES=(java php cpp swift go rust scala python);
for LANGUAGE in ${SUBMIT_LANGUAGES[@]}; do
DATA_NAME=ise-uiuc/Magicoder-OSS-Instruct-75K
MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-base

PEFT_DIR=/data/DeepSeek-Coder/experiments-magi-lora/${MODEL_NAME}/${DATA_NAME}/${LANGUAGE}
largest_checkpoint=$(ls -d ${PEFT_DIR}/checkpoint-* 2>/dev/null | grep -oP 'checkpoint-\K[0-9]+' | sort -nr | head -n 1)
EXACT_PEFT_DIR=${PEFT_DIR}/checkpoint-${largest_checkpoint}

OUTPUT_DIR=/data/DeepSeek-Coder/experiments-magi-lora-merged/${MODEL_NAME}/${DATA_NAME}/${LANGUAGE}/checkpoint-${largest_checkpoint};
mkdir -p OUTPUT_DIR;

ts --gpus 2 sh ./merge_peft.sh ${MODEL_NAME} ${EXACT_PEFT_DIR} ${OUTPUT_DIR}
done;
    """

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True)
    argparser.add_argument("--peft", type=str, required=True)
    argparser.add_argument("--output", type=str, required=True)
    argparser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        help="Model precision, from: fp32, fp16 or bf16",
    )

    argparser.add_argument(
        "--revision",
        default=None,
        help="Model revision to use",
    )
    argparser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Use the token generated when running `huggingface-cli login` (necessary for private model).",
    )
    argparser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Use a model with custom code, this requires executing code by the author of the model.",
    )
    args = argparser.parse_args()

    print(args)

    accelerator = Accelerator()

    # here we generate code and save it (evaluation is optional but True by default)
    dict_precisions = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if args.precision not in dict_precisions:
        raise ValueError(
            f"Non valid precision {args.precision}, choose from: fp16, fp32, bf16"
        )

    model_kwargs = {
        "revision": args.revision,
        "trust_remote_code": args.trust_remote_code,
        "token": args.use_auth_token,
        # "attn_implementation": "flash_attention_2"
    }
    print(f"Loading model in {args.precision}")
    model_kwargs["torch_dtype"] = dict_precisions[args.precision]

    model_kwargs["device_map"] = "auto"
    print("Loading model in auto mode")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kwargs,
    )

    from peft import PeftModel  # dynamic import to avoid dependency on peft

    model = PeftModel.from_pretrained(model, args.peft)
    print("Loaded PEFT model. Merging...")
    merged_model = model.merge_and_unload()
    print("Merge complete.")

    merged_model.save_pretrained(args.output)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.save_pretrained(args.output)