## How to Fine-tune DeepSeek-Coder

We provide script `finetune_deepseekcoder.py` for users to finetune our models on downstream tasks.

The script supports the training with [DeepSpeed](https://github.com/microsoft/DeepSpeed). You need install required packages by:

```bash
pip install -r requirements.txt
```

Please follow [Sample Dataset Format](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1) to prepare your training data.
Each line is a json-serialized string with two required fields `instruction` and `output`.

After data preparation, you can use the sample shell script to finetune `deepseek-ai/deepseek-coder-6.7b-instruct`. 
Remember to specify `DATA_PATH`, `OUTPUT_PATH`.
And please choose appropriate hyper-parameters(e.g., `learning_rate`, `per_device_train_batch_size`) according to your scenario.

```bash
DATA_PATH="<your_data_path>"
OUTPUT_PATH="<your_output_path>"
MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-instruct"

deepspeed finetune_deepseekcoder.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 3 \
    --model_max_length 1024 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --deepspeed configs/ds_config_zero3.json \
    --bf16 True
```


# PLEX commands

## For SFT
### Train LT-SFT

Note that all the custom layers (such as `c_attn`, `mlp.c_fc`) in `run_clm.py` and `run_clm_peft.py` must be changed properly to be used in models other than GPTBigCode.

```bash
export train_file="data/java_concat.json"
export output_dir="output"
export params_num=33739038 # about 3% of the total number of SantaCoder-1.1B parameters
python finetune_deepseekcoder_lt_sft.py \
    --model_name_or_path bigcode``/gpt_bigcode-santacoder --max_eval_samples 10000 --report_to none \
    --train_file ${train_file} \
    --output_dir ${output_dir} \
    --do_train \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --overwrite_output_dir \
    --ft_params_num ${params_num} --mask_embed=True \
    --freeze_layer_norm \
    --full_l1_reg 0.0 \
    --sparse_l1_reg 0.0 \
    --learning_rate 2e-5 \
    --full_ft_max_epochs_per_iteration 3 \
    --sparse_ft_max_epochs_per_iteration 3 \
    --evaluation_strategy no \
    --eval_steps 100000000 \
    --save_strategy epoch \
    --fp16 \
    --validation_split_percentage 0 \
    --save_total_limit 2 --report_to none
```

### Train PLEX
Just add `--freeze_all --unfreeze_attn --unfreeze_ffn ` to the above command.

## For Lora
### Train LoRA

To train LoRA, use the following command:
```bash
export peft_output=$PEFT_OUTPUT
export alpha=1024

python examples/pytorch/language-modeling/run_clm_peft.py \
--model_name_or_path bigcode/gpt_bigcode-santacoder \
--train_file $train_file \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 8 \
--do_train \
--warmup_ratio=0.1 \
--save_total_limit 2 --learning_rate 5e-5 \
--output_dir $peft_output \
--overwrite_output_dir --report_to=none --peft_lora_r=$alpha --peft_lora_alpha=$alpha
```

### Train PLEX-E
First, merge the LoRA weight

```
export merged_output=MERGE_PATH
python merge_peft.py bigcode/gpt_bigcode-santacoder $peft_output $merged_output
```

Then, Just add `--full_ckpt_path=$merged_output ` to the above command.

