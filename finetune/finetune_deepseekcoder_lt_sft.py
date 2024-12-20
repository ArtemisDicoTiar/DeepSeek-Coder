import copy
import os
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import deepspeed
import torch
import torch.distributed
import transformers
from deepspeed.runtime import zero
from transformers import Trainer, AdamW
from datasets import load_dataset

from sft import SftArguments, LotteryTicketSparseFineTuner

IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"

def build_instruction_prompt(instruction: str):
    return '''
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{}
### Response:
'''.format(instruction.strip()).lstrip()

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="deepseek-ai/deepseek-coder-6.7b-instruct")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

    # mlm_probability: float = field(
    #     default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    # )

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer):
    sources = [
        build_instruction_prompt(instruction)
        for instruction in examples['instruction']
    ]
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples['output']]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, SftArguments))
    model_args, data_args, training_args, sft_args = parser.parse_args_into_dataclasses()
    
    if training_args.local_rank == 0:
        print('='*100)
        print(training_args)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )

    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    if training_args.local_rank == 0:
        print("Load tokenizer from {} over.".format(model_args.model_name_or_path))

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16
    )

    if training_args.local_rank == 0:
        print("Load model from {} over.".format(model_args.model_name_or_path))


    raw_train_datasets = load_dataset(
        'json',
        data_files=data_args.data_path,
        split="train",
        cache_dir=training_args.cache_dir
    )
    if training_args.local_rank > 0: 
        torch.distributed.barrier()
        
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True, # not args.overwrite_cache
        desc="Running Encoding",
        fn_kwargs={ "tokenizer": tokenizer }
    )

    if training_args.local_rank == 0:
        torch.distributed.barrier()
    
    if training_args.local_rank == 0:
        print("Training dataset samples:", len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            print(f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
            print(f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    # ======================== LT-SFT ======================== #
    raw_device_ids = os.environ["CUDA_VISIBLE_DEVICES"]
    print(f"Raw device ids: {raw_device_ids}")
    device_ids = raw_device_ids.split(",")
    device_ids = [int(d) for d in device_ids]
    target_device = min(device_ids)
    print(f"Target device: {target_device}")

    # if sft_args.freeze_layer_norm:
    #     for n, p in model.named_parameters():
    #         if 'layernorm' in n.lower():
    #             p.requires_grad = False

    maskable_param_nums = 0
    total_params = 0
    trainable_params = 0
    maskable_trainable_param_nums = 0
    maskable_params = []

    """Deepseek Coder model structure:
    model.embed_tokens.weight

    model.layers.20
    model.layers.20.self_attn
    model.layers.20.self_attn.q_proj
    model.layers.20.self_attn.k_proj
    model.layers.20.self_attn.v_proj
    model.layers.20.self_attn.o_proj
    model.layers.20.self_attn.rotary_emb
    model.layers.20.mlp
    model.layers.20.mlp.gate_proj
    model.layers.20.mlp.up_proj
    model.layers.20.mlp.down_proj
    model.layers.20.mlp.act_fn
    model.layers.20.input_layernorm
    model.layers.20.post_attention_layernorm
    
    model.norm.weight
    lm_head.weight
    """

    for n, p in model.named_parameters():
        with zero.GatheredParameters(p, modifier_rank=target_device):
            model.lm_head.weight = torch.nn.Parameter(
                model.lm_head.weight.clone().detach().requires_grad_(False)
            )

            attn_layers = ['self_attn', 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']
            ffn_layers = ['mlp', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
            if 'layernorm' in n.lower():
                if sft_args.freeze_layer_norm:
                    p.requires_grad = False

            if sft_args.freeze_all:
                if any([name in n for name in attn_layers]):
                    if sft_args.unfreeze_attn:
                        # p.requires_grad = True
                        pass

                elif any([name in n for name in ffn_layers]):
                    if sft_args.unfreeze_ffn:
                        # p.requires_grad = True
                        pass
                else:
                    if any([name not in n for name in attn_layers + ffn_layers]):
                        p.requires_grad = False


    # if sft_args.freeze_all:
    #     for n, p in model.named_parameters():
    #         p.requires_grad = False
    #
    # if sft_args.unfreeze_attn:
    #     attn_layers = ['self_attn', 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
    #                    'self_attn.o_proj']
    #     for n, p in model.named_parameters():
    #         if any([name in n for name in attn_layers]):
    #             p.requires_grad = True
    #
    # if sft_args.unfreeze_ffn:
    #     ffn_layers = ['mlp', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
    #     for n, p in model.named_parameters():
    #         if any([name in n for name in ffn_layers]):
    #             p.requires_grad = True

    for n, p in model.named_parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

        if n.startswith(model.base_model_prefix) and p.requires_grad:
            maskable_params.append(n)
            maskable_param_nums += p.numel()
            if p.requires_grad:
                maskable_trainable_param_nums += p.numel()
        else:
            print(n)
            # pass

    print(f'Maskable params: {maskable_params}')

    print(f'Total params: {total_params}')
    print(f'Trainable params: {trainable_params}')
    print(f'Maskable params: {maskable_param_nums}')
    print(f'Maskable trainable params: {maskable_trainable_param_nums}')

    # Initialize our Trainer
    trainer_cls = Trainer
    wrapper_cls = LotteryTicketSparseFineTuner
    trainer_cls = wrapper_cls(trainer_cls)

    # ======================================================== #
    # trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
    # # Create an optimizer with only the trainable parameters
    # optimizer = AdamW(trainable_params)

    trainer = trainer_cls(
        sft_args=sft_args,
        maskable_params=maskable_params,
        compute_metrics=None,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    trainer.train()
    trainer.save_state()
    trainer.save_sft()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
