import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.distributed
import transformers

from transformers import Trainer, get_linear_schedule_with_warmup
from datasets import load_dataset

from sora.sparse_optimizer import SparseAdamW
from sora.trainer import SparseTrainer
from sora.util import create_optimizer_and_scheduler, GATE_PARAM_NAME

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


@dataclass
class SparseArguments:
    sparse_lambda: Optional[float] = field(
        default=1e-3, metadata={"help": "loss penalty term for gate param"}
    )
    sparse_lambda_2: Optional[float] = field(
        default=1e-3, metadata={"help": "clipping scale for gate param"}
    )
    sparse_lr: Optional[float] = field(
        default=None,
        metadata={"help": "lr for gate parameter in sparse lora, default to same as learning rate for other parameters"}
    )
    lora_r: Optional[int] = field(
        default=16, metadata={"help": "matrix rank in lora"}
    )
    lora_alpha: Optional[int] = field(
        default=16, metadata={"help": "matrix rank in lora"}
    )
    lambda_schedule: Optional[str] = field(
        default=None, metadata={"help": "scheduling of lambda_2, {linear, log_linear}"}
    )
    max_lambda: Optional[float] = field(
        default=10, metadata={"help": "maximum value of lambda_2 in scheduling"}
    )
    lambda_num: Optional[int] = field(
        default=10, metadata={"help": "total number of lambdas in scheduling"}
    )


@dataclass
class SparseTrainingArguments:
    train_sparse: Optional[bool] = field(
        default=False, metadata={"help": "whether use sparse lora"}
    )
    debug_mode: Optional[bool] = field(
        default=False, metadata={"help": "debug mode"}
    )


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
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, SparseTrainingArguments, SparseArguments))
    model_args, data_args, training_args, sparse_training_args, sparse_args = parser.parse_args_into_dataclasses()

    # ============ sora ============ #
    if sparse_training_args.train_sparse:
        if sparse_args.sparse_lr is None:
            sparse_args.sparse_lr = sparse_training_args.learning_rate

        if sparse_training_args.debug_mode:
            training_args.output_dir += "-debug"
        print(f"save model to {training_args.output_dir}")
    # ================================ #

    if training_args.local_rank == 0:
        print('=' * 100)
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

    if sparse_training_args.train_sparse:
        print("loading from src.lora")
        from sora.lora import LoraModel, LoraConfig
    else:
        from opendelta.delta_models import LoraModel, LoraConfig

    lora_config = {
        # 'modified_modules': ['c_attn', 'self_attn.q_proj', 'c_fc', 'c_proj'],
        'modified_modules': ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj",
                             "mlp.down_proj", "self_attn.o_proj"],
        "lora_r": sparse_args.lora_r,
        "lora_alpha": sparse_args.lora_alpha
    }
    lora_config = LoraConfig.from_dict(lora_config)
    delta_model = LoraModel.from_config(lora_config, backbone_model=model)
    delta_model.freeze_module(set_state_dict=True)
    delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=False)

    # if training_args.local_rank == 0:
    #     delta_model.print_trainable_parameters()
    #     print("Load model from {} over.".format(model_args.model_name_or_path))

    raw_train_datasets = load_dataset(
        'json',
        data_files=data_args.data_path,
        split="train",
        cache_dir=training_args.cache_dir
    )
    # if training_args.local_rank > 0:
    #     torch.distributed.barrier()

    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,  # not args.overwrite_cache
        desc="Running Encoding",
        fn_kwargs={"tokenizer": tokenizer}
    )

    # if training_args.local_rank == 0:
    #     torch.distributed.barrier()

    # if training_args.local_rank == 0:
    print("Training dataset samples:", len(train_dataset))
    for index in random.sample(range(len(train_dataset)), 3):
        print(
            f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
        print(f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    optimizer, lr_scheduler = create_optimizer_and_scheduler(training_args, model, num_training_steps=int(
        training_args.num_train_epochs * (len(train_dataset) / training_args.train_batch_size)))
    sparse_optimizer = None
    sparse_scheduler = None
    if sparse_training_args.train_sparse:
        print("building sparse optimizer and scheduler")

        valid_param_name = []
        for n, p in model.named_parameters():
            print(n)
            if GATE_PARAM_NAME in n:
                valid_param_name.append(n)
        print("valid param name:", valid_param_name)
        sparse_optimizer = SparseAdamW(sparse_lambda=sparse_args.sparse_lambda_2,
                                       lambda_schedule=sparse_args.lambda_schedule, max_lambda=sparse_args.max_lambda,
                                       lambda_num=sparse_args.lambda_num,
                                       params=[p for n, p in model.named_parameters() if
                                               GATE_PARAM_NAME in n and p.requires_grad], lr=sparse_args.sparse_lr)
        sparse_scheduler = get_linear_schedule_with_warmup(sparse_optimizer,
                                                           num_warmup_steps=int(training_args.num_train_epochs * (
                                                                   len(train_dataset) / training_args.train_batch_size) * training_args.warmup_ratio),
                                                           num_training_steps=int(training_args.num_train_epochs * (
                                                                   len(train_dataset) / training_args.train_batch_size)))

    # trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    class SparseDeepspeedTrainer(SparseTrainer):
        def create_optimizer(self):
            self.optimizer = SparseAdamW(sparse_lambda=sparse_args.sparse_lambda_2,
                                         lambda_schedule=sparse_args.lambda_schedule, max_lambda=sparse_args.max_lambda,
                                         lambda_num=sparse_args.lambda_num,
                                         params=[p for n, p in model.named_parameters() if
                                                 GATE_PARAM_NAME in n and p.requires_grad], lr=sparse_args.sparse_lr)
            return self.optimizer

        def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
            if optimizer is None:
                optimizer = self.optimizer

            self.lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                                num_warmup_steps=int(training_args.num_train_epochs * (
                                                                        len(train_dataset) / training_args.train_batch_size) * training_args.warmup_ratio),
                                                                num_training_steps=int(
                                                                    training_args.num_train_epochs * (
                                                                            len(train_dataset) / training_args.train_batch_size)))
            return self.lr_scheduler

    trainer = SparseTrainer(
    # trainer = SparseDeepspeedTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
        optimizers=(optimizer, lr_scheduler),
        sparse_lambda=sparse_args.sparse_lambda,
        sparse_optimizer=(sparse_optimizer, sparse_scheduler),
        preprocess_logits_for_metrics=None,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics
        # if training_args.do_eval and not is_torch_tpu_available()
        # else None,
    )

    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
