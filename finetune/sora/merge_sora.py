from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, HfArgumentParser, AutoTokenizer
import torch

@dataclass
class Args:
    model_name_or_path: str = field(default='bigcode/gpt_bigcode-santacoder')
    sora_path: str = field(default="")
    output_path: str = field(default="")

parser = HfArgumentParser((Args))
args = parser.parse_args_into_dataclasses()[0]

sora_weights = torch.load(args.sora_path)
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).cuda()
param = next(model.parameters())
new_weights = {}

A = B = gate = None
for lora_key, lora_tensor in sora_weights.items():
    if 'lora_A' in lora_key:
        A = lora_tensor.to(param)
    elif 'lora_B' in lora_key:
        B = lora_tensor.to(param)
    elif 'gate' in lora_key:
        gate = lora_tensor.to(param)

    if A is not None and B is not None and gate is not None:
        original_name = lora_key.split('.lora.')[0]
        model.get_parameter(original_name + '.weight').data += B.mul(gate).mm(A)
        mul_A = A.T.mul(gate).T
        nonzero_row_mul_A = mul_A[mul_A.sum(dim=1) != 0]
        new_weights[original_name + '.lora.lora_A'] = nonzero_row_mul_A

        mul_B = B.mul(gate)
        nonzero_col_mul_B = mul_B[:, mul_B.sum(dim=0) != 0]
        new_weights[original_name + '.lora.lora_B'] = nonzero_col_mul_B
        A = B = gate = None

model.save_pretrained(args.output_path)
torch.save(new_weights, args.output_path + '/lora_weights.bin')
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
tokenizer.save_pretrained(args.output_path)
