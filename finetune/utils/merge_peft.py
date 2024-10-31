import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    sys.argv[1],
)
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])

from peft import PeftModel  # dynamic import to avoid dependency on peft
model = PeftModel.from_pretrained(model, sys.argv[2])
print("Loaded PEFT model. Merging...")
model = model.merge_and_unload()
model.resize_token_embeddings(len(tokenizer))

model.save_pretrained(sys.argv[3])
tokenizer.save_pretrained(sys.argv[3])