import random
from pathlib import Path

if __name__ == '__main__':
    """
    datasets: 
        rombodawg/MegaCodeTraining
        theblackcat102/evol-codealpaca-v1 
        ise-uiuc/Magicoder-OSS-Instruct-75K 
        ise-uiuc/Magicoder-Evol-Instruct-110K
        m-a-p/CodeFeedback-Filtered-Instruction
    """

    sample_size = 3_000

    total_dataset_dir = Path("/workspace/DeepSeek-Coder/data")
    for dataset in ['rombodawg/MegaCodeTraining', 'theblackcat102/evol-codealpaca-v1', 'ise-uiuc/Magicoder-OSS-Instruct-75K', 'ise-uiuc/Magicoder-Evol-Instruct-110K', 'm-a-p/CodeFeedback-Filtered-Instruction']:
        target_dataset_file = total_dataset_dir / dataset / "python.jsonl"
        # copy target dataset file to python-original.jsonl
        try:
            target_dataset_file.rename(total_dataset_dir / dataset / "python-original.jsonl")
        except FileNotFoundError:
            pass

        # undersample the dataset
        target_dataset_file = total_dataset_dir / dataset / "python-original.jsonl"
        save_dataset_file = total_dataset_dir / dataset / "python.jsonl"

        with target_dataset_file.open("r") as f:
            total_n_lines = sum(1 for _ in f)

        sampled_lines = random.sample(range(total_n_lines), sample_size)

        with (
            target_dataset_file.open("r") as fIN,
            save_dataset_file.open("w") as fOUT
        ):
            for i, line in enumerate(fIN):
                if i in sampled_lines:
                    fOUT.write(line)

