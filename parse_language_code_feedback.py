import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

pattern = r'^```(\w+)\s*\n'

if __name__ == '__main__':
    """
    Parse the dataset to extract the language of the code snippet
    
    datasets: 
        
        m-a-p/Code-Feedback
    splits:
        train
        train
        train
        train
    i:
        USER ASSISTANT
        instruction output
        problem solution
        instruction response
    
    DATASETS=("rombodawg/MegaCodeTraining" "theblackcat102/evol-codealpaca-v1" "ise-uiuc/Magicoder-OSS-Instruct-75K" "ise-uiuc/Magicoder-Evol-Instruct-110K");
    INPUTS=(USER instruction problem instruction);
    OUTPUTS=(ASSISTANT output solution response);
    for i in {1..4}; do
        echo "Parsing ${DATASETS[$i]}";
        python3 parse_language.py \
            --dataset=${DATASETS[$i]} \
            --split=train \
            --input_key=${INPUTS[$i]} \
            --output_key=${OUTPUTS[$i]} \
            --output_dir=data/${DATASETS[$i]}
    done
    
    """
    argparser = argparse.ArgumentParser()
    # argparser.add_argument('--languages', nargs='+', type=str, help='Languages to parse')
    argparser.add_argument("--dataset", type=str, help="Dataset to parse")
    argparser.add_argument("--split", type=str, help="Split to parse")
    argparser.add_argument("--input_key", type=str, help="Key to parse in the dataset")
    argparser.add_argument("--output_key", type=str, help="Key to save in the parsed dataset")
    argparser.add_argument("--output_dir", type=str, help="Output path to save the parsed dataset")
    args = argparser.parse_args()

    # languages = args.languages
    dataset = args.dataset
    split = args.split
    input_key = args.input_key
    output_key = args.output_key
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("====================================")
    print(f'Parsing dataset {dataset} \nsaving to {output_dir}')
    print("====================================")
    print()

    # load dataset
    dataset_object = load_dataset(dataset, split=split) if split else load_dataset(dataset)

    # output should be instruction, output
    for row in tqdm(dataset_object):
        instruction = row[input_key]
        output = row[output_key]

        result = re.findall(pattern, output, re.DOTALL | re.MULTILINE)
        extracted_language = result[0].lower() if result else None
        save_obj = {
            'instruction': instruction,
            'output': output,
        }
        with (output_dir / f'{extracted_language}.jsonl').open('a') as f:
            f.write(json.dumps(save_obj) + '\n')

    # stats
    print(output_dir.lstat())
    # print each file line count
    for file in output_dir.iterdir():
        print(f'{file.name}: {sum(1 for _ in open(file))} lines')

