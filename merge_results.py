import argparse
import json
from pathlib import Path

import pandas as pd

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--target_dir", type=str, required=True)

    args = argparser.parse_args()
    target_dir = Path(args.target_dir)
    assert target_dir.exists(), f"{target_dir} does not exist"

    evaluation_files = list(target_dir.glob("evaluation*.json"))

    # Merge all results
    merged_results = []
    for evaluation_file in evaluation_files:
        with evaluation_file.open("r") as f:
            f_obj = json.load(f)
            f_obj.pop("config")
            eval_target = list(f_obj.keys())[0]
            eval_values = f_obj[eval_target]
            merged_results.append({
                "task-language": eval_target,
                **eval_values
            })

    merged_results_df = pd.DataFrame(merged_results)
    merged_results_df.sort_values(by="task-language", inplace=True)

    merged_results_df.to_csv(target_dir / "total_pass.tsv", index=False, sep="\t")



