import gc
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep

import deepspeed
import numpy as np
import torch

from deepspeed.runtime import zero
from deepspeed.utils.debug import param_names
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from torch.distributed import broadcast

from tqdm import tqdm
from deepspeed import zero


from .trainer import SparseFineTuner

logger = logging.getLogger(__name__)


def LotteryTicketSparseFineTuner(_Trainer):

    _SparseFineTuner = SparseFineTuner(_Trainer)

    class _LotteryTicketSparseFineTuner(_SparseFineTuner):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            logger.setLevel(self.args.get_process_log_level())
            if self.sft_args.ft_params_num is None:
                self.n_tunable_params = int(
                    self.sft_args.ft_params_proportion * self._num_maskable_params
                )
            else:
                self.n_tunable_params = self.sft_args.ft_params_num

        def unfreeze_k_most_changed_params(self, k):
            raw_device_ids = os.environ["CUDA_VISIBLE_DEVICES"]
            device_ids = raw_device_ids.split(",")
            device_ids = [int(d) for d in device_ids]
            target_device = min(device_ids)

            # make a temp file to indicate that following script is running
            # use temp_dir
            temp_dir = Path("/tmp")
            temp_file = temp_dir / "ltsft" / f"{raw_device_ids}.running"
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            if deepspeed.comm.get_rank() == target_device:
                temp_file.touch()
            else:
                sleep(5)

            with torch.no_grad():
                diffs = []
                for n, p in tqdm(
                    list(self.model.named_parameters()),
                    desc='Finding masking threshold',
                    disable=self.args.local_rank != target_device or self.args.disable_tqdm,
                ):
                    p.grad = None # save some memory to use for the diff calculation
                    # if n in self.maskable_params:
                    if any([n in param_name for param_name in self.maskable_params]):
                        with zero.GatheredParameters(p, modifier_rank=target_device):
                            if deepspeed.comm.get_rank() == target_device:
                                delta = p - self._original_params[n].to(p.device)
                                delta = delta.view(-1)
                                self._mask[n] = self._mask[n].to(p.device)
                                valid_indices = (~self._mask[n]).view(-1)
                                valid_deltas = delta[valid_indices]
                                abs_deltas = torch.abs(valid_deltas)
                                diffs.extend(abs_deltas.tolist())

                if deepspeed.comm.get_rank() == target_device:
                    if k > len(diffs):
                        raise ValueError(
                            'Was requested to unfreeze {k} params, but only '
                            '{len(diffs)} are frozen.'
                        )
                    print(f'Found {len(diffs)} diffs')
                    print("Calculating threshold")
                    diffs = np.partition(diffs, len(diffs) - k)
                    thresh = diffs[len(diffs) - k] + 1e-6
                    print(f'Masking threshold = {thresh}')

                    n_masked = 0
                    for n, p in tqdm(
                        list(self.model.named_parameters()),
                        desc='Updating masks',
                        disable=self.args.local_rank > 0 or self.args.disable_tqdm,
                    ):
                        if n in self.maskable_params:
                            abs_delta = (p - self._original_params[n].to(p.device)).abs()
                            to_mask = (abs_delta >= thresh) & (~self._mask[n])
                            self._mask[n] = to_mask | self._mask[n]
                            n_masked += to_mask.sum()

                    print(f'Masked {n_masked} params')

                mask_file = temp_dir / "ltsft" / f"{raw_device_ids}.mask.pkl"
                mask_file.parent.mkdir(parents=True, exist_ok=True)
                if deepspeed.comm.get_rank() == target_device:
                    # save the mask to pkl file
                    torch.save(self._mask, mask_file)
                    temp_file.unlink()

                while True:
                    # wait for the temp file to be deleted
                    if not temp_file.exists():
                        # load the mask from the file
                        this_device_id = deepspeed.comm.get_rank()
                        if this_device_id != target_device:
                            self._mask = torch.load(mask_file)
                            print("Mask loaded")
                            loading_status_file = temp_dir / "ltsft" / f"{this_device_id}.loaded"
                            loading_status_file.touch()

                        print("Temp file deleted")
                        break

                if deepspeed.comm.get_rank() == target_device:
                    while True:
                        # wait for all loading_status_file to be created
                        if len(list(temp_dir.glob("ltsft/*.loaded"))) == len(device_ids):
                            # delete all loading_status_file
                            for f in temp_dir.glob("ltsft/*.loaded"):
                                f.unlink()
                            print("All loaded files deleted")

                            # remove the mask file
                            mask_file.unlink()

                            break

        def train(self, **kwargs):
            self.freeze()
            result = None
            
            for it in range(self.sft_args.n_ft_iterations):
                logger.info(f'Fine-tuning iteration {it+1}')
                with torch.no_grad():
                    previous_params = {
                        n: torch.zeros_like(p, device='cpu').copy_(p)
                        for n, p in self.model.named_parameters()
                    }

                self.disable_masking()
                self.optimizer = None
                self.lr_scheduler = None
                self.set_training_len(
                    self.sft_args.full_ft_min_steps_per_iteration,
                    self.sft_args.full_ft_max_steps_per_iteration,
                    self.sft_args.full_ft_max_epochs_per_iteration,
                )
                if self.sft_args.full_ckpt_path:
                    print("Loading..")
                    self._load_from_checkpoint(self.sft_args.full_ckpt_path)
                    # self.model = load_state_dict_from_zero_checkpoint(self.self.sft_args.full_ckpt_path)
                    # self.model.to(self.args.device)
                    print("Loaded!")
                else:
                    super().train(**kwargs)

                self.unfreeze_k_most_changed_params(
                    self.n_tunable_params // self.sft_args.n_ft_iterations
                )
                
                with torch.no_grad():
                    for n, p in self.model.named_parameters():
                        p.copy_(previous_params[n])

                self.enable_masking()
                self.optimizer = None
                self.lr_scheduler = None
                self.set_training_len(
                    self.sft_args.sparse_ft_min_steps_per_iteration,
                    self.sft_args.sparse_ft_max_steps_per_iteration,
                    self.sft_args.sparse_ft_max_epochs_per_iteration,
                )
                result = super().train(**kwargs)
            
            return result

    return _LotteryTicketSparseFineTuner
