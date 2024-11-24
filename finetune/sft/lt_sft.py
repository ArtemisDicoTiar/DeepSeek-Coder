import gc
import heapq
import json
import logging
import os
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep, time
from typing import Optional, Union, Dict, Any, List

import deepspeed
import numpy as np
import torch

from deepspeed.runtime import zero
from deepspeed.utils.debug import param_names
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from torch.distributed import broadcast

from tqdm import tqdm
from deepspeed import zero
from transformers import TrainerState, Trainer, enable_full_determinism
from transformers.trainer_utils import get_last_checkpoint, find_executable_batch_size
from transformers.utils import is_sagemaker_mp_enabled

import dask.array as da
from dask.diagnostics import ProgressBar
ProgressBar().register()

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
            print(f"Unfreezing {k} most changed params")

            # make a temp file to indicate that following script is running
            # use temp_dir
            raw_device_ids = os.environ["CUDA_VISIBLE_DEVICES"]
            print(f"Raw device ids: {raw_device_ids}")
            device_ids = raw_device_ids.split(",")
            device_ids = [int(d) for d in device_ids]
            target_device = min(device_ids)
            print(f"Target device: {target_device}")
            temp_dir = Path("/tmp")
            temp_file = temp_dir / "ltsft" / f"{raw_device_ids}.running"
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            if torch.distributed.get_rank() == target_device:
                print("Creating temp file")
                temp_file.touch()
            else:
                while not temp_file.exists():
                    sleep(1)

            with torch.no_grad():
                diffs = []
                for n, p in tqdm(
                    list(self.model.named_parameters()),
                    desc='Finding masking threshold',
                    disable=self.args.local_rank != target_device or self.args.disable_tqdm,
                ):
                    # p.grad = None # save some memory to use for the diff calculation
                    # if n in self.maskable_params:
                    if any([n in param_name for param_name in self.maskable_params]):
                        with zero.GatheredParameters(p, modifier_rank=target_device):
                            if torch.distributed.get_rank() == target_device:
                                delta = p - self._original_params[n].to(p.device)
                                delta = delta.view(-1)
                                self._mask[n] = self._mask[n].to(p.device)
                                valid_indices = (~self._mask[n]).view(-1)
                                valid_deltas = delta[valid_indices]
                                abs_deltas = torch.abs(valid_deltas)
                                diffs.extend(abs_deltas.tolist())

                if torch.distributed.get_rank() == target_device:
                    if k > len(diffs):
                        raise ValueError(
                            'Was requested to unfreeze {k} params, but only '
                            '{len(diffs)} are frozen.'
                        )
                    print(f'Found {len(diffs)} diffs')
                    print("Calculating threshold")

                    diffs_da = da.from_array(diffs, chunks="auto")

                    thresh = da.topk(diffs_da, k)[-1].compute()

                    # thresh = np.partition(diffs, - k)[-k]
                    print(f'Masking threshold = {thresh}')

                # # https://chatgpt.com/share/673db0ec-0534-8005-930d-bf7a3b915d0f
                # diffs = []
                # # ordered by large to small
                # for n, p in tqdm(
                #         list(self.model.named_parameters()),
                #         desc='Finding masking threshold',
                #         position=0,
                #         disable=self.args.local_rank != target_device or self.args.disable_tqdm,
                # ):
                #     if any(n in param_name for param_name in self.maskable_params):
                #         with zero.GatheredParameters(p, modifier_rank=target_device):
                #             if torch.distributed.get_rank() == target_device:
                #                 delta = p - self._original_params[n].to(p.device)
                #                 delta = delta.view(-1)
                #                 self._mask[n] = self._mask[n].to(p.device)
                #                 valid_indices = (~self._mask[n]).view(-1)
                #                 valid_deltas = delta[valid_indices]
                #                 abs_deltas = torch.abs(valid_deltas)
                #                 # reverse sort, large to small
                #                 sorted_abs_deltas = torch.sort(abs_deltas, descending=True).values
                #                 sorted_top_k_abs_deltas = sorted_abs_deltas[:k]
                #                 max_abs_delta = sorted_top_k_abs_deltas[0]
                #                 min_abs_delta = sorted_top_k_abs_deltas[-1]
                #
                #                 if len(diffs) > 0 and max_abs_delta < min(diffs):
                #                     continue
                #
                #                 if len(diffs) > 0 and min_abs_delta > max(diffs):
                #                     diffs = (sorted_top_k_abs_deltas + diffs)[:k]
                #                     continue
                #
                #                 for abs_delta in tqdm(sorted_top_k_abs_deltas.tolist(), desc='Finding top k index',
                #                                       position=1):
                #                     # if diff does not have size of k, add the value
                #                     if len(diffs) < k:
                #                         heapq.heappush(diffs, abs_delta)
                #                     else:
                #                         # if the value is smaller than the smallest value in diffs, break
                #                         if abs_delta < min(diffs):
                #                             break
                #
                #                         # if the value is larger than the largest value in diffs, replace it
                #                         if abs_delta > max(diffs):
                #                             heapq.heappushpop(diffs, abs_delta)
                #
                # if torch.distributed.get_rank() == target_device:
                #     if len(diffs) < k:
                #         raise ValueError(
                #             f'Was requested to unfreeze {k} params, but only '
                #             f'{len(diffs)} are frozen.'
                #         )
                #     print(f'Found {len(diffs)} diffs')
                #     print("Calculating threshold")
                #     thresh = diffs[-1]
                #     print(f'Masking threshold = {thresh}')

                    n_masked = 0
                    for n, p in tqdm(
                        list(self.model.named_parameters()),
                        desc='Updating masks',
                        disable=self.args.local_rank != target_device or self.args.disable_tqdm,
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
                        this_device_id = torch.distributed.get_rank()
                        if torch.distributed.get_rank() != target_device:
                            self._mask = torch.load(mask_file, map_location=f"cuda:{this_device_id}")
                            print("Mask loaded")
                            loading_status_file = temp_dir / "ltsft" / f"{this_device_id}.loaded"
                            loading_status_file.touch()

                        print("Temp file deleted")
                        break

                if torch.distributed.get_rank() == target_device:
                    while True:
                        # wait for all loading_status_file to be created
                        if len(list(temp_dir.glob("ltsft/*.loaded"))) == len(device_ids) - 1:
                            # delete all loading_status_file
                            for f in temp_dir.glob("ltsft/*.loaded"):
                                f.unlink()
                            print("All loaded files deleted")

                            # remove the mask file
                            mask_file.unlink()

                            break

        def _inner_post_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None,
                                      ignore_keys_for_eval=None):
            # Hack to fix: https://github.com/huggingface/transformers/issues/24558
            if self.args.auto_find_batch_size:
                self.model_wrapped = self.model
                self.deepspeed = None
            return super()._inner_training_loop(batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)

        def post_train(self,
                       resume_from_checkpoint: Optional[Union[str, bool]] = None,
                       trial: Union["optuna.Trial", Dict[str, Any]] = None,
                       ignore_keys_for_eval: Optional[List[str]] = None,
                       **kwargs, ):
            args = self.args
            inner_training_loop = find_executable_batch_size(
                self._inner_post_training_loop, self._train_batch_size, args.auto_find_batch_size
            )
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )

        def train(self, **kwargs):
            self.freeze()
            result = None

            for it in range(self.sft_args.n_ft_iterations):
                logger.info(f'Fine-tuning iteration {it + 1}')
                with torch.no_grad():
                    previous_params = {
                        n: torch.zeros_like(p, device='cpu').copy_(p)
                        for n, p in self.model.named_parameters()
                    }

                self.disable_masking()
                self.model_wrapped = self.model
                self.deepspeed = None
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
                super().train(**kwargs)

                print("Unfreezing..")
                self.unfreeze_k_most_changed_params(
                    self.n_tunable_params // self.sft_args.n_ft_iterations
                )

                with torch.no_grad():
                    for n, p in self.model.named_parameters():
                        p.copy_(previous_params[n])

                self.enable_masking()
                # self.optimizer = None
                # self.lr_scheduler = None
                self.set_training_len(
                    self.sft_args.sparse_ft_min_steps_per_iteration,
                    self.sft_args.sparse_ft_max_steps_per_iteration,
                    self.sft_args.sparse_ft_max_epochs_per_iteration,
                )

                self.model_wrapped = self.model
                self.deepspeed = None
                result = super().train(**kwargs)

            return result

        def save_sft(self, **kwargs):
            raw_device_ids = os.environ["CUDA_VISIBLE_DEVICES"]
            device_ids = raw_device_ids.split(",")
            device_ids = [int(d) for d in device_ids]
            target_device = min(device_ids)

            checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

            if torch.distributed.get_rank() == target_device:
                print("Saving the trainer SFT to disk.")
                self.sft().save(checkpoint_dir)
                print("Trainer SFT saved.")

    return _LotteryTicketSparseFineTuner
