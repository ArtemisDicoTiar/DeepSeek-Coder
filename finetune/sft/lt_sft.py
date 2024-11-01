import logging
import os

import numpy as np
import torch
from tqdm import tqdm

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
            with torch.no_grad():
                diffs = []
                for n, p in tqdm(
                    list(self.model.named_parameters()),
                    desc='Finding masking threshold',
                    disable=self.args.local_rank > 0 or self.args.disable_tqdm,
                ):
                    p.grad = None # save some memory to use for the diff calculation
                    if n in self.maskable_params:
                        delta = p - self._original_params[n].to(p.device)
                        delta = delta.view(-1)
                        self._mask[n] = self._mask[n].to(p.device)
                        valid_indices = (~self._mask[n]).view(-1)
                        valid_deltas = delta[valid_indices]
                        abs_deltas = torch.abs(valid_deltas)
                        diffs.extend(abs_deltas.tolist())
                
                if k > len(diffs):
                    raise ValueError(
                        'Was requested to unfreeze {k} params, but only '
                        '{len(diffs)} are frozen.'
                    )
                diffs = np.partition(diffs, len(diffs) - k)
                thresh = diffs[len(diffs) - k]
                logger.info(f'Masking threshold = {thresh}')
                
                # n_masked = 0
                for n, p in tqdm(
                    list(self.model.named_parameters()),
                    desc='Updating masks',
                    disable=self.args.local_rank > 0 or self.args.disable_tqdm,
                ):
                    if n in self.maskable_params:
                        abs_delta = (p - self._original_params[n].to(p.device)).abs()
                        to_mask = (abs_delta >= thresh) & (~self._mask[n])
                        self._mask[n] = to_mask | self._mask[n]
                        # n_masked += to_mask.sum()

                # logger.info(f'Masked {n_masked} params')

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
