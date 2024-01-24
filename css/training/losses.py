from itertools import permutations
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class PitWrapper(nn.Module):
    """
    Permutation Invariant Wrapper to allow Permutation Invariant Training
    (PIT) with existing losses.
    Permutation invariance is calculated over the sources axis which is
    assumed to be the rightmost dimension.
    Predictions and targets tensors are assumed to have shape [batch, ..., sources].
    """

    def __init__(self, base_loss: Callable):
        """
        Args:
            base_loss (callable):
                Base loss function, e.g. torch.nn.MSELoss. It is assumed that it takes
                two arguments:
                predictions and targets and no reduction is performed.
                (if a pytorch loss is used, the user must specify reduction="none").
        """
        super(PitWrapper, self).__init__()
        self.base_loss = base_loss

    def _fast_pit(self, loss_mat):
        """
        Args:
            loss_mat : Tensor of shape [sources, source] containing loss values for each
            prediction-target pair.

        Returns:
            loss : torch.Tensor, scalar, minimum loss over all permutations.
            target_perm (list) : Optimial permutation i.e. loss(predictions, targets[:, target_perm])
                returns the minimum loss.
        """
        left_inds, right_inds = linear_sum_assignment(loss_mat.data.cpu())
        assert (left_inds == range(len(left_inds))).all()
        target_perm = right_inds
        loss = loss_mat[left_inds, right_inds].mean()

        return loss, target_perm

    def _opt_perm_loss(self, pred, target):
        n_sources = target.shape[-1]

        # [..., sources] -> [..., sources, sources] replicated along second to last dim
        ones = [1] * (len(target.shape) - 1)
        target = target.unsqueeze(-2).repeat(*ones, n_sources, 1)

        # [..., sources] -> [..., sources, sources] replicated along last dim
        ones = [1] * (len(pred.shape) - 1)
        pred = pred.unsqueeze(-1).repeat(1, *ones, n_sources)

        # Average over time and freq dims. Do not reduce over the last two dims.
        loss_mat = self.base_loss(pred, target).mean(dim=(0, 1))

        assert (
            len(loss_mat.shape) >= 2 and loss_mat.shape[-2:] == target.shape[-2:]
        ), "Base loss should not perform any reduction operation"
        mean_over = tuple(range(loss_mat.dim() - 2))  # all but the last two dims
        if mean_over:
            loss_mat = loss_mat.mean(dim=mean_over)
        # loss_mat: [sources, sources]

        return self._fast_pit(loss_mat)

    def forward(self, preds, targets):
        """
        Args:
            preds: predictions tensor, of shape [batch, ..., sources].
            targets: targets tensor, of shape [batch, ..., sources].

        Returns:
            -------
            loss : Permutation invariant loss per instance, tensor of shape [batch].
            perms (list) :
                List of indexes for optimal permutation of the targets per instance.
                Example: [(0, 1, 2), (2, 1, 0)] for three sources and batch size 2.
        """
        losses = []
        perms = []

        assert preds.shape[-1] == targets.shape[-1], \
            "preds and targets expected to be padded to the same number of sources"

        for pred, label in zip(preds, targets):
            loss, p = self._opt_perm_loss(pred, label)
            perms.append(p)
            losses.append(loss)
        loss = torch.stack(losses)
        return loss, perms


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes MSE loss without any reduction."""
    return F.mse_loss(pred, target, reduction="none")


def test_pit_wrapper():
    pit_mse = PitWrapper(mse_loss)

    for i in range(20):
        # (batch, time, freq, sources)
        targets = torch.rand((2, 100, 257, 4))
        p = (3, 0, 2, 1)
        predictions = targets[..., p]
        loss, target_perm = pit_mse(predictions, targets)

        assert (loss == 0.).all()
        assert (predictions[0] == targets[0,..., target_perm[0]]).all()
        assert (np.stack(target_perm) == np.stack([p, p])).all()


if __name__ == "__main__":
    test_pit_wrapper()