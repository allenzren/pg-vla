from typing import List

import torch


def get_action_accuracy(
    gt: torch.FloatTensor,  # [Batch_Size, Horizon, Action_Dim]
    pred: torch.FloatTensor,
    thresholds: List[float] = [0.1, 0.2],
) -> torch.FloatTensor:
    diff = torch.abs(gt - pred).reshape(-1, gt.shape[-1])

    # get the percentage of diff lower than threshold for all action dimensions
    accuracies = torch.zeros(len(thresholds))
    for idx, threshold in enumerate(thresholds):
        accuracy = torch.mean(
            (torch.mean((diff < threshold).float(), dim=1) >= 1.0).float()
        )
        accuracies[idx] = accuracy
    return accuracies
