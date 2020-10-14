import torch
import numpy as np
from typing import List, Union


def exact_match_rate(
        real_start: Union[List[List[torch.Tensor]], torch.Tensor],
        real_end: Union[List[List[torch.Tensor]], torch.Tensor],
        pred_start: torch.Tensor,
        pred_end: torch.Tensor
):
    assert real_start.shape == real_end.shape, "real_start and real_end shapes do not match."
    assert len(pred_start) == len(pred_end), "pred_start and pred_end lengths do not match."
    assert len(real_start) == len(pred_start), \
        f"Datasets mismatch: {len(real_start)} correct labels and {len(pred_start)} predictions were provided."

    correct = 0
    total_indices = len(pred_start) + len(pred_end)
    for i, (pred_start_sample, pred_end_sample) in enumerate(zip(pred_start, pred_end)):
        '''The list below list will store how many correct predictions (start+end) the algorithm makes for every
        correct possible answer. E.g. if there are 3 possible correct answers, and the algorithm predicts start+end 
        correctly for the first answer, only correct start of the second possible answer, and not the third 
        possible answer the list will be [2, 1, 0]. We'll take take the max (in this case 2), as any correct possible
        answer means our model made a correct prediction.'''
        match_options = []
        # each sample might have j correct possible answers
        for real_start_sample, real_end_sample in zip(real_start[i], real_end[i]):
            matches = 0
            if pred_start_sample == real_start_sample:
                matches += 1
            if pred_end_sample == real_end_sample:
                matches += 1
            match_options.append(matches)
            correct += max(match_options
    match_rate = correct / total_indices
    return correct, total_indices, match_rate


def f1_score(
        real_start: Union[List[List[torch.Tensor]], torch.Tensor],
        real_end: Union[List[List[torch.Tensor]], torch.Tensor],
        pred_start: torch.Tensor,
        pred_end: torch.Tensor
):
    all_f1 = []
    for i, (pred_start_sample, pred_end_sample) in enumerate(zip(pred_start, pred_end)):
        '''The list below list will store how many correct predictions (start+end) the algorithm makes for every
        correct possible answer. E.g. if there are 3 possible correct answers, and the algorithm predicts start+end 
        correctly for the first answer, only correct start of the second possible answer, and not the third 
        possible answer the list will be [2, 1, 0]. We'll take take the max (in this case 2), as any correct possible
        answer means our model made a correct prediction.'''

        pred_indices = set(range(pred_start_sample, pred_end_sample + 1))

        f1_options = []
        # each sample might have j correct possible answers
        for real_start_sample, real_end_sample in zip(real_start[i], real_end[i]):

            real_indices = set(range(real_start_sample, real_end_sample + 1))  # consider adding int() around tensors
            num_same_tokens = real_indices.intersection(pred_indices)

            precision = num_same_tokens / len(pred_indices)
            recall = num_same_tokens / len(real_indices)
            f1_sample = (2 * precision * recall) / (precision + recall)
            f1_options.append(f1_sample)
        all_f1.append(max(f1_options))

        average_f1 = np.mean(all_f1)
        return all_f1, average_f1


