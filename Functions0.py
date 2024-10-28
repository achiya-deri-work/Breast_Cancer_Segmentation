import torch
import numpy as np

def compute_accuracy(predicted, target):
    # Compute the absolute difference between the predicted and target values
    diff = (predicted - target).abs()

    # Compute the mean of the absolute difference
    mae = diff.mean()

    # Compute the accuracy as 1 - MAE
    accuracy = 1 - mae

    return accuracy

def randint_distinct(a: int, b: int, n_samples: int) -> torch.Tensor:
    """
    Returns a tensor of `n_samples` distinct random integers in the range [a, b].

    Parameters:
        a (int): The start of the range (inclusive).
        b (int): The end of the range (inclusive).
        n_samples (int): The number of distinct random integers to generate.

    Returns:
        torch.Tensor: A tensor of `n_samples` distinct random integers in the range [a, b].
    """
    # Generate all possible values in the range [a, b] and shuffle them
    all_values = torch.arange(a, b)
    shuffled_values = all_values[torch.randperm(b - a)]
    # Return the first n_samples values from the shuffled tensor
    return shuffled_values[:n_samples]
