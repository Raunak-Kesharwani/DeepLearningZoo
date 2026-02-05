import torch

def preprocess_sine_window(values):
    """
    values: list or iterable of length 20
    Returns tensor of shape (1, 20, 1)
    """
    if len(values) != 20:
        raise ValueError("Expected exactly 20 values")

    tensor = torch.tensor(values, dtype=torch.float32)
    return tensor.view(1, 20, 1)
