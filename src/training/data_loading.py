import numpy.typing as npt
import numpy as np
import torch

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length - 1
    indices = np.random.randint(0, max_start + 1, size=batch_size)

    inputs_np = np.stack([dataset[idx : idx + context_length] for idx in indices])
    targets_np = np.stack([dataset[idx + 1 : idx + context_length + 1] for idx in indices])

    inputs = torch.from_numpy(inputs_np.astype(np.int64)).to(device)
    targets = torch.from_numpy(targets_np.astype(np.int64)).to(device)
    return inputs, targets
