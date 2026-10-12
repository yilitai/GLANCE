import numpy as np
import torch


def to_tensor_view(x, device=None, dtype=None):
    """
    Return a torch.Tensor view of x with minimal copies.

    - numpy -> torch (CPU): zero-copy (shared memory)
    - torch -> torch: no copy unless device/dtype change
    """

    if isinstance(x, torch.Tensor):
        if device is None and dtype is None:
            return x
        return x.to(device=device or x.device, dtype=dtype or x.dtype)

    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)  # zero-copy (CPU)
        if device is None and dtype is None:
            return t
        return t.to(device=device or t.device, dtype=dtype or t.dtype)

    raise TypeError("Expected numpy.ndarray or torch.Tensor")


def to_dual_view(x, device=None, dtype=None):
    """
    Return (tensor, numpy) views with minimal copies.

    - numpy -> tensor: zero-copy (CPU)
    - tensor(CPU) -> numpy: zero-copy
    - tensor(GPU) -> numpy: copy required
    """

    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)  # zero-copy
        if device is not None or dtype is not None:
            t = t.to(device=device or t.device, dtype=dtype or t.dtype)
        return t, x

    if isinstance(x, torch.Tensor):
        t = (
            x
            if (device is None and dtype is None)
            else x.to(device=device or x.device, dtype=dtype or x.dtype)
        )

        if t.device.type == "cpu":
            n = t.detach().numpy()  # zero-copy
        else:
            n = t.detach().cpu().numpy()  # unavoidable copy

        return t, n

    raise TypeError("Expected numpy.ndarray or torch.Tensor")
