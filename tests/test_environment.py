import numpy as np
import torch

from tiny_transformer_world import __version__


def test_project_package_is_importable() -> None:
    assert __version__ == "0.1.0"


def test_numpy_and_torch_computation() -> None:
    array = np.array([[1.0, 2.0]])
    tensor = torch.from_numpy(array)
    result = tensor @ torch.tensor([[3.0], [4.0]], dtype=torch.float64)
    assert result.item() == 11.0
