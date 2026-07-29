"""Confirm that the research environment has all core dependencies."""

from importlib import import_module
from pathlib import Path
import os
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))


DEPENDENCIES = {
    "torch": "PyTorch",
    "numpy": "NumPy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "matplotlib": "Matplotlib",
    "seaborn": "seaborn",
    "jupyterlab": "JupyterLab",
    "tqdm": "tqdm",
    "pytest": "pytest",
}


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    for module_name, display_name in DEPENDENCIES.items():
        module = import_module(module_name)
        version = getattr(module, "__version__", "installed")
        print(f"{display_name}: {version}")

    torch = import_module("torch")
    device = (
        "mps"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    sample = torch.tensor([[1.0, 2.0]]) @ torch.tensor([[3.0], [4.0]])
    assert sample.item() == 11.0
    print(f"PyTorch device available: {device}")
    print("Tensor smoke test: passed")

    required_directories = [
        "data",
        "notebooks",
        "figures",
        "paper",
        "checkpoints",
        "results",
        "src",
        "tests",
    ]
    missing = [name for name in required_directories if not (ROOT / name).is_dir()]
    if missing:
        raise RuntimeError(f"Missing project directories: {', '.join(missing)}")
    print("Project structure: passed")
    print("\nEnvironment ready.")


if __name__ == "__main__":
    main()
