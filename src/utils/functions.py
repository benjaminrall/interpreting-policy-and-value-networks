import torch
import random
import numpy as np

def get_device() -> torch.device:
    """
    Gets the device to be used by torch.
    - `cuda` for NVIDIA and AMD
    - `mps` for Apple
    - `cpu` otherwise
    """
    return torch.device(
        'cuda' if torch.cuda.is_available() else 
        'mps' if torch.mps.is_available() else 
        'cpu'
    )

    
def to_nested_dict(obj) -> dict:
    """Recursively converts an object and its nested objects into a dictionary."""
    if isinstance(obj, dict):
        return {k: to_nested_dict(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        return {
            k: to_nested_dict(v) for k, v in vars(obj).items()
            if not k.startswith("_") and not callable(v)
        }
    elif isinstance(obj, (list, tuple)):
        return [to_nested_dict(i) for i in obj]
    else:
        return obj

def get_random_state() -> dict:
    """Returns a dictionary containing the current random state."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
    }

def restore_random_state(state: dict) -> None:
    """Restores a random state from a dictionary."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    for i, state in enumerate(state["cuda"]):
        torch.cuda.set_rng_state(state, device=i)