import os
import pickle
from typing import Optional, Tuple, List, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def is_dist_env_available():
    return os.environ.get('WORLD_SIZE', None) is not None


def is_dist_initialized():
    """
    Returns True if distributed mode has been initiated (torch.distributed.init_process_group)
    """
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_initialized() else 0


def is_zero_rank():
    return get_rank() == 0


def move_to(obj: Any, device: torch.device, numpy: bool=False) -> Any:
    """recursively visit a tuple/list/dict structure (can extend to more types if required)"""
    # pylint: disable=unidiomatic-typecheck # explicitly differentiate tuple from NamedTuple
    if type(obj) is tuple or isinstance(obj, list):  # modify sequence by rebuilding it
        # noinspection PyArgumentList
        return type(obj)(move_to(x, device, numpy) for x in obj)
    if hasattr(obj, 'to'):
        # noinspection PyCallingNonCallable
        obj = obj.to(device)
        # convert floating point types to float32.
        obj = obj.float() if 'float' in str(obj.dtype) else obj
        return obj.numpy() if numpy else obj
    if isinstance(obj, dict):
        return type(obj)(**{k: move_to(v, device, numpy) for k,v in obj.items()})
    if isinstance(obj, tuple):  # NamedTuple case
        # noinspection PyArgumentList
        return type(obj)(*(move_to(x, device, numpy) for x in obj))
    return obj


def catch_unused_params(model: nn.Module):
    """
    Throws error and reports unused parameters in case there are any.
    Useful for catching such parameters to prevent torch.nn.parallel.DistributedDataParallel from crashing.

    Note: Call this after backward pass.
    """
    unused = [name for name, param in model.named_parameters()
              if param.grad is None and param.requires_grad]
    unused_str = "\n" + "\n".join(unused)
    assert len(unused) == 0, f'Found unused parameters: {unused_str}'


def reduce_dict_to_rank0(input_dict: Dict, average: bool):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
