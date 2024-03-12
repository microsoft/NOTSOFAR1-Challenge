import os
from typing import Any, Dict
import pandas as pd

import torch
import torch.nn as nn
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


def barrier():
    if is_dist_initialized():
        dist.barrier()


def get_device_name():
    if is_dist_initialized():
        # when the number of nodes is 1, we can use only get_rank() to get the device_id
        # but when the number of nodes is greater than 1, the device_id can be calculated by:
        device_id = get_rank() % torch.cuda.device_count()
        return f'cuda:{device_id}'

    return "cuda" if torch.cuda.is_available() else "cpu"


class DDPRowIterator:
    """ A class that wraps a DataFrame, such that the returned DataFrame row number is divided by the world size
        (i.e. the number of processes created by the DDP). The padded rows are filled with the row at the given
        dummy_row_idx field.
        This is useful for distributed inference, where we want to distribute the data across all processes, such that
        all processes are working on different rows at the same time, while no process is idle (DDP assumption).
        The next() method returns a tuple of (row, row_idx, is_dummy) where is_dummy is True if the row is a padded row.
        Each process will iterate over the rows that are assigned to it, and then stop when the rows are exhausted.

    Args:
        df (pd.DataFrame): the DataFrame to iterate over
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.world_size = get_world_size()
        self.current_process_idx = get_rank()
        self.rows_per_chunk = len(df) // self.world_size
        self.remainder = len(df) % self.world_size
        self.current_row_idx = 0
        self.dummy_row_idx = self.current_process_idx
        assert self.dummy_row_idx < len(self.df), f'{self.dummy_row_idx=} must be less than {len(self.df)=}'

    @property
    def _padded_df_len(self):
        return len(self.df) + ((self.world_size - self.remainder) if self.remainder > 0 else 0)

    def __len__(self):
        return int(self._padded_df_len / self.world_size)

    def __iter__(self):
        self.current_row_idx = self.current_process_idx
        return self

    def __next__(self):
        if self.current_row_idx >= self._padded_df_len:
            # Wait for all processes to finish processing
            barrier()
            raise StopIteration

        row_idx = self.current_row_idx

        if row_idx < len(self.df):
            is_dummy = False
            row = self.df.iloc[row_idx]
        else:
            # if we are here, we are padding the DataFrame (self.current_row_idx >= len(self.df))
            is_dummy = True
            row = self.df.iloc[self.dummy_row_idx]

        self.current_row_idx += self.world_size
        return row, row_idx, is_dummy


def initialize_ddp(logger):
    """ Process group initialization for distributed inference """
    if is_dist_env_available():
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        logger.info(f'Distributed: {get_rank()=}, {get_world_size()=}')
        # NOTE! must call set_device or allocations go to GPU 0 disproportionally, causing CUDA OOM.
        torch.cuda.set_device(torch.device(get_device_name()))
        dist.barrier()

    return get_device_name()


def get_max_value(value: int) -> int:
    """ Returns the maximum value from all processes """
    if not is_dist_initialized():
        return value

    tensor = torch.tensor(value).cuda()
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


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
