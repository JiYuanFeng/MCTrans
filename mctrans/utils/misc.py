import random
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

def sync_param(input, reduction='mean'):
    if isinstance(input, np.ndarray):
        sync_input = torch.from_numpy(input).cuda()
    elif isinstance(input, torch.Tensor):
        sync_input = input.clone()
    else:
        raise ValueError('input should be torch tensor or ndarray')
    dist.all_reduce(sync_input)
    if reduction == 'mean':
        sync_input.div_(dist.get_world_size())
    return sync_input

def is_distributed():
    if dist.is_available() and dist.is_initialized():
        return True
    else:
        return False


