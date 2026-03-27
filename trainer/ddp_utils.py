from __future__ import annotations

import os
import random
import numpy as np
import torch
import torch.distributed as dist


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def seed_everything(seed: int, rank: int = 0):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
