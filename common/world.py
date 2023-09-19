""" Information about the world (working environment) """
import os
import torch
import torch.distributed

size = 1
""" World size. """

rank = 0
""" 
    If DDP is used, the rank of the current process. Set to 'main' for the main process. 
    If DDP is not used, set to 0.
"""

distributed = False

cache_dir = '../cache'
""" Cache directory. """

def init(size_, rank_):
    """ Init world, shall be called before using anything else. """
    global size, rank
    size = size_
    rank = rank_


def ddp_setup(devices, master_addr, master_port):
    # Each process will only see its GPU under id 0, do this setting before any other CUDA call.
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{devices[rank]}'
    print(f"Rank {rank} uses physical device {os.environ['CUDA_VISIBLE_DEVICES']}")

    if not torch.cuda.is_available():
        raise RuntimeError('Only CUDA training is supported')

    # Initialize process group
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)

    backend = 'nccl' if torch.distributed.is_nccl_available() else 'gloo'
    print(f'Rank {rank} uses backend {backend}')
    world_size = len(devices)
    torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)

    global distributed
    distributed = True


def ddp_cleanup():
    torch.distributed.destroy_process_group()
