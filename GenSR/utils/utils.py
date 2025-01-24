import random
import numpy as np
import torch
import logging
import sys
import os
import math
import datetime

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def get_epoch_indices(dataset_length, sample_size, epoch_seed):
    # Create a local random number generator
    local_random = random.Random(epoch_seed)
    
    # Use the local random number generator to sample indices
    indices = local_random.sample(range(dataset_length), sample_size)
    return indices

def setup_logging(args):
    args.log_name = log_name(args)
    folder = './log'
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, args.log_name + '.log')
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file, level=args.logging_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    return

def log_name(args):
    name = args.phase
    date_time = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    name = name + '_' + date_time
    return name
    # params = [str(args.distributed), str(args.sample_prompt), str(args.his_prefix), str(args.skip_empty_his), str(args.max_his), str(args.master_port), folder_name, args.tasks, args.backbone, args.item_indexing, str(args.lr), str(args.epochs), str(args.batch_size), args.sample_num, args.his_rec_prompt_file[3:-4]]
    # return '_'.join(params)

def init_distributed_mode(args):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True
    args.world_size = int(os.environ["SLURM_NTASKS"])

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}, world {}): {}".format(
            args.rank, args.world_size, args.dist_url
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(
            days=365
        ),  # allow auto-downloading and de-compressing
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
