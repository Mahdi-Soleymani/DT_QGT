import sys
sys.path.append("mingpt/")  # Adjust if needed
sys.path.append(sys.path[0])
import mingpt.model_QGT as m
import mingpt.trainer_QGT as t
#import atari.sample_generator_on_the_fly as s
import torch
import csv
import logging
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.model_QGT import DecisionTransformer, GPTConfig
from mingpt.trainer_QGT import Trainer, TrainerConfig
from mingpt.utils import sample
from collections import deque
import random
import torch
import pickle
#import blosc
import argparse
import run_QGT as run
import time
import wandb
import wandb
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import h5py
from torch.utils.data import DataLoader, TensorDataset



# Argument parsing
parser = argparse.ArgumentParser()

# Add arguments for each config parameter
parser.add_argument('--seed', type=int, default=123, help='Random seed for reproducibility')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.95), help='Betas for Adam optimizer')
parser.add_argument('--grad_norm_clip', type=float, default=1.0, help='Gradient norm clipping')
parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay for optimizer')
parser.add_argument('--lr_decay', type=bool, default=True, help='Whether to apply learning rate decay')
parser.add_argument('--warmup_tokens', type=float, default=18e6, help='Number of warmup tokens')
parser.add_argument('--final_tokens', type=float, default=18e8, help='Total number of tokens for training')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
parser.add_argument('--rtg_dim', type=int, default=1, help='Reward-to-go dimension')
parser.add_argument('--n_embd', type=int, default=512, help='Embedding size')
parser.add_argument('--query_result_dim', type=int, default=1, help='Query result dimension')
parser.add_argument('--block_size', type=int, default=10, help='Max timesteps in sequence')
parser.add_argument('--embd_pdrop', type=float, default=0.1, help='Embedding dropout probability')
parser.add_argument('--n_layer', type=int, default=6, help='Number of transformer layers')
parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads')
parser.add_argument('--attn_pdrop', type=float, default=0.1, help='Attention dropout probability')
parser.add_argument('--resid_pdrop', type=float, default=0.1, help='Residual dropout probability')
parser.add_argument('--pad_scalar_val', type=float, default=-10, help='Padding scalar value')
parser.add_argument('--pad_vec_val', type=float, default=-30, help='Padding vector value')
parser.add_argument('--dataset_path', type=str, default="atari/data_6e6.h5", help='Path to Dataset')
parser.add_argument('--criterion', type=str, default='bce', help='Loss criterion (e.g., mse, mae)')
parser.add_argument('--clip_grad', type=bool, default=False, help='Whether to apply gradient clipping')
parser.add_argument('--label_smoothing', type=float, default=0.0,
                    help='Label smoothing factor for binary targets (0.0 = no smoothing)')


# Parse arguments
args = parser.parse_args()

# Set the seed for reproducibility
set_seed(args.seed)

# Initialize the TrainerConfig using command-line arguments
config = t.TrainerConfig(
    k=10,
    query_dim=0,
    max_epochs=args.epochs,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    betas=args.betas,
    grad_norm_clip=args.grad_norm_clip,
    weight_decay=args.weight_decay,
    lr_decay=args.lr_decay,
    warmup_tokens=args.warmup_tokens,
    final_tokens=args.final_tokens,
    num_workers=args.num_workers,
    rtg_dim=args.rtg_dim,
    n_embd=args.n_embd,
    query_result_dim=args.query_result_dim,
    block_size=args.block_size,
    embd_pdrop=args.embd_pdrop,
    n_layer=args.n_layer,
    n_head=args.n_head,
    dataset_path=args.dataset_path,
    attn_pdrop=args.attn_pdrop,
    resid_pdrop=args.resid_pdrop,
    pad_scalar_val=args.pad_scalar_val,
    pad_vec_val=args.pad_vec_val,
    seed=args.seed,
    criterion=args.criterion,
    clip_grad=args.clip_grad,
    label_smoothing=args.label_smoothing
)




set_seed(config.seed)
config.query_dim=config.k


def dataset():
#"""Load dataset from HDF5 file and create DataLoader."""
    with h5py.File(config.dataset_path, "r") as f:
        queries = torch.tensor(f["queries"][:], dtype=torch.float32)
        results = torch.tensor(f["results"][:], dtype=torch.float32)
        rtgs = torch.tensor(f["rtgs"][:], dtype=torch.float32)
        mask_lengths = torch.tensor(f["mask_lengths"][:], dtype=torch.long)

    dataset = TensorDataset(queries, results, rtgs, mask_lengths)
    #self.data_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers)
    return dataset

def setup_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", init_method="env://")
    return local_rank, rank, world_size

def cleanup_distributed():
    dist.destroy_process_group()








def main():

    local_rank, rank, world_size = setup_distributed()
    model = m.DecisionTransformer(config).to(local_rank)

 
    
    #model = DDP(model, device_ids=[local_rank])
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    data = dataset()
    sampler = DistributedSampler(data, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(data, batch_size=config.batch_size, sampler=sampler)

    if rank==0:
        #wandb.init(mpde="disabled")
        wandb.init(project="DT", config=config)


    print("Trainer is:", Trainer)
    print("Type:", type(Trainer))
    import inspect
    print("Signature:", inspect.signature(Trainer))

    trainer = t.Trainer(model, dataloader, local_rank, rank, config, False)
    trainer.train()

    cleanup_distributed()



if __name__ == "__main__":
    main()






