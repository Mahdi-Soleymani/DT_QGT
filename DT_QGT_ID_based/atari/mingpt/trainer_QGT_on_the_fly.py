import math
import logging
from tqdm import tqdm
import numpy as np
import time
import torch
import torch.optim as optim

logger = logging.getLogger(__name__)

import torch.nn.functional as F  # Import functional for padding

def pad_sequence(seq, max_len, pad_value=0):
    """Pads a sequence to max_len with pad_value"""
    seq = torch.tensor(seq, dtype=torch.float32)  # Convert to tensor
    pad_size = max_len - seq.shape[0]

    if pad_size > 0:
        zero_vector = pad_value*torch.ones(pad_size)
        seq = torch.cat((seq, zero_vector))

    return seq




class TrainerConfig:
    k=10
    max_epochs = 10
    batch_size = 64
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = False
    warmup_tokens = 375e6
    final_tokens = 260e9
    ckpt_path = None
    num_workers = 0 
    rtg_dim=1
    n_embd=128
    query_result_dim=1
    query_dim=10
    block_size=128
    embd_pdrop = 0.1
    n_layer=6
    n_head=8
    attn_pdrop=0.1
    resid_pdrop=0.1
    pad_scalar_val=-100
    pad_vec_val=0


    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    

class Trainer:
    def pad_sequence2d(self, seq, max_len, pad_value=0):
        """Pads a batch of sequences to max_len with pad_value"""
        
        # Convert the list of lists into a tensor
        if seq==[]:
            seq=[torch.full((self.config.k,), self.config.pad_vec_val, dtype=torch.int)]

        else:
            seq = [torch.tensor(q, dtype=torch.float32) for q in seq]  # Convert each query to a tensor
        # Stack into a 2D tensor (batch_size, seq_len)
        seq = torch.stack(seq)  # Shape: (batch_size, query_length)
        
        pad_size = max_len - seq.shape[0]
        
        if pad_size > 0:
            seq = F.pad(seq, (0, 0, 0, pad_size), value=pad_value)  # Pad along sequence dimension
        
        return seq


    def __init__(self, model, seq_fn, config):
        self.model = model
        self.seq_fn = seq_fn  # Function to generate sequences on the fly
        self.config = config

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        # Check if CUDA (or ROCm for AMD GPUs) is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Check if the model is being moved to the GPU correctly
        model = model.to(device)
        print(f"Model is on device: {next(model.parameters()).device}")
                
    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        gen = self.seq_fn(self.config.k)
        def generate_batch():
            # queries, results, rtgs , mask_lengths,targets= [], [], [], [],[]
            queries, results, rtgs , mask_lengths= [], [], [], []
            max_len = config.block_size  # Set max length
            pad_scalar_val=config.pad_scalar_val
            pad_vec_val=config.pad_vec_val
            
            for _ in range(config.batch_size):
                q, r, rtg, mask_length= self.seq_fn(self.config.k)  # Generate a sequence
                #q, r, rtg, mask_length, target=next(gen)
                #q, r, rtg, mask_length=next(gen)
                queries.append(self.pad_sequence2d(q, max_len,pad_vec_val))  # Pad queries
                results.append(pad_sequence(r, max_len,pad_scalar_val))
                rtgs.append(pad_sequence(rtg, max_len,pad_scalar_val))
                mask_lengths.append(mask_length)
                #targets.append(target)
            
            # print(queries[0])
            # print(results[0])
            # print(rtgs[0])
            # print(mask_lengths[0])
            # time.sleep(30)


            queries = torch.stack(queries).to(self.device)
            results = torch.stack(results).to(self.device)
            rtgs = torch.stack(rtgs).to(self.device)
            mask_lengths = torch.tensor(mask_lengths).to(self.device)
            #targets = torch.tensor(targets ,dtype=torch.float32).to(self.device)

            # print(queries[0])
            # print(results[0])
            # print(rtgs[0])
            # print(masks[0])
            # time.sleep(30)
            
            # return queries, results, rtgs, mask_lengths, targets
            return queries, results, rtgs, mask_lengths

        self.tokens = 0  
        def run_epoch(mode,epoch_num=0):
            is_train = mode == 'train'
            model.train(is_train)
            losses = []
            pbar = tqdm(range(config.batch_size), total=config.batch_size, desc=f"Epoch {epoch_num+1}")

            for _ in pbar:
                #queries, results, rtgs , mask_lengths
                # q, r, rtg, mask_lengths,targets= generate_batch()
                q, r, rtg, mask_lengths= generate_batch()
                                # place data on the correct device
                q = q.to(self.device)
                r = r.to(self.device)
                rtg = rtg.to(self.device)
                mask_lengths=mask_lengths.to(self.device)
                #targets=targets.to(self.device)
                #t = t.to(self.device)

                with torch.set_grad_enabled(True):
                     #rtgs,  query_results, queries,targets
                    # _, loss = model(mask_lengths,rtg, r, q, targets )
                    _, loss = model(mask_lengths,rtg, r, q, q)

                                        
                    # print(f"mask_lengths:\n{mask_lengths}\n")
                    # print(f"rtg:\n{rtg}\n")
                    # print(f"r:\n{r}\n")
                    # print(f"q:\n{q}\n")
                    # print(f"targets:\n{targets}\n")
                    # time.sleep(3)

                    
                    
                    loss = loss.mean()
                    losses.append(loss.item())

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    # for name, param in model.named_parameters():
                    #     # if param.grad is not None:
                    #     #     print(f"{name} gradient sum: {param.grad.sum()}")
                    #     grad_norm = param.grad.norm()
                    #     print(f"Gradient norm for {name}: {grad_norm}")
                    # print("new")
                    # print("\n") 
                    # time.sleep(1)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()
                        if self.tokens < config.warmup_tokens:
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    #report progress
                    pbar.set_description(f"Epoch {epoch_num+1} Loss: {loss.item():.5f}, LR: {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss
            
            avg_loss = float(np.mean(losses))
            logger.info("Epoch %d - Avg Loss: %f", epoch + 1, avg_loss)

            if config.ckpt_path is not None:
                self.save_checkpoint()
                print("check_point saved")

        for epoch in range(config.max_epochs):
            run_epoch('train', epoch_num=epoch)

