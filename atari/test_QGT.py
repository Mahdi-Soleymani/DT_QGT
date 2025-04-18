import torch
import sys
import time
sys.path.append("mingpt/")  # Adjust if needed
import mingpt.trainer_QGT as t
# Import your model class
#from model_QGT import model
import numpy as np
import random
from gurobipy import GRB, LinExpr
from torch.nn import functional as F
import gurobipy as gp

import matplotlib.pyplot as plt
import os



    
from mingpt.model_QGT import DecisionTransformer as QGT_model  # Alias your custom model
from gurobipy import Model as GurobiModel  # Alias Gurobi's Model


def random_integer_vector(k):
    x = np.zeros(k, dtype=int) # Initialize a zero vector
    x_half = np.zeros(k, dtype=int) # Initialize a zero vector
    for i in range(k):
        id=np.random.choice(k,1)
        x[id]+=1
        if random.random() < 0.5:
            x_half[id]+=1
    return x.reshape(-1,1),x_half.reshape(-1,1)

def pad_sequence(seq, max_len, pad_value=0):
    """Pads a sequence to max_len with pad_value"""

    seq = torch.tensor(seq, dtype=torch.float32)  # Convert to tensor



    pad_size = max_len - seq.shape[0]

    if pad_size > 0:
        zero_vector = pad_value*torch.ones(pad_size)
        seq = torch.cat((seq, zero_vector))

    return seq


def pad_sequence2d(seq, max_len, pad_value=0):
    """Pads a batch of sequences to max_len with pad_value"""

    # Convert the list of lists into a tensor
    #seq = [torch.tensor(q, dtype=torch.float32) for q in seq]  # Convert each query to a tensor
    seq = [q.clone().detach().to(dtype=torch.float32) for q in seq]
    # Stack into a 2D tensor (batch_size, seq_len)
    seq = torch.stack(seq)  # Shape: (batch_size, query_length)
    
    pad_size = max_len - seq.shape[0]
    
    if pad_size > 0:
        seq = F.pad(seq, (0, 0, 0, pad_size), value=pad_value)  # Pad along sequence dimension
    
    return seq



def test_sample(desired_num_of_queries):
    # Initialize the model and config
    config = t.TrainerConfig(
        k=10,
        query_dim=10,
        lr=3e-4,
        max_epochs = 2,
        batch_size = 1,
        learning_rate = 3e-4,
        betas = (0.9, 0.95),
        grad_norm_clip = 1.0,
        weight_decay = 0.1,
        lr_decay = False,
        warmup_tokens = 375e6,
        final_tokens = 260e9,
        ckpt_path="comic-mountain-67.pth",  # Set a valid path if you want to save checkpoints
        num_workers=0,
        rtg_dim=1,
        n_embd=512,
        query_result_dim=1,
        block_size=10,### number of max timesteps in sequence (seq len=3 times this)
        embd_pdrop = 0.1,
        n_layer=6,
        n_head=8,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        pad_scalar_val=-100,
        pad_vec_val=-30,
        desired_num_of_queries=8
    )
    config.query_dim=config.k
    config.desired_num_of_queries=desired_num_of_queries
    # Initialize your model architecture (it should be the same as during training)
    DT_model = QGT_model(config)  # Use the same configuration used during training

    # Load the saved model checkpoint  
    
    #checkpoint = torch.load("comic-mountain-67.pth",  map_location=torch.device("cpu"))
    #checkpoint = torch.load("zany-hill-68.pth",  map_location=torch.device("cpu"))
    #checkpoint = torch.load("misunderstood-serenity-69.pth",  map_location=torch.device("cpu"))
    #checkpoint = torch.load("morning-vortex-71.pth",  map_location='cpu', weights_only=True)
    checkpoint = torch.load("dulcet-field-88.pth",  map_location='cpu', weights_only=True)

    # Load the model weights directly from the checkpoint
    DT_model.load_state_dict(checkpoint)

    # Set the model to evaluation mode
    DT_model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    max_len = config.block_size  # Set max length
    pad_scalar_val=config.pad_scalar_val
    pad_vec_val=config.pad_vec_val



    q, r, rtg, mask_length= [ torch.full((config.k,), pad_vec_val, dtype=torch.int)],[config.k],[-config.desired_num_of_queries],1  # Generate a sequence
    queries=(pad_sequence2d(q, max_len,pad_vec_val))  # Pad queries
    results=(pad_sequence(r, max_len,pad_scalar_val))
    rtgs=(pad_sequence(rtg, max_len,pad_scalar_val))


    mask_length = torch.tensor(mask_length).to(device)
    results = results.to(device)
    rtgs    = rtgs.to(device)
    queries = queries.to(device)






    x,x_half=random_integer_vector(config.k)
    x_half_tensor=torch.tensor(x_half,dtype=torch.float32).to(device)
    G_model = GurobiModel("Incremental_ILP")



    #### to write nothing in the log 
    G_model.setParam(GRB.Param.OutputFlag, 0)
    # Create a list to store the variables for ILP
    variables = []

    # Add variables dynamically
    for i in range(0, config.k):
        variables.append(G_model.addVar(vtype=GRB.INTEGER, lb=0, ub=int(x[i].item()), name=f"x{i}"))


    # ###initial constraint
    # G_model.addConstr(gp.quicksum(variables) == config.k, name="sum_constraint")

    ### Enables solutions pool
    G_model.setParam(GRB.Param.PoolSearchMode, 2)

    # Set the objective (e.g., maximize x + y)
    G_model.setObjective(1 , GRB.MAXIMIZE)


    num_of_constraints=0
    is_solved=False

    rtgs = rtgs.unsqueeze(0)  # Adds batch dimension, result shape: [1, 10]
    results = results.unsqueeze(0)  # Adds batch dimension, result shape: [1, 10]
    queries = queries.unsqueeze(0)
    mask_length = mask_length.unsqueeze(0)  # Adds batch dimension, result shape: [1, 10, 10]

    while not is_solved and num_of_constraints<config.block_size:

        with torch.no_grad():  # No need to track gradients during inference

            ### from model
            probs,_=DT_model( mask_length, rtgs,  results, queries)


            ######## Random queries
            #probs=.5*torch.ones(config.batch_size,config.block_size,config.k).float()
            #probs = torch.randint(0, 2, (config.batch_size, config.block_size, config.k)).float()

        
        
        ###Sampling (soft)
        next_query = torch.bernoulli(probs[:,num_of_constraints,:])
       
        ### hard thresholding
        #print("probs")

        #probs=probs[:,num_of_constraints,:]

        #next_query = (probs > 0.5).float()
        #print (probs[:,num_of_constraints,:])

        #when using model 
        next_query=next_query[0,:]

        #when using random
        #next_query=next_query[0,0,:]
        #print(next_query)
       
        queries[:,num_of_constraints,:]=next_query


        selected_variables=[]
        for i in range(config.k):
            if next_query[i]==1:
                selected_variables.append(variables[i])

        new_result=torch.matmul(next_query,x_half_tensor)
            
        
        constraint = sum(selected_variables) == new_result.item()
        
        # Add the new constraint

    
        G_model.addConstr(constraint, name=f"{num_of_constraints}")
        num_of_constraints+=1
 
    
        
        # Optimize the initial model
        G_model.optimize()

        # Check the initial solution
        if G_model.status == GRB.OPTIMAL:
            # Get the total number of sn        olutions
            num_of_solutions=G_model.SolCount
            if num_of_solutions<=1:
                is_solved=True
            else:
                if num_of_constraints<config.block_size:
                    rtgs[:,    num_of_constraints]=min(-1,-config.desired_num_of_queries+num_of_constraints)
                    results[:,num_of_constraints]=new_result
                    mask_length[:,]=num_of_constraints+1
                
        else:
            print(f"No solution found!")
        



    return num_of_constraints



def run_test_sample(des_len,_):
    return test_sample(des_len)



def main():
    import concurrent.futures
    from tqdm import tqdm
    import argparse
    from functools import partial


    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iter", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--num_cores", type=int, default=6, help="Number of CPU cores to use")
    parser.add_argument("--des_len", type=int, default=3, help="Number of CPU cores to use")

    args = parser.parse_args()
    worker_fn = partial(run_test_sample, args.des_len)

    inputs = [args.des_len] * args.num_iter  # 👈 make it iterable!


    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_cores) as executor:
        results = list(tqdm(executor.map(worker_fn,range(args.num_iter)), total=args.num_iter))

    # results=[]
    # for l in range(1000):
    #     results.append(test_sample())
    #     print(l)


    results=np.array(results)
    print(results.mean())
    print(results.std())


if __name__ == "__main__":

    main()



    



        
