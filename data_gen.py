import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import Model, GRB, LinExpr
import random
import sys
import os
import time 
import concurrent.futures
import h5py
from tqdm import tqdm




def pad_sequence(seq, max_len, pad_value=0):
    """Pads a 1D sequence to max_len with pad_value"""
    seq = np.array(seq, dtype=np.int8)  # Convert to NumPy array
    pad_size = max_len - seq.shape[0]

    if pad_size > 0:
        seq = np.pad(seq, (0, pad_size), mode='constant', constant_values=np.int8(pad_value))

    return seq





def pad_sequence2d(seq, max_len, pad_value=0):
    """Pads a list of sequences (batch of sequences) to max_len with pad_value"""
    
    # If seq is empty, create an empty list of padded sequences
    if len(seq) == 0:
        return np.full((max_len, len(seq[0])), pad_value, dtype=np.int8)  # Pad with empty sequences
    
    # Convert each sequence in the list to a NumPy array (if not already)
    seq = [np.array(q, dtype=np.int8) for q in seq]
    
    # Calculate the number of sequences and the sequence length
    num_sequences = len(seq)
    seq_length = len(seq[0])

    # Ensure all sequences have the same length (padding if necessary)
    if num_sequences < max_len:
        # Create a padding sequence of pad_value with the same length as each sequence
        pad_matrix = np.full((max_len - num_sequences, seq_length), int(pad_value), dtype=np.int8)
        # Stack the original sequences with the padding matrix
        seq = np.vstack((seq, pad_matrix))
    
    return seq





def random_integer_vector(k):
    x = np.zeros(k, dtype=int) # Initialize a zero vector
    x_half = np.zeros(k, dtype=int) # Initialize a zero vector
    for i in range(k):
        id=np.random.choice(k,1)
        x[id]+=1
        if random.random() < 0.5:
            x_half[id]+=1
    return x.reshape(-1,1),x_half.reshape(-1,1)


def seq_fn(k,max_len, pad_scalar_val,pad_vec_val):
    while True:
        x,x_half=random_integer_vector(k)
        model = Model("Incremental_ILP")
        
        #### to write nothing in the log 
        model.setParam(GRB.Param.OutputFlag, 0)

        model.setParam(GRB.Param.Threads, 1)
        # Create a list to store the variables for ILP
        variables = []

        # Add variables dynamically
        for i in range(0, k):
            variables.append(model.addVar(vtype=GRB.INTEGER, lb=0, ub=int(x[i].item()), name=f"x{i}"))

        ### Enables solutions pool
        model.setParam(GRB.Param.PoolSearchMode, 2)
        
        # Set the objective (e.g., maximize x + y)
        model.setObjective(1 , GRB.MAXIMIZE)
        model.optimize()
        q=[]
        r=[k]
        rwrd=[-1]
        #t=[]
        #sequence=[1,k]
        num_of_constraints=0
        is_solved=False
        # Define the constraint dynamically
        while not is_solved:
            # Randomly decide whether to include each variable (1/2 probability)
            new_test=np.zeros(k, dtype=int)
            selected_variables=[]
            while not selected_variables:
                index=0
                for var in variables:
                    if random.random() < 0.5:
                        selected_variables.append(var)
                        new_test[index]=1
                    index+=1
            
            new_result=np.matmul(new_test,x_half)
            q.append(new_test)
            constraint = sum(selected_variables) == new_result
        
            # Add the new constraint
            model.addConstr(constraint, name=f"{num_of_constraints}")
            num_of_constraints+=1
            #t.append(num_of_constraints)

            # Optimize the initial model
            model.optimize()
            # Check the initial solution
            if model.status == GRB.OPTIMAL:
                # Get the total number of solutions
                num_of_solutions=model.SolCount
                if num_of_solutions<=1:
                    is_solved=True
                    rwrd.append(0)
                    r.append(int(new_result[0]))
                    
                    
                else:
                    rwrd.append(-1)
                    r.append(int(new_result))
                    
            # else:
            #     #model.status == GRB.INFEASIBLE:
            #     print("No solution found! Model is infeasible.")
            #     model.computeIIS()  # Identify infeasible constraints
            #     model.write("infeasible_constraints.ilp")  # Save constraints causing infeasibility
            #     exit() 
            #     is_solved=True          
            else:
                print(f"No solution found!")
                is_solved=True
        rtg=[]
        s=0
        for reward in reversed(rwrd):
            s+=reward
            rtg.append(s)
        
        rtg=list(reversed(rtg))
        # mask=[1 for _ in range(len(r))]
        mask_length=len(rtg)
        if mask_length>10:
            q=q[:10]
            r=r[:10]
            rtg=rtg[:10]
            mask_length=10
        
        # Pad sequences
        q_padded = pad_sequence2d(q, max_len, pad_vec_val)
        r_padded = pad_sequence(r, max_len, pad_scalar_val)
        rtg_padded = pad_sequence(rtg, max_len, pad_scalar_val)
        
        return q_padded, r_padded, rtg_padded, np.int8(mask_length)






# Example usage
class Config:
    def __init__(self):
        self.block_size = 10  # Example block size, change as needed
        self.pad_scalar_val = -10  # Padding value for scalar sequences
        self.pad_vec_val = -30  # Padding value for vector sequences
        self.k = 10  # Length of the query vector, for example
        self.max_len=10
# Instantiate the config
config = Config()


#save_dataset("dataset.h5", num_samples=1000, seq_fn=seq_fn, config=config)







def generate_and_store_sample(worker_id, num_samples_per_worker, seq_fn, config, file_prefix):
    """Generate and store multiple samples in a single HDF5 file per worker."""
    file_name = f"{file_prefix}_{worker_id}.h5"
    
    with h5py.File(file_name, 'w') as f:
        d_queries = f.create_dataset("queries", (num_samples_per_worker, config.max_len, config.k), dtype='i1')
        d_results = f.create_dataset("results", (num_samples_per_worker,config.max_len), dtype='i1')
        d_rtgs = f.create_dataset("rtgs", (num_samples_per_worker,config.max_len), dtype='i1')
        d_mask_lengths = f.create_dataset("mask_lengths", (num_samples_per_worker,), dtype='i1')
        
        # # Initialize tqdm with manual updates
        pbar = tqdm(total=num_samples_per_worker, position=worker_id, desc=f"Worker {worker_id}", leave=True)

        for i in range(int(num_samples_per_worker)):
            try:
                q, r, rtg, mask_length = seq_fn(config.k, config.max_len, config.pad_scalar_val, config.pad_vec_val)

                
            except Exception as e:
                print(f"Error in worker {worker_id}: {e}")
                continue
            
    
            d_queries[i] = q
            d_results[i] = r
            d_rtgs[i] = rtg
            d_mask_lengths[i] = mask_length
            if i % 100 == 0:
                pbar.update(100)  # Manually update progress
            #print(f"Worker {worker_id}: {i}/{num_samples_per_worker}", flush=True)

        pbar.close()
    print(f"Worker {worker_id} saved {num_samples_per_worker} samples to {file_name}")







def save_dataset_parallel(filename, num_samples, seq_fn, config,num_cores):
    """Generate dataset samples in parallel and merge them into one HDF5 file."""
    #num_cores = os.cpu_count()  # Get number of available cores
    file_prefix = "temp_sample_file_worker"
    start = time.time()
    
    print(f"Using {num_cores} CPU cores for parallel processing...")
    
    samples_per_worker = num_samples // num_cores
    extra_samples = num_samples % num_cores  # Handle remainder
    print(f"samples_per_worker is {samples_per_worker}")
    print(f"extra_samples is {extra_samples}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for worker_id in range(num_cores):
            num_samples_for_worker = samples_per_worker + (1 if worker_id < extra_samples else 0)
            if num_samples_for_worker > 0:
                ##### for debugging 
                #generate_and_store_sample( worker_id, num_samples_for_worker, seq_fn, config, file_prefix)
                try:
                    futures.append(executor.submit(generate_and_store_sample, worker_id, num_samples_for_worker, seq_fn, config, file_prefix))
                except Exception as e:
                    print(f"Error in worker {worker_id}: {e}")
        concurrent.futures.wait(futures)

    print(f"Data generation took {time.time() - start:.2f} seconds.")

    # Merge worker files into the final dataset
    #f"{file_prefix}_worker_{worker_id}.h5"
    worker_files = [f"{file_prefix}_{worker_id}.h5" for worker_id in range(num_cores) if os.path.exists(f"{file_prefix}_{worker_id}.h5")]
  
    merge_datasets(worker_files, filename)





def merge_datasets(input_files, output_file):
    """Merge multiple HDF5 files into a single file."""
    with h5py.File(output_file, 'w') as f_out:
        for i, file in enumerate(input_files):
            with h5py.File(file, 'r') as f_in:
                for key in f_in.keys():
                    data = f_in[key][:]
                    if key in f_out:
                        # Append to the existing dataset
                        f_out[key].resize((f_out[key].shape[0] + data.shape[0]), axis=0)
                        f_out[key][-data.shape[0]:] = data
                    else:
                        # Create dataset with chunking and resizing enabled
                        maxshape = (None,) + data.shape[1:]  # Allow unlimited growth on axis 0
                        f_out.create_dataset(key, data=data, maxshape=maxshape, chunks=True)

    print(f"Merged {len(input_files)} files into {output_file}")



    # Delete temporary files
    for file in input_files:
        os.remove(file)

def count_samples_in_h5(file_name):
    try:
        with h5py.File(file_name, 'r') as f:
            # List all datasets in the HDF5 file
            queries = f['queries']
            results=f['results'] 
            rtgs=f['rtgs']
            mask_lengths=f["mask_lengths"]
            num_samples = queries.shape[0]  # Assuming 'queries' dataset holds the samples
            print(f"Number of samples in {file_name}: {num_samples}")
            # for item in results:
            #     print(item)
    except Exception as e:
        print(f"Error reading {file_name}: {e}")





# Example usage:
# save_dataset_parallel("final_dataset.h5", num_samples=1000, seq_fn=seq_fn, config=config, max_len=100, pad_scalar_val=0, pad_vec_val=0)
if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    import argparse
    import os
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cores", type=int, default=4, help="Number of CPU cores to use")
    parser.add_argument('--num_samples', type=int, default=100, help='Total number of samples to generate')
    parser.add_argument('--file_name', type=str, default="dataset", help='Total number of samples to generate')
    parser.add_argument("--k", type=int, default=10, help="k")
    parser.add_argument("--max_len", type=int, default=10, help="Maximum length data sequences have")

    args = parser.parse_args()
    config.k=args.k
    config.max_len=args.max_len 
    n_cores = min(args.n_cores, os.cpu_count())
    num_samples = args.num_samples
    f_name=args.file_name
    f_name=f"{f_name}_k{config.k}"
    save_dataset_parallel(f"{f_name}.h5", num_samples, seq_fn, config, n_cores)
    count_samples_in_h5(f"{f_name}.h5")
    file_path = f"{f_name}.h5"

    file_size = os.path.getsize(file_path)

    #print(f"File size: {file_size} bytes")

    file_size_mb = file_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

