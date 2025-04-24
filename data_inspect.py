import h5py

file_path = "dataset_k5.h5"  # or f"{f_name}.h5"

with h5py.File(file_path, "r") as f:
    print("Keys in dataset:", list(f.keys()))  # List all datasets/groups
    
    # Example: print first few entries from each dataset
    for key in f.keys():
        data = f[key][:]
        print(f"\nDataset '{key}' shape: {data.shape}")
        print(f"First few entries of '{key}':\n", data[:5])
