import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import Model, GRB, LinExpr
import random
import sys
import os
import time 


def random_integer_vector(k):
    x = np.zeros(k, dtype=int) # Initialize a zero vector
    x_half = np.zeros(k, dtype=int) # Initialize a zero vector
    for i in range(k):
        id=np.random.choice(k,1)
        x[id]+=1
        if random.random() < 0.5:
            x_half[id]+=1
    return x.reshape(-1,1),x_half.reshape(-1,1)


#if __name__ == "__main__":
def seq_fn(k):
    while True:
        x,x_half=random_integer_vector(k)
        model = Model("Incremental_ILP")
        #### to write nothing in the log 
        model.setParam(GRB.Param.OutputFlag, 0)
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
                    r.append(int(new_result))
                    
                    
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

        # for i in range(mask_length-1):
        #     yield q, r, rtg, i+1
            #yield q[:i], r[:i+1], rtg[:i+1], i+1, q[i]
            
        return q, r, rtg, mask_length


# for q, r, rwrd, length in seq_fn(10):
#     # Now you can process q, r, rwrd, and length in each iteration
#     print(q, r, rwrd, length)

# for q, r, rwrd, length in seq_fn(10):
#     # Now you can process q, r, rwrd, and length in each iteration
#     print(q, r, rwrd, length)
#     print("\n")

# gen = seq_fn(10)

# for i in range(20):
#     print(next(gen))
#     print("\n")
    # print (i)
    # print("\n")

# for i in range(2):
#     print(next(gen))
#     print("\n")
#     print (i)
#     print("\n")
#q,r,rwrd,a=seq_fn(10)
# #print(len(seq_fn(10)))
# print(f"queries are  {q}")
# print(f"results are {r}")
# print(f"returns are {rtg}")
# print(f"masks are {mask}")
# #print(f"time f{t}")
# a=seq_fn(10)
# print(len(a))