import numpy as np
import torch
import time
print(torch.get_num_threads()) 
import torch
import numpy as np

print("PyTorch threads:", torch.get_num_threads())

import mkl
print("MKL threads:", mkl.get_max_threads())
import torch
torch.set_num_threads(1)

print("PyTorch threads:", torch.get_num_threads())

a_np = np.random.rand(1000, 1000)
b_np = np.random.rand(1000, 1000)

a_torch = torch.tensor(a_np)
b_torch = torch.tensor(b_np)

# NumPy 加法
start = time.time()
for _ in range(1000):
    c = a_np + b_np
print("NumPy time:", time.time() - start)

# PyTorch Tensor 加法
start = time.time()
for _ in range(1000):
    c = a_torch + b_torch
print("Torch time:", time.time() - start)
