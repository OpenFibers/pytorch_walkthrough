import torch
import numpy as np


a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html