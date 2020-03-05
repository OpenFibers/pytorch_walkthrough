import torch
import numpy as np

a = torch.ones(5)
b = a.numpy()
print(b)  # np.array

a.add_(1)
print(a)  # tensor
print(b)  # np.array

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)  # np.array
print(b)  # tensor

x = torch.ones(5)
if torch.cuda.is_available():
    print("cuda!")
    device = torch.device('cuda')
    y = torch.ones_like(x, device=device)  # directly create on device
    x = x.to(device)  # copy to device
    z = x + y
    print(z)  # tensor([0.1174], device='cuda:0')
    print(z.to('cpu', torch.double))  # tensor([0.1174], dtype=torch.float64)
else:
    print("cpu")