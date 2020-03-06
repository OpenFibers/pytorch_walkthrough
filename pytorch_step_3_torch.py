import torch

x = torch.rand(5, 3)
print(x)
print(torch.is_tensor(x))
print(torch.is_storage(x))
print(torch.is_floating_point(x))

print('')
print(torch.get_default_dtype())  # torch.float32
print(torch.tensor([1.2, 3]).dtype)  # default is torch.float32
torch.set_default_dtype(torch.float64)
print(torch.tensor([1.2, 3]).dtype)

print('')
torch.set_default_dtype(torch.float64)
print(torch.get_default_dtype())
torch.set_default_tensor_type(torch.FloatTensor)
print(torch.get_default_dtype())

print('')
x = torch.tensor([5, 3])
print(x)
print(torch.numel(x))
x = torch.rand(5, 3)
print(x)
print(torch.numel(x))

print('')
print(torch.set_flush_denormal(True))
print(torch.set_flush_denormal(False))
