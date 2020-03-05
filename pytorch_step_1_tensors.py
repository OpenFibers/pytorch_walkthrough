import torch

x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.double)
print(x)

print(x.size())

y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

y.add_(x)
print(y)

print(x[:, 1])
print(x[1, :])

x = torch.randn(4, 4)
y = x.view(16)
print(y)
z = x.view(2, 8)
print(z)
z = x.view(-1, 8)
print(z)
print(x.size(), y.size(), z.size())

x = torch.randn(1)
print(x)
print(x.item())