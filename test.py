import torch
from torch import nn

random_tensor = torch.rand((1, 6, 6, 3))
qkv = nn.Linear(3, 3)
tensor = qkv(random_tensor)

print(random_tensor)
print(tensor)