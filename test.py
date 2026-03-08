import torch
x=torch.arange(3).reshape((3,1))
y=torch.arange(2).reshape((1,2))
z= x+y
print(z)
print(z.shape)
