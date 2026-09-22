import torch
from net.CIDNet import CIDNet

model = CIDNet().cuda()

x = torch.randn(1,3,256,256).cuda()

y = model(x)

total_params = sum(p.numel() for p in model.parameters())

print("Total parameters:", total_params)
print("Total parameters (M):", total_params / 1e6)