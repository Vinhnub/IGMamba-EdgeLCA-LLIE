from thop import profile
import torch
import time
from net.CIDNet_base import CIDNet


model = CIDNet().to('cuda')  
input = torch.rand(1,3,256,256).to('cuda')  
model.eval()

# Warmup to initialize CUDA and avoid cold start overhead
with torch.no_grad():
    for _ in range(50):
        _ = model(input)

# Measure time
torch.cuda.synchronize()
time_start = time.time()
num_iterations = 100
with torch.no_grad():
    for _ in range(num_iterations):
        _ = model(input)
torch.cuda.synchronize()
time_end = time.time()

time_avg = (time_end - time_start) / num_iterations
print(f"Average Time over {num_iterations} iterations: {time_avg:.6f} s")
n_param = sum([p.nelement() for p in model.parameters()])  
n_paras = f"n_paras: {(n_param/2**20)}M\n"
print(n_paras)
macs, params = profile(model, inputs=(input,)) 
print(f'FLOPs:{macs/(2**30)}G')
