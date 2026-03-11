# calculate tflops using pytorch's GEMM/MATMUL
# must have cuda or rocm compiled pytorch
import torch
assert torch.cuda.is_available(), "pytorch wasnt compiled with cuda/rocm"

# Dimensions: M, K, N
M, K, N = 4096, 4096, 4096
device = 'cuda'

# float16/bfloat16 has more perf in high perf computing cards
A = torch.randn(M, K, device=device, dtype=torch.float32)
B = torch.randn(K, N, device=device, dtype=torch.float32)

# warmup gpu kernels
for _ in range(10): torch.matmul(A, B)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
num_iterations = 200
for i in range(num_iterations): 
  torch.matmul(A, B)
  print(f"matmul {i}/{num_iterations} done")
end_event.record()
torch.cuda.synchronize()

# 1 TFLOP = 10^12 operations
elapsed_time = start_event.elapsed_time(end_event) / 1000  # Convert ms to s
avg_time = elapsed_time / num_iterations
total_flops = 2 * M * N * K
tflops = (total_flops / avg_time) / 1e12

print(f"\nAverage Time: {avg_time*1000:.3f} ms")
print(f"Performance: {tflops:.2f} TFLOPS")
