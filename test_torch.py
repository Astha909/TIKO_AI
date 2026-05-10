import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

tensor = torch.tensor([1, 2, 3])

print("Tensor:", tensor)