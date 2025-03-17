import torch

# Check if CUDA is available
print(torch.cuda.is_available())

# Get the number of CUDA devices
print(torch.cuda.device_count())

# Get the name of the current CUDA device
print(torch.cuda.get_device_name(0))

# Move a tensor to the CUDA device
tensor = torch.randn(10).cuda()
print(tensor.device)
