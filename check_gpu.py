import torch

print(torch.cuda.device(0))
print(torch.cuda.get_device_name())
print(torch.cuda.is_available())


print('hello')