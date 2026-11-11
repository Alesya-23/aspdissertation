import torch

print("CUDA доступно:", torch.cuda.is_available())
print("Количество GPU:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Имя GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(3, 3).cuda()
    print("Тензор на GPU:", x)
