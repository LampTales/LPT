import os
print(os.environ.get('CUDA_PATH'))

import torch
print(torch.cuda.is_available())
from torch.backends import cudnn
print(cudnn.is_available())

import torch
print("Pytorch version：")
print(torch.__version__)
print("CUDA Version: ")
print(torch.version.cuda)
print("cuDNN version is :")
print(torch.backends.cudnn.version())