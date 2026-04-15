import torch 
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

root = "./Data"

transformation = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5)),
])

train_set = CIFAR10(train=True, transform=transformation, root=root, download=True)
batch_size = 10
train_data_loader = DataLoader(train_set, batch_size = batch_size, shuffle = False)

test_set = CIFAR10(train=True, transform=transformation, root=root, download=True)
test_data_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)