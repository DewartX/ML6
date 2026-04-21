import torch
import numpy as np
import torch.nn as nn
import torchvision as tv
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

#Оценка точности
best_accuracy = 0.0

def test(max_batches=100):
	model.eval()
	correct = 0
	total = 0

	with torch.no_grad():
		for i, (images, labels) in enumerate(test_data_loader):
			if i>=max_batches:
				break

			output = model(images)
			predict = torch.argmax(output, dim = 1)

			correct += (predict == labels).sum().item()
			total += labels.size(0)
	return 100 * correct/total

#Данные
root = "./Data"
model_save_path="./Model.pth"

transformation = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5)),
])

train_set = CIFAR10(train=True, transform=transformation, root=root, download=True)
batch_size = 20
train_data_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)

test_set = CIFAR10(train=False, transform=transformation, root=root, download=True)
test_data_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#Модель
class ImageModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(12)

		self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

		self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(12)

		self.fc = nn.Linear(12*16*16, out_features = 10)

	def forward(self, inp):
		out = F.relu(self.bn1(self.conv1(inp)))
		out = self.pool(out)
		
		out = F.relu(self.bn2(self.conv2(out)))

		out = torch.flatten(out, start_dim=1)
		out = self.fc(out)
		return out

model = ImageModel()

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0005, weight_decay = 0.0001)
'''
#Обучение
num_epoch = 5
for epoch in range(num_epoch):
	for i, (images, labels) in enumerate(train_data_loader,0):
		optimizer.zero_grad()
		output = model(images)
		error = loss(output, labels)
		error.backward()
		optimizer.step()

	acc = test(50)
	print('Эпоха: %d; Точность: %d%%' %(epoch+1, acc))

	if(acc > best_accuracy):
		best_accuracy = acc
		torch.save(model.state_dict(), model_save_path)
'''
#Импортируем модель
load_model = ImageModel()
load_model.load_state_dict(torch.load(model_save_path))

load_model.eval()

#Получаем тестовые изображения и их теги
images, true_labels = next(iter(test_data_loader))

print('Labels: ', end='')
for i in range(20):
	print(classes[true_labels[i]], end=' ')
print()

#Предсказания нейросети
output = load_model(images)
predict = torch.argmax(output, dim=1)

print('Predicted: ', end ='')
for i in range(20):
	print(classes[predict[i]], end=' ')
print()

correct = (predict == true_labels).sum().item()
total = true_labels.size(0)
accuracy = 100 * correct / total

print(f"Правильных ответов: {correct} / {total}")
print(f"Точность: {accuracy:.2f}%")

#Изображения
images = tv.utils.make_grid(images)
images = images/2 + 0.5
plt.imshow(np.transpose(images.numpy(), axes = (1,2,0)))
plt.show()
