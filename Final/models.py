from torch import nn
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
	transforms.RandomResizedCrop(32, scale=(0.6, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
	transforms.RandomHorizontalFlip(),
	transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
	transforms.RandomRotation(15),
	transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
	transforms.Resize((32, 32), transforms.InterpolationMode.BICUBIC),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_dataset(path, transform, batch_size=32):
	import os

	if not os.path.exists(path):
		return None, None
	
	dataset = datasets.ImageFolder(root=path,transform=transform)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	return dataset, dataloader

raw_train_set, raw_train_loader = load_dataset('data/train', test_transform)
train_set, train_loader = load_dataset('data/train', train_transform)

test_set, test_loader = load_dataset('data/valid', test_transform)

class MultiLayerPerceptron(nn.Module):
	def __init__(self, num_classes=10):
		super().__init__()

		self.flatten = nn.Flatten()

		self.network = nn.Sequential(
			nn.Linear(3 * 32 * 32, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(),

			nn.Linear(512, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(),

			nn.Linear(256, 64),
			nn.BatchNorm1d(64),
			nn.ReLU(),

			nn.Linear(64, num_classes)
		)

	def forward(self, x):
		return self.network(self.flatten(x))
	
	def save(self, path):
		torch.save(self.state_dict(), path)

	def load(model, path):
		model.load_state_dict(torch.load(path))

class CNNModel(nn.Module):
	def __init__(self, num_classes=10):
		super().__init__()

		self.features = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(2), # 16

			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(2), # 8

			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(2), # 4
		)

		self.classifier = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.BatchNorm1d(128),
			nn.Linear(128, num_classes)
		)

	def forward(self, x):
		x = self.features(x)
		x = self.classifier(x)
		return x
	
	def save(self, path):
		torch.save(self.state_dict(), path)

	def load(model, path):
		model.load_state_dict(torch.load(path))