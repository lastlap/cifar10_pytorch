import torch
torch.cuda.current_device()
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse


parser=argparse.ArgumentParser()

parser.add_argument("-epochs", "--epochs", type=int, default=5, help="Enter Number of epochs")
parser.add_argument("-learning_rate", "--learning_rate", type=float, default=0.003, help="Enter Learning Rate")
args=parser.parse_args()

train_on_gpu = torch.cuda.is_available()

num_workers = 0
batch_size = 20
validation_size = 0.2

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

train_data = datasets.CIFAR10('data', train=True,
							 download = True, transform = train_transforms)

num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(validation_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
    sampler=valid_sampler, num_workers=num_workers)

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.conv1 = nn.Conv2d(3,16,3,padding=1)
		self.conv2 = nn.Conv2d(16,32,3,padding=1)
		self.conv3 = nn.Conv2d(32,64,3,padding=1)

		self.pool = nn.MaxPool2d(2,2)

		self.fc1 = nn.Linear(64*4*4,500)
		self.fc2 = nn.Linear(500,10)

		self.dropout = nn.Dropout(0,25)

	def forward(self,x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.pool(F.relu(self.conv3(x)))

		x = x.view(-1, 64*4*4)
		x = self.dropout(x)
		x = F.relu(self.fc1(x))
		x = F.log_softmax(self.fc2(x),dim=1)
		return x

model = Net()

epochs = args.epochs
learning_rate = args.learning_rate

if train_on_gpu:
	model.cuda()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
valid_loss_min = np.Inf
for epoch in range(epochs):
	train_loss = 0
	valid_loss = 0

	model.train()
	for inputs, labels in train_loader:
		if train_on_gpu:
			inputs, labels = inputs.cuda(), labels.cuda()
		optimizer.zero_grad()

		logps = model.forward(inputs)
		loss = criterion(logps, labels)
		loss.backward()
		optimizer.step()
		train_loss += loss.item()*inputs.size(0)

	model.eval()
	for inputs, labels in valid_loader:
		if train_on_gpu:
			inputs, labels = inputs.cuda(), labels.cuda()

		logps = model.forward(inputs)
		loss = criterion(logps, labels)
		valid_loss += loss.item()*inputs.size(0)

	train_loss = train_loss/len(train_loader.sampler)
	valid_loss = valid_loss/len(valid_loader.sampler)

	print('Epoch:',epoch+1,'Training Loss:',train_loss,'Validation Loss:',valid_loss)

	if valid_loss <= valid_loss_min:
		print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
		        valid_loss_min,
		        valid_loss))
		torch.save(model.state_dict(), 'modelq1.pt')
		valid_loss_min = valid_loss
