import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

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

model.load_state_dict(torch.load('modelq1.pt'))

num_workers = 0
batch_size = 20

train_on_gpu = torch.cuda.is_available()

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=test_transforms)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

criterion = nn.NLLLoss()

test_loss = 0.0
class_correct = list(0. for i in range(10)) #list of zeros
class_total = list(0. for i in range(10))

model.eval()

for inputs, labels in test_loader:

	logps = model.forward(inputs)
	loss = criterion(logps, labels)

	test_loss += loss.item()*inputs.size(0)

	_, pred = torch.max(logps, 1) 

	correct_tensor = pred.eq(labels.data.view_as(pred))
	correct = np.squeeze(correct_tensor.numpy())#if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())

	for i in range(batch_size):
		label = labels.data[i]
		class_correct[label] += correct[i].item()
		class_total[label] += 1

test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))