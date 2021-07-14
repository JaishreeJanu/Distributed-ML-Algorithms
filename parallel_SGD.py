import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.utils.data
import tqdm
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms

from mpi4py import MPI
import joblib

PATH_ARTIFACT = "pytorch_mpi_mnist.model"

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Hyperparameters
num_epochs = 7
num_classes = 10
batch_size = 64
learning_rate = 0.001
train_len = 40*1000  # Number of training images

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.25)

        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(13*13*64, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512)
        )


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.dropout(out)
        out = self.layer3(out)

        out = out.reshape(out.size(0), -1)
        out = self.fc_layer(out)

        return out

DATA_PATH = 'data'
MODEL_PATH = 'data/results'

train_dataset = MNIST(root = DATA_PATH, download=True, transform=transforms.Compose([
    transforms.ToTensor()
]))
test_dataset = MNIST(root = DATA_PATH, train=False, transform=transforms.Compose([
    transforms.ToTensor()
]))


part_train = torch.utils.data.random_split(train_dataset, [train_len, len(train_dataset)-train_len])[0]

train_sampler = DistributedSampler(part_train, num_replicas=size, rank=rank)
train_dataloader = DataLoader(dataset=part_train, batch_size=batch_size, shuffle=False, sampler=train_sampler)

model = ConvNet()


## Train Code here

# Loss and opitimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss = []
train_acc = []
input_data = None

for epoch in range(num_epochs):

    for data, labels in tqdm.tqdm(train_dataloader):
        # Run the forward pass
        input_data = data
        outputs = model(input_data)
        loss = criterion(outputs, labels)
        train_loss.append(loss.item())

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        train_acc.append(correct / total)

comm.barrier() # Wait to finish for all workers

params = []

# Now loop over each parameter, gather parameter from all workers and average the values
# Tasks of worker-0 : gather parameters, find average and save the final model
for param in model.parameters():
    all_params = comm.gather(param.data, root=0)
    if rank==0:
        if all_params:
            print(len(all_params))
            list_param_tensors = [x for x in all_params]
            this_param = torch.mean(torch.stack(list_param_tensors), dim=0)
            params.append(this_param)
            param.data = this_param


print(params)

if rank==0:
    joblib.dump(params, "params.joblib")
    torch.save(model.state_dict(), PATH_ARTIFACT)
    torch.onnx.export(model,               # model being run
                  input_data,                         # model input (or a tuple for multiple inputs)
                  "mnist_net_mpi.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})



