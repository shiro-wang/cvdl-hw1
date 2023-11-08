import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data

from torch.utils.data import random_split
from torch.autograd import Variable
import os
#import matplotlib.image as mpimg
import matplotlib.pyplot as plt

DOANLOAD_DATASET = True
EPOCH = 60
BATCH_SIZE = 100
LR = 1e-3
MODELS_PATH = './models'
MODELS_SAVE = 'VGG19_model_1.pth'

class VGG19(torch.nn.Module):
    def __init__(self,num_classes):
        super(VGG19, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.block5  = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.classifier = nn.Sequential(
            nn.Linear(512*1*1,4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096,num_classes)
        )
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        logits = self.classifier(x.view(-1,512*1*1))
        probas = F.softmax(logits,dim = 1)
        return logits,probas

def main():
    transform = torchvision.transforms.Compose([
        # torchvision.transforms.RandomCrop(32, 4),
        # torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    
    train_dataset = torchvision.datasets.CIFAR10(root='data', train=True, transform=transform, download=True)
    train_data ,val_data = random_split(train_dataset,[40000,10000],generator=torch.Generator().manual_seed(56))
    test_dataset = torchvision.datasets.CIFAR10(root='data', train=False, transform=transform, download=True)

    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = Data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    model = VGG19(len(classes)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_function = nn.CrossEntropyLoss()
    train_loss_history = []
    valid_loss_history = []
    train_acc_history = []
    valid_acc_history = []

    for epoch in range(EPOCH):
        model.train()
        train_acc_total = 0
        for step, (inputs, labels) in enumerate(train_loader):
            # b_inputs = Variable(inputs, requires_grad=False).to(device)
            # b_labels = Variable(labels, requires_grad=False).to(device)
            b_inputs = inputs.to(device)
            b_labels = labels.to(device)

            optimizer.zero_grad()
            out, prob = model(b_inputs)
            loss = loss_function(out, b_labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(out.data, 1)
            train_acc_total += (predicted.to("cpu") == labels).sum().item()

            if step % 2000 == 0:
                print('Epoch: {} | Step: {} | Loss: {}'.format(epoch + 1, step, loss.item()))
        train_loss_history.append(loss.item())
        train_acc_history.append(train_acc_total / len(train_loader))

        valid_loss_total = 0
        valid_acc_total = 0
        model.eval()
        for step, (inputs, labels) in enumerate(val_loader):
            b_inputs = Variable(inputs, requires_grad=False).to(device)
            b_labels = Variable(labels, requires_grad=False).to(device)

            out, prob = model(b_inputs)
            loss = loss_function(out, b_labels)
            valid_loss_total += loss.item()
            _, predicted = torch.max(out.data, 1)
            valid_acc_total += (predicted.to("cpu") == labels).sum().item()

        valid_loss_history.append(valid_loss_total / len(val_loader))
        valid_acc_history.append(valid_acc_total / len(val_loader))
        print('Valid Epoch: {} | Loss: {}'.format(epoch + 1, valid_loss_total/len(val_loader)))

    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)
    torch.save(model.state_dict(), os.path.join(MODELS_PATH, MODELS_SAVE))

    # plot
    x_axis = range(0, EPOCH)
    # plot loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, train_loss_history, label='Train')
    ax.plot(x_axis, valid_loss_history, label='Validation')
    ax.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('VGG19 loss')
    plt.savefig("loss.png")
    plt.show()
    
    # plot acc
    fig, ax = plt.subplots()
    ax.plot(x_axis, train_acc_history, label='Train')
    ax.plot(x_axis, valid_acc_history, label='Validation')
    ax.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('VGG19 acc')
    plt.savefig("acc.png")
    plt.show()
    
    
if __name__=='__main__':
    main()