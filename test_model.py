import torch
import os
from torch import nn
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ResnetCifar(nn.Module):
    def __init__(self, classes):
        super(ResnetCifar, self).__init__()
        self.resnet = torchvision.models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 7x7 kernel to 3x3 kernel
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # delete maxpool layer
        self.resnet.maxpool = nn.Identity()
        # change output dimension
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, classes)

    def forward(self, x):
        return self.resnet(x)
    
class MobileNetV3(nn.Module):
    def __init__(self, classes):
        super(MobileNetV3, self).__init__()
        self.mobilenet = torchvision.models.mobilenet_v3_large(weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        # do some change in mobilenet
        # reduce down-sampling times from 5 to 2
        # because mobilenet is proposed to classify (224x224) images
        # down-sampling from 224 to 224/2**5 = 7
        # we only down-sampling 2 times, from 32 to 32/2**2 = 8
        self.mobilenet.features[0][0].stride = (1,1)
        self.mobilenet.features[2].block[1][0].stride = (1,1)
        self.mobilenet.features[4].block[1][0].stride = (1,1)
        # modify classifier
        self.mobilenet.classifier[3] = nn.Linear(1280, classes)
        # reduce conv layers
        del self.mobilenet.features[1]
        del self.mobilenet.features[2]
        del self.mobilenet.features[3]

    def forward(self, x):
        return self.mobilenet(x)
    
class MobileNetV3_small(nn.Module):
    def __init__(self, classes):
        super(MobileNetV3_small, self).__init__()
        self.mobilenet = torchvision.models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.mobilenet.features[0][0].stride = (1,1)
        self.mobilenet.features[1].block[0][0].stride = (1,1)
        self.mobilenet.features[2].block[1][0].stride = (1,1)
        self.mobilenet.classifier[3] = nn.Linear(1024, classes)
        del self.mobilenet.features[3]
        del self.mobilenet.features[4]
        

    def forward(self, x):
        return self.mobilenet(x)
    
class CNNCifar(nn.Module):
    def __init__(self, classes):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def test_img(net_g, datatest):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    device = torch.device('mps')
    data_loader = DataLoader(datatest, batch_size=1000)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        log_probs = net_g(data)
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy.item()
    
if __name__ == "__main__":
    device = torch.device('mps')

    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)

    #net = MobileNetV3(10).to(device)
    net = MobileNetV3_small(10).to(device)
    #print(net)
    #exit('test')
    net.train()
    """ net.load_state_dict(torch.load('./save/resnet_cifar2.pth'))
    print(test_img(net,dataset_train))
    
    print(net.state_dict())
    if os.path.isfile('./save/resnet_cifar.pth'):
        net.load_state_dict(torch.load('./save/resnet_cifar.pth'))
        print('load weight')
        print(net.state_dict()) 
        print(test_img(net,dataset_test)) """
    
    # train and update
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.8)
    loss_func = nn.CrossEntropyLoss()

    ldr_train = DataLoader(dataset_train, batch_size=2500, shuffle=True)

    epoch_loss = []
    epoch_test_acc = []
    epoch_train_acc = []
    for iter in range(100):
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(device), labels.to(device)
            net.zero_grad()
            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
            #print("epoch", iter, "batch", batch_idx, "loss:", loss.item())
        net.eval()
        with torch.no_grad():
            acc_test = test_img(net, dataset_test)
            acc_train = test_img(net, dataset_train)
        net.train()
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        epoch_test_acc.append(acc_test)
        epoch_train_acc.append(acc_train)
        print("epoch", iter+1, "loss:", sum(batch_loss)/len(batch_loss))
        print("epoch", iter+1, "train acc: ", acc_train)
        print("epoch", iter+1, "test acc: ", acc_test)
        if iter != 0 and iter % 10 == 0:
            torch.save(net.state_dict(), './save/mobilenet_small.pth')

    net.eval()

    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.ylabel('train_loss')
    plt.savefig('./save/mobilenet_small_loss.png')
    plt.close()

    plt.figure()
    plt.plot(range(len(epoch_test_acc)), epoch_test_acc)
    plt.ylabel('test_acc')
    plt.savefig('./save/mobilenet_small_test_acc.png')
    plt.close()

    plt.figure()
    plt.plot(range(len(epoch_train_acc)), epoch_train_acc)
    plt.ylabel('train_acc')
    plt.savefig('./save/mobilenet_small_train_acc.png')
    plt.close()