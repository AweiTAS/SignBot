import torch
import torch.nn as nn
from torch import Tensor
from typing import Type
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

ta = 90
dim = (224, 224)
dim3 = (100, 224, 224)
batch_size = 4

class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return  out





class ResNet18(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        num_classes: int = 10
    ) -> None:
        super(ResNet18, self).__init__()
        layers = [2, 2, 2, 2]
        self.expansion = 1
        
        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7, 
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)
    def _make_layer(
        self, 
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            BasicBlock(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(BasicBlock(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # The spatial dimension of the final layer's feature 
        # map should be (7, 7) for all ResNets.
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    



#constructing training set DataLoader
class ArrowData(Dataset):
    def __init__(self, path, csv, device) -> None:
        super().__init__()
        temp = np.zeros((ta, 224, 224))

        mat = pd.read_csv(path + csv)
        mat.iloc[:, 0] = path + mat.iloc[:, 0]

        for i in range(0, ta):
            img = cv2.imread(mat["path"][i])[:,:,1]/256
            img = cv2.resize(img, dim)
            temp[i] = img
        self.x = torch.tensor(temp.reshape(ta, 1, 224, 224), dtype=torch.float32).to(device)
        self.y = torch.tensor(mat.dir[:ta]).to(device)

        self.n_sample = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.n_sample

#constructing test set DataLoader
class ArrowTestData(Dataset):
    def __init__(self, path, csv, device) -> None:
        super().__init__()
        temp = np.zeros((100-ta, 224, 224))
        
        mat = pd.read_csv(path + csv)
        mat.iloc[:, 0] = path + mat.iloc[:, 0]

        for i in range(ta, 100):
            img = cv2.imread(mat["path"][i])[:,:,1]/256
            img = cv2.resize(img, dim)
            temp[i-ta] = img
        self.x = torch.tensor(temp.reshape(100-ta, 1, 224, 224), dtype=torch.float32).to(device)
        self.y = torch.tensor(mat.dir[ta:].values).to(device)

        self.n_sample = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.n_sample








if __name__== "__main__" :

    #instantiate a dataloader
    epochs = 200
    lr = 0.001

    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    path = 'D:/workspace/Python/engPic/arrow/'
    csv = 'arrowdata.csv'

    dataloader = DataLoader(ArrowData(path, csv, device), batch_size=batch_size, shuffle=True)
    Testdataloader = DataLoader(ArrowTestData(path, csv, device), batch_size=1, shuffle=False)

    model = ResNet18(img_channels = 1, num_classes = 4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        for batch in dataloader:
            optimizer.zero_grad()

            x, y = batch
            output = model(x)
            loss = criterion(output, y)
            
            loss.backward()
            optimizer.step()

            output = torch.argmax(output, dim=1)

            total_acc += torch.sum(output == y)
            total_loss += loss*batch_size
        print(f'Epoch:{epoch+1}\t Loss:{total_loss}')
        print(f'\t Correct:{total_acc}')

    for batch in Testdataloader:
        x, y = batch
        output = torch.argmax(model(x))
        print(f'output:{output}\t y:{y}')

    torch.save(model.state_dict(), "D:/workspace/Python/models/ArrowNet")