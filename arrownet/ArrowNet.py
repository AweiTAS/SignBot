import torch
import torch.nn as nn
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

ta = 90
dim = (32, 32)
dim3 = (100, 32, 32)
batch_size = 4

class ArrowNet(nn.Module):
    def __init__(self) -> None:
        super(ArrowNet, self).__init__()
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(4, 4)

        self.c1 = nn.Conv2d(1, 10, 3, padding=1)
        self.c2 = nn.Conv2d(10, 32, 3, padding=1)
        self.f  = nn.Flatten()
        self.l1 = nn.Linear(128, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 4)
        #self.o = nn.Softmax(dim=1)

    def forward(self, x):
        outc1 = self.relu(self.mp(self.c1(x)))
        res = self.mp(self.mp(x))
        outc2 = self.relu(self.mp(self.c2(outc1)) + res)
        res = self.f(outc2)
        outl1 = self.relu(self.l1(self.f(outc2)))
        outl2 = self.relu(self.l2(outl1) + res)
        t = self.l3(outl2)
        #out = self.o(self.l3(outl2))
        out = self.l3(outl2)
        return out

#constructing training set DataLoader
class ArrowData(Dataset):
    def __init__(self, path, csv, device) -> None:
        super().__init__()
        temp = np.zeros((ta, 32, 32))

        mat = pd.read_csv(path + csv)
        mat.iloc[:, 0] = path + mat.iloc[:, 0]

        for i in range(0, ta):
            img = cv2.imread(mat["path"][i])[:,:,1]/256
            img = cv2.resize(img, dim)
            temp[i] = img
        self.x = torch.tensor(temp.reshape(ta, 1, 32, 32), dtype=torch.float32).to(device)
        self.y = torch.tensor(mat.dir[:ta]).to(device)

        self.n_sample = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.n_sample

class SignBotData(Dataset):
    def __init__(self, path, csv, device) -> None:
        super().__init__()

        mat = pd.read_csv(path + csv)
        mat.iloc[:, 0] = path + mat.iloc[:, 0]
        self.x = mat.filename[:]
        self.t = mat.target[:]
        self.d = mat.dir[:]

        self.n_sample = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.d[index]
    def __len__(self):
        return self.n_sample



#constructing test set DataLoader
class ArrowTestData(Dataset):
    def __init__(self, path, csv, device) -> None:
        super().__init__()
        temp = np.zeros((100-ta, 32, 32))
        
        mat = pd.read_csv(path + csv)
        mat.iloc[:, 0] = path + mat.iloc[:, 0]

        for i in range(ta, 100):
            img = cv2.imread(mat["path"][i])[:,:,1]/256
            img = cv2.resize(img, dim)
            temp[i-ta] = img
        self.x = torch.tensor(temp.reshape(100-ta, 1, 32, 32), dtype=torch.float32).to(device)
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

    model = ArrowNet().to(device)
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