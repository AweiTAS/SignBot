import torch
from torch.utils.data import DataLoader
import ResNet18

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    path = 'D:/workspace/Python/engPic/arrow/'
    csv = 'arrowdata.csv'
    batch_size = 4

    model = ResNet18.ArrowNet().to(device)
    model.load_state_dict(torch.load("D:/workspace/Python/models/ArrowNet"))
    model.eval()

    dataloader = DataLoader(ResNet18.ArrowData(path, csv, device), batch_size=batch_size, shuffle=True)
    Testdataloader = DataLoader(ResNet18.ArrowTestData(path, csv, device), batch_size=1, shuffle=False)

    total_acc = 0
    for batch in dataloader:

        x, y = batch
        
        print(x.shape)
        print(type(x))
        output = model(x)

        output = torch.argmax(output, dim=1)

        total_acc += torch.sum(output == y)
    print(f'\t Correct:{total_acc}')