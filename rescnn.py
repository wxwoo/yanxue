import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from random import randint

CHANNEL = 128
BLOCKNUM = 20
BOARDSIZE = 8

class resBlock(nn.Module):
    def __init__(self, x):
        super(resBlock, self).__init__()
        self.resBlock = nn.Sequential(
            nn.Conv2d(x, x, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(x),
            nn.ReLU(inplace=True),
            nn.Conv2d(x, x, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(x)
        )
        self.relu = nn.ReLU(inplace=True)
            
    def forward(self, x):
        shortCut = x
        out = self.resBlock(x)
        out += shortCut
        out = self.relu(out)
        return out
    
class resCNN(nn.Module):
    def __init__(self):
        super(resCNN, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(3, CHANNEL, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(CHANNEL),
            nn.ReLU(inplace=True)
        )
        self.resnet = nn.Sequential()
        for i in range(BLOCKNUM):
            self.resnet.add_module(str(i),resBlock(CHANNEL))
        
        self.ph = nn.Sequential(
            nn.Conv2d(CHANNEL, 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(BOARDSIZE*BOARDSIZE*2, BOARDSIZE*BOARDSIZE),
            # nn.Softmax(dim=1)
        )
        self.vh = nn.Sequential(
            nn.Conv2d(CHANNEL, 1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(BOARDSIZE*BOARDSIZE, CHANNEL),
            nn.ReLU(inplace=True),
            nn.Linear(CHANNEL, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        model = self.input(x)
        model = self.resnet(model)
        p = self.ph(model)
        v = self.vh(model)
        return p, v

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cnn = resCNN()
cnn.to(device)

archivePath = './rescnn_archive/rescnn-iteration0.pth'
torch.save(cnn.state_dict(), archivePath)

with open('./iteration.txt', 'w') as f:
    f.write('1')

torch.save(cnn.state_dict(), r'./rescnn.pth')


print(device)
def count_parameters(model):
    return sum(p.numel() for p in cnn.parameters() if p.requires_grad)
print('parameters_count:',count_parameters(cnn))
import torchsummary
torchsummary.summary(cnn, input_size=(3, BOARDSIZE, BOARDSIZE))

# from random import randint

# input = torch.zeros(3,8,8)
# for i in range(8):
#     for j in range(8):
#         input[0,i,j] = randint(0,1)
# for i in range(8):
#     for j in range(8):
#         if input[0,i,j] == 0:
#             input[1,i,j] = randint(0,1)
# for i in range(8):
#     for j in range(8):
#         input[2,i,j] = 1
# input = input.to(device)
# input = input.unsqueeze(0)
# print(cnn(input))
