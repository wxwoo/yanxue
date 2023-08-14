import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from random import randint
import numpy

CHANNEL = 32
BLOCKNUM = 10
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
cnn.load_state_dict(torch.load(r'D:/Desktop/yanxue/rescnn.pth'))
cnn.to(device)
cnn.eval()


numpy.set_printoptions(suppress=True, precision=3)
while 1:
    read = torch.zeros(3,8,8)
    for i in range(8):
        raw = input()
        row = tuple(map(int, raw.strip().split(' ')))        
        for j in range(8):
            if row[j] == 1:
                read[0,i,j] = 1
            elif row [j] == -1:
                read[1,i,j] = 1
    f = int(input())
    for i in range(8):
        for j in range(8):
            read[2,i,j] = f
    read = read.to(device)
    read = read.unsqueeze(0)
    policy, value = cnn(read)
    npolicy = F.softmax(policy,dim=-1).detach().to('cpu').numpy()
    for i in range(8):
        for j in range(8):
            print(npolicy[0][i*8+j],end = ' ')
        print('')
    print(float(value.detach()))
