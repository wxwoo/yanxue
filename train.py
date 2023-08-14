import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from random import randint
import numpy as np
import subprocess
import multiprocessing
import concurrent.futures
from time import time
from math import sqrt

CHANNEL = 32
BLOCKNUM = 10
BOARDSIZE = 8
BATCH = 50
EPOCHS = 20
DATASIZE = 7200
DATAUSE = 2000
ROUNDLIMIT = 500
PROCESS = 3

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
optimizer = Adam(cnn.parameters(), weight_decay=1e-4)

stateData = torch.zeros(DATASIZE, 3, 8, 8, dtype=float)
policyData = torch.zeros(DATASIZE, 64, dtype=float)
valueData = torch.zeros(DATASIZE, 1, dtype=float)
policyLossFunc = nn.CrossEntropyLoss()
valueLossFunc = nn.MSELoss()

def calc(cood):
    return cood[0] * BOARDSIZE + cood[1]

def lossFunction(policyOutput, valueOutput, policyTarget, valueTarget):
    policyLoss = policyLossFunc(policyOutput, policyTarget)
    valueLoss = valueLossFunc(valueOutput, valueTarget)
    return policyLoss + valueLoss

def train():
    cnn.train()
    use = torch.zeros(DATASIZE)
    inputData = torch.zeros(DATAUSE,3,8,8)
    policyTargetData = torch.zeros(DATAUSE,64)
    valueTargetData = torch.zeros(DATAUSE,1)
    
    i = 0
    while i < DATAUSE:
        x = randint(0, DATASIZE - 1)
        if use[x] == 1:
            continue
        inputData[i] = stateData[x]
        policyTargetData[i] = policyData[x]
        valueTargetData[i] = valueData[x]
        use[x] = 1
        i += 1
        
    optimizer.zero_grad()
    for i in range(EPOCHS):
        policyLossAvg = 0.0
        valueLossAvg = 0.0
        print(f'epoch {i+1}:')

        for j in range(0, DATAUSE, BATCH):

            input = inputData[j:j+BATCH]
            policyTarget = policyTargetData[j:j+BATCH]
            valueTarget = valueTargetData[j:j+BATCH]

            policyOutput, valueOutput = cnn(input.to(device))

            policyLoss = policyLossFunc(policyOutput, policyTarget.to(device))
            valueLoss = valueLossFunc(valueOutput, valueTarget.to(device))
            loss = policyLoss + valueLoss
            policyLossAvg += float(policyLoss)
            valueLossAvg += float(valueLoss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'    policy loss: {policyLossAvg / (DATAUSE / BATCH)}')
        print(f'    value loss: {valueLossAvg / (DATAUSE / BATCH)}')
        print(f'    total loss: {(policyLossAvg + valueLossAvg) / (DATAUSE / BATCH)}')
        
    torch.save(cnn.state_dict(), r'D:/Desktop/yanxue/rescnn.pth')

def gen_cpp():
    cnt = 0
    cnn.eval()
    mcts = subprocess.Popen(r'./rawCNNMCTSselfmatch.exe', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    startTime = time()
    input = str(ROUNDLIMIT) + '\n'
    mcts.stdin.write(input.encode())
    mcts.stdin.flush()
    while cnt < DATASIZE:
        lst = cnt
        print('new game')
        output = mcts.stdout.readline()
        output = tuple(map(float, output.decode().strip().split(' ')))
        while int(output[0]) != -1:
            # print('o',output)
            input = ''
            if int(output[0]) == 0:
                stateInput = torch.tensor(output[1:],dtype=torch.float32)
                stateInput = stateInput.view(3, 8, 8)
                stateInput.unsqueeze_(0)
                policyOutput, valueOutput = cnn(stateInput.to(device))
                policyOutput = F.softmax(policyOutput, dim=-1)
                input = str(policyOutput.detach().to('cpu').numpy()) + str(round(float(valueOutput[0]), 7)) + '\n'
                # print('s', policyOutput.sum())
                # print('i',input)
                mcts.stdin.write(input.encode())
                mcts.stdin.flush()
            elif int(output[0]) == 1:
                if cnt < DATASIZE:
                    policyData[cnt] = torch.tensor(output[1:65],dtype=torch.float32)
                    stateData[cnt] = torch.tensor(output[65:65+3*8*8]).view(3,8,8)
                    # print(stateData[cnt])
                    # print(policyData[cnt].view(8, 8))
                    cnt += 1
                    print(f'{cnt} / {DATASIZE} ; time elapsed: {time() - startTime}\n', end='')
            elif output[0] == 2:
                print(f'turn {int(output[1])} finished')
            
            output = mcts.stdout.readline()
            # print(len(output.decode().strip().split(' ')))
            output = tuple(map(float, output.decode().strip().split(' ')))

        valueData[lst:cnt] = output[1]
        # print(valueData[lst:cnt])
        # input = '1\n'
        # mcts.stdin.write

    mcts.kill()

def gen_subProcess(queue,processid,startTime):
    cnt = 0
    subcnn = resCNN()
    subcnn.load_state_dict(torch.load(r'D:/Desktop/yanxue/rescnn.pth'))
    subcnn.to(device)
    subcnn.eval()
    mcts = subprocess.Popen(r'./rawCNNMCTSselfmatch.exe', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    subState = torch.zeros(60,3,8,8)
    subPolicy = torch.zeros(60,64)
    subValue = torch.zeros(60,1)
    input = str(ROUNDLIMIT) + '\n'
    mcts.stdin.write(input.encode())
    mcts.stdin.flush()
    while cnt < DATASIZE / PROCESS:
        cur = 0
        # print('new game')
        # input = str(ROUNDLIMIT) + ' 0\n'
        output = mcts.stdout.readline()
        output = tuple(map(float, output.decode().strip().split(' ')))
        while int(output[0]) != -1:
            input = ''
            if int(output[0]) == 0:
                stateInput = torch.tensor(output[1:])
                stateInput = stateInput.view(3, 8, 8)
                stateInput.unsqueeze_(0)
                policyOutput, valueOutput = cnn(stateInput.to(device))
                policyOutput = F.softmax(policyOutput, dim=-1)
                input = str(policyOutput.detach().to('cpu').numpy()) + str(float(valueOutput[0])) + '\n'
                mcts.stdin.write(input.encode())
                mcts.stdin.flush()
            elif int(output[0]) == 1:
                if cnt < DATASIZE / PROCESS:
                    subPolicy[cur] = torch.tensor(output[1:65])
                    subState[cur] = torch.tensor(output[65:65+3*8*8]).view(3,8,8)
                    # print(subPolicy[cnt].view(8, 8))
                    cnt += 1
                    cur += 1
                # print(f'{cnt} / {DATASIZE} ; time elapsed: {time() - startTime}\n', end='')
            elif output[0] == 2:
                print(f'\x1b[{PROCESS - processid +1}Aprocess {processid} turn {int(output[1])} finished ; {cnt} / {int(DATASIZE / PROCESS)} ; time elapsed: {time() - startTime}\r\x1b[{PROCESS - processid + 1}B', end = '')
            
            output = mcts.stdout.readline()
            output = tuple(map(float, output.decode().strip().split(' ')))

        subValue[:cur] = output[1]
        queue.put((subState, subPolicy, subValue, cur))

    mcts.kill()

def gen_mainProcess():
    startTime = time()

    queue = multiprocessing.Queue()
    processList = []
    for i in range(PROCESS):
        p = multiprocessing.Process(target=gen_subProcess, args=(queue,i,startTime,))
        p.start()
        processList.append(p)
        print(f'process {i} start')
    print('')

    cnt = 0
    while cnt < DATASIZE:
        data = queue.get()
        stateData[cnt:cnt+data[3]] = data[0][:data[3]]
        policyData[cnt:cnt+data[3]] = data[1][:data[3]]
        valueData[cnt:cnt+data[3]] = data[2][:data[3]]
        cnt += data[3]
        print(f'\x1b[1A{cnt} / {DATASIZE} ; time elapsed: {time() - startTime}\r\x1b[1B', end='')

    for p in processList:
        p.join()

if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=7)
    multiprocessing.freeze_support()
    while 1 :
        with open('./iteration.txt', 'r') as f:
            times = int(f.read())
        print(f'iteration {times}:')
        print('self-matching:')
        # gen_py()    # unable to use, archive at train_py.py
        # gen_cpp()
        gen_mainProcess()
        print('train start:')
        train()
        archivePath = 'D:/Desktop/yanxue/rescnn_archive/rescnn-iteration' + str(times) +'.pth'
        torch.save(cnn.state_dict(), archivePath)
        with open('./iteration.txt', 'w') as f:
            f.write(str(times + 1))