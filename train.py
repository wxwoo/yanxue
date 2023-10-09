import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from random import randint, shuffle
import numpy as np
import subprocess
import torch.multiprocessing as mp
# import concurrent.futures
from time import time
from math import sqrt

CHANNEL = 128
BLOCKNUM = 20
BOARDSIZE = 8
BATCH = 400
EPOCHS = 2
DATASIZE = 10800
DATAUSE = 4000 # total use *2
ITERATIONLIMIT = 167
SEARCHDEPTH = 6
PROCESS = 10
DISPLAY_INFO = 0
ITERAION_ROUND = 30

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

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stateData = torch.zeros(DATASIZE, 3, 8, 8, dtype=float)
    policyData = torch.zeros(DATASIZE, 64, dtype=float)
    valueData = torch.zeros(DATASIZE, 1, dtype=float)

def calc(cood):
    return cood[0] * BOARDSIZE + cood[1]

# def lossFunction(policyOutput, valueOutput, policyTarget, valueTarget):
#     policyLoss = policyLossFunc(policyOutput, policyTarget)
#     valueLoss = valueLossFunc(valueOutput, valueTarget)
#     return policyLoss + valueLoss

def train(times:int):
    cnn = resCNN()
    cnn.load_state_dict(torch.load(r'./rescnn.pth', map_location=device))
    cnn.to(device)
    cnn.train()
    optimizer = Adam(cnn.parameters(), weight_decay=1e-4)
    inputData = torch.zeros(DATAUSE*2,3,8,8)
    policyTargetData = torch.zeros(DATAUSE*2,64)
    valueTargetData = torch.zeros(DATAUSE*2,1)
    policyLossFunc = nn.CrossEntropyLoss()
    valueLossFunc = nn.MSELoss()
    policyLossFunc = policyLossFunc.to(device)
    valueLossFunc = valueLossFunc.to(device)
    
    use = list(range(DATASIZE))
    shuffle(use)
    for i in range(DATAUSE):
        x = use[i]
        inputData[i] = stateData[x]
        policyTargetData[i] = policyData[x]
        valueTargetData[i] = valueData[x]
        
        inputTemp = stateData[x].clone()
        policyTemp = policyData[x].clone()
        policyTemp = policyTemp.view(8,8)
        if randint(0,1) == 1:
            inputTemp = inputTemp.flip(dims=[1])
            policyTemp = policyTemp.flip(dims=[0])
        else:
            inputTemp = inputTemp.flip(dims=[2])
            policyTemp = policyTemp.flip(dims=[1])
        rotateAngle = randint(1,3)
        inputTemp = inputTemp.rot90(k=rotateAngle,dims=[1,2])
        policyTemp = policyTemp.rot90(k=rotateAngle,dims=[0,1])
        
        # if i % 40 == 0 and DISPLAY_INFO == 1:
        #     print(stateData[x])
        #     print(policyData[x])
        #     print(inputTemp)
        #     print(policyTemp.reshape(64))
        #     print('')
        
        inputData[i+DATAUSE] = inputTemp.clone()
        policyTargetData[i+DATAUSE] = policyTemp.reshape(64).clone()
        valueTargetData[i+DATAUSE] = valueData[x]
        
        
    optimizer.zero_grad()
    for i in range(EPOCHS):
        policyLossAvg = 0.0
        valueLossAvg = 0.0
        print(f'epoch {i+1}:')

        for j in range(0, DATAUSE*2, BATCH):

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
        
    torch.save(cnn.state_dict(), r'./rescnn.pth')
    archivePath = './rescnn_archive/rescnn-iteration' + str(times) +'.pth'
    torch.save(cnn.state_dict(), archivePath)

def gen_cpp():
    cnn = resCNN()
    cnn.load_state_dict(torch.load(r'./rescnn.pth', map_location=device))
    cnn.to(device)
    cnt = 0
    cnn.eval()
    mcts = subprocess.Popen(r'./rawCNNMCTSselfmatch.out', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    startTime = time()
    input = str(SEARCHDEPTH) + ' ' + str(ITERATIONLIMIT) + '\n'
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

def gen_subProcess(processid, queue, startTime):
    with torch.no_grad():
        cnt = 0
        subcnn = resCNN()
        subcnn.load_state_dict(torch.load(r'./rescnn.pth'))
        gpuid = processid % torch.cuda.device_count()
        subcnn = subcnn.cuda(gpuid)
        subcnn.eval()
        mcts = subprocess.Popen(r'./rawCNNMCTSselfmatch.out', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        subState = torch.zeros(60,3,8,8)
        subPolicy = torch.zeros(60,64)
        subValue = torch.zeros(60,1)
        input = str(SEARCHDEPTH) + ' ' + str(ITERATIONLIMIT) + '\n'
        mcts.stdin.write(input.encode())
        mcts.stdin.flush()
        genCnt = 0
        if DISPLAY_INFO == 1:
            print(f'process {processid} using gpu {gpuid}')
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
                    policyOutput, valueOutput = subcnn(stateInput.cuda(gpuid))
                    policyOutput = F.softmax(policyOutput, dim=-1)
                    input = str(policyOutput.detach().to('cpu').numpy()) + str(float(valueOutput[0])) + '\n'
                    mcts.stdin.write(input.encode())
                    mcts.stdin.flush()
                    genCnt += 1
                elif int(output[0]) == 1:
                    if cnt < DATASIZE / PROCESS:
                        subPolicy[cur] = torch.tensor(output[1:65])
                        subState[cur] = torch.tensor(output[65:65+3*8*8]).view(3,8,8)
                        cnt += 1
                        cur += 1
                    # print(f'{cnt} / {DATASIZE} ; time elapsed: {time() - startTime}\n', end='')
                elif output[0] == 2:
                    if DISPLAY_INFO == 1:
                        print(f'process {processid} turn {int(output[1])} finished ; {cnt} / {int(DATASIZE / PROCESS)} ; generate count: {genCnt} ; time elapsed: {round(time() - startTime, 3)}')
                    genCnt = 0
                    # print(f'\x1b[{PROCESS - processid}Aprocess {processid} turn {int(output[1])} finished ; {cnt} / {int(DATASIZE / PROCESS)} ; time elapsed: {time() - startTime}\r\x1b[{PROCESS - processid}B', end = '')
                
                output = mcts.stdout.readline()
                output = tuple(map(float, output.decode().strip().split(' ')))

            subValue[:cur] = output[1]
            queue.put((subState, subPolicy, subValue, cur))

    mcts.kill()

def gen_mainProcess():
    startTime = time()

    queue = mp.Queue()
    processList = []
    for i in range(PROCESS):
        p = mp.Process(target=gen_subProcess, args=(i,queue,startTime,))
        p.start()
        processList.append(p)
        # print(f'process {i} start')
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

def gen_mainProcess_new():
    startTime = time()
    queue = mp.Manager().Queue(maxsize=DATASIZE)
    mp.spawn(gen_subProcess, nprocs=PROCESS, args=(queue, startTime))
    
    cnt = 0
    diff = 0
    while cnt < DATASIZE and not queue.empty():
        data = queue.get()
        stateData[cnt:cnt+data[3]] = data[0][:data[3]]
        policyData[cnt:cnt+data[3]] = data[1][:data[3]]
        valueData[cnt:cnt+data[3]] = data[2][:data[3]]
        diff += int(valueData[cnt][0])
        cnt += data[3]
    print(f'time elasped: {time() - startTime}')
    print(f'black win - white win: {diff}')

if __name__ == '__main__':
    # mp.set_start_method("spawn")
    np.set_printoptions(suppress=True, precision=7)
    mp.freeze_support()
    print(f'gpu count:{torch.cuda.device_count()}')
    # while 1 :
    for i in range(ITERAION_ROUND):
        with open('./iteration.txt', 'r') as f:
            times = int(f.read())
        print(f'iteration {times}:')
        print('self-matching:')
        # gen_py()    # unable to use, archive at train_py.py
        # gen_cpp()
        # gen_mainProcess()
        gen_mainProcess_new()
        print('train start:')
        train(times)
        with open('./iteration.txt', 'w') as f:
            f.write(str(times + 1))
    