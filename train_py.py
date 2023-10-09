
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from random import randint
import numpy as np
# import subprocess
# import multiprocessing
# import concurrent.futures
from time import time
from math import sqrt

CHANNEL = 256
BLOCKNUM = 40
BOARDSIZE = 8
BATCH = 50
EPOCHS = 20
DATASIZE = 7200
DATAUSE = 2000
ROUNDLIMIT = 500
PROCESS = 3
OUTPUT_INFO = 1

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
cnn.load_state_dict(torch.load(r'./rescnn.pth'))
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

        if OUTPUT_INFO:
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

        if OUTPUT_INFO:
            print(f'    policy loss: {policyLossAvg / (DATAUSE / BATCH)}')
            print(f'    value loss: {valueLossAvg / (DATAUSE / BATCH)}')
            print(f'    total loss: {(policyLossAvg + valueLossAvg) / (DATAUSE / BATCH)}')
        
    torch.save(cnn.state_dict(), r'./rescnn.pth')


class GameState:
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=np.int8) # 0 ~ 7
        self.board[3, 3] = self.board[4, 4] = -1
        self.board[3, 4] = self.board[4, 3] = 1 #Black 1 White -1
        self.history = []
        
    def copy(self):
        state = GameState()
        state.board = np.copy(self.board)
        state.history = self.history[:]
        return state
        
    def makeMove(self, move, player):
        self.history.append(move)
        self.board[move] = player
        for d in (-1, 0, 1):
            for e in (-1, 0, 1):
                if d == 0 and e == 0:
                    continue
                x, y = move
                x += d
                y += e
                to_flip = []
                while x >= 0 and y >= 0 and x < 8 and y < 8 and self.board[x, y] == -player:
                    to_flip.append((x, y))
                    x += d
                    y += e
                if x >= 0 and y >= 0 and x < 8 and y < 8 and self.board[x, y] == player:
                    for f in to_flip:
                        self.board[f] = player
    
    def isValid(self, move, player):
        if self.board[move] != 0:
            return False
        for d in (-1, 0, 1):
            for e in (-1, 0, 1):
                if d == 0 and e == 0:
                    continue
                x, y = move
                x += d
                y += e
                num = 0
                while x >= 0 and y >= 0 and x < 8 and y < 8 and self.board[x, y] == -player:
                    x += d
                    y += e
                    num += 1
                if num > 0 and x >= 0 and y >= 0 and x < 8 and y < 8 and self.board[x, y] == player:
                    return True
        return False
    
    def getValidMoves(self, player):
        moves = []
        for i in range(8):
            for j in range(8):
                if self.isValid((i, j), player):
                    moves.append((i, j))
        return moves
    
    def isTerminal(self):
        if len(self.getValidMoves(1)) > 0:
            return False
        if len(self.getValidMoves(-1)) > 0:
            return False
        return True
    
    def getWinner(self):
        count = np.sum(self.board)
        if count > 0:
            return 1
        elif count < 0:
            return -1
        else:
            return 0
    
    def getScore(self, player):
        cnt = 0
        for i in range(8):
            for j in range(8):
                if self.board[i,j] == player:
                    cnt += 1
        return cnt
    
    def print(self):
        print('  ',end='')
        for i in range(8):
            print(i,end=' ')
        print('')
        for i in range(8):
            print(i,end=' ')
            for j in range(8):
                if self.board[i,j] == 1:
                    print('#',end=' ')
                elif self.board[i,j] == -1:
                    print('O',end=' ')
                else:
                    print('.',end=' ')
            print('')

PUCT_CONSTANT = 1
class MCTSNode:
    def __init__(self, state:GameState, player):
        self.state:GameState = state.copy()
        self.parent:MCTSNode = None
        self.children = []
        self.unexploredMoves = state.getValidMoves(player)
        self.player = player
        self.n = 0
        self.v = 0.0
        self.p = 0.0
        self.policyPredict = torch.zeros(64)
        self.valuePredict = 0.0
        if type == 2:
            input = torch.zeros(3,8,8)
            for i in range(8):
                for j in range(8):
                    if state.board[i,j] == 1:
                        input[0,i,j] = 1
            for i in range(8):
                for j in range(8):
                    if state.board[i,j] == -1:
                        input[1,i,j] = 1
            for i in range(8):
                for j in range(8):
                    input[2,i,j] = player
            input.unsqueeze_(0)
            output = cnn(input.to(device))
            self.policyPredict = F.softmax(output[0][0], dim=-1)
            self.valuePredict = float(output[1][0])
        
    def expand(self):
        if len(self.unexploredMoves) <= 0:
            return None
        
        move = self.unexploredMoves.pop()
        newState = self.state.copy()
        newState.makeMove(move, self.player)

        child = None
        if len(newState.getValidMoves(-self.player)) > 0:
            child = MCTSNode(newState, -self.player)
        else:
            child = MCTSNode(newState, self.player)
        child.parent = self
        child.p = float(self.policyPredict[calc(move)])
        self.children.append(child)
        return child

    def puct(self, player):
        Q = self.v / self.n
        U = PUCT_CONSTANT * self.p * sqrt(self.parent.n + 1) / (self.n + 1)
        if player == -1:
            Q = -Q
        return Q + U
    
    def select(self, player):
        return max(self.children, key=lambda c: c.puct(player))
    
    def backpropagate(self, v):
        self.n += 1
        self.v += v
        if self.parent:
            self.parent.backpropagate(v)
           
class CNNMCTS:
    def __init__(self):
        return

    def CNNMCTSBestMove(self, state, player, timeIterations):
        rootNode = MCTSNode(state, player)

        for i in range(timeIterations):
            node = rootNode
        
            while len(node.unexploredMoves) == 0 and node.state.isTerminal() == False:
                if len(node.children) > 0:
                    node = node.select(player)
                else:
                    break
            
            if len(node.unexploredMoves) > 0 and node.state.isTerminal() == False:
                node = node.expand()

            if node.state.isTerminal() == False:
                node.backpropagate(node.valuePredict)
            else:
                node.backpropagate(node.state.getWinner())
            
        bestChild = rootNode.children[0]
        for child in rootNode.children:
            if child.n > bestChild.n:
                bestChild = child
        return bestChild.state.history[-1]

def gen_py():
    MCTS = CNNMCTS()
    cnt = 0
    cnn.eval()
    while cnt < DATASIZE:
        c_state = GameState()
        currentPlayer = 1
        cur = 0
        lst = cnt

        while c_state.isTerminal() == 0:
            if len(c_state.getValidMoves(currentPlayer)) <= 0:
                currentPlayer = -currentPlayer
                continue
            bestMove = MCTS.CNNMCTSBestMove(c_state, currentPlayer, ROUNDLIMIT)
            cur += 1
            if 5 <= cur and cur <= 54 and cnt < DATASIZE:
                for i in range(8):
                    for j in range(8):
                        if c_state.board[i,j] == 1:
                            stateData[cnt,0,i,j] = 1
                for i in range(8):
                    for j in range(8):
                        if c_state.board[i,j] == -1:
                            stateData[cnt,1,i,j] = 1
                for i in range(8):
                    for j in range(8):
                        stateData[cnt,2,i,j] = currentPlayer
                policyData[cnt] = calc(bestMove)
                cnt += 1

            c_state.makeMove(bestMove, currentPlayer)
            currentPlayer = -currentPlayer
        
        valueData[lst:cnt] = c_state.getWinner()
        if OUTPUT_INFO:
            print(f'{cnt} / {DATASIZE}\r', end='')

    if OUTPUT_INFO:
        print('')


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=7)
    # multiprocessing.freeze_support()
    times = 0
    while 1 :
        if OUTPUT_INFO:
            print(f'iteration {times}:')
            print('self-matching:')

        gen_py()
        # gen_cpp()
        # gen_mainProcess() # in train.py

        if OUTPUT_INFO:
            print('train start:')

        train()
        # archivePath = 'D:/Desktop/yanxue/rescnn_archive/rescnn-iteration' + str(times) +'.pth'
        # torch.save(cnn.state_dict(), archivePath)