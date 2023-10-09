import numpy as np
# import random as rand
import subprocess
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
# import tkinter as tk
# from math import sqrt,log
# import sys

CHANNEL = 128
BLOCKNUM = 20
BOARDSIZE = 8
DISPLAY_INFO = 0
ITERATIONLIMIT = 100
SEARCHDEPTH = 10
BLACK_ITERATION = 90
WHITE_ITERATION = 90

def calc(cood):
    return cood[0] * BOARDSIZE + cood[1]

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
            nn.Softmax(dim=1)
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


class CNNMCTS:
    def __init__(self, iteration):
        self.cnn = resCNN()
        self.cnn.load_state_dict(torch.load(f'./rescnn_archive/rescnn-iteration{iteration}.pth'))
        self.cnn = self.cnn.cuda(0)
        self.cnn.eval()
    
    def getBestMove(self, state, player, timeIterations, turn):
        mcts = subprocess.Popen('./rawCNNMCTS.out', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        input = '' 
        for i in range(8):
            for j in range(8):
                input += str(state.board[i, j]) + ' '
        input += str(player) + ' ' + str(SEARCHDEPTH) + ' ' + str(timeIterations) + ' ' + str(turn) + '\n'
        mcts.stdin.write(input.encode())
        mcts.stdin.flush()
        output = mcts.stdout.readline()
        output = tuple(map(float, output.decode().strip().split(' ')))
        while int(output[0]) != -1:
        # print('o',output)
            input = ''
            if int(output[0]) == 0:
                stateInput = torch.tensor(output[1:],dtype=torch.float32)
                stateInput = stateInput.view(3, 8, 8)
                stateInput.unsqueeze_(0)
                policyOutput, valueOutput = self.cnn(stateInput.cuda(0))
                policyOutput = F.softmax(policyOutput, dim=-1)
                input = str(policyOutput.detach().to('cpu').numpy()) + str(round(float(valueOutput[0]), 7)) + '\n'
                mcts.stdin.write(input.encode())
                mcts.stdin.flush()
        
            output = mcts.stdout.readline()
            output = tuple(map(float, output.decode().strip().split(' ')))
            
        return (int(output[-2]), int(output[-1]))
    

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

def cmp():
    c_state = GameState()
    BlackMCTS = CNNMCTS(BLACK_ITERATION)
    WhiteMCTS = CNNMCTS(WHITE_ITERATION)
    BlackWin = 0
    WhiteWin = 0
    Draw = 0
    
    while 1:
        
        c_state = st_state.copy()
        c_state.print()
        turn = 1
        startTime = time()
        while not c_state.isTerminal():
            
            print(f'black({BLACK_ITERATION}) turn')
            if len(c_state.getValidMoves(1)) > 0:
                c_state.makeMove(BlackMCTS.getBestMove(c_state, 1, ITERATIONLIMIT, turn), 1)
                turn += 1
            else:
                print('black has no available move')
                
            c_state.print()
            if c_state.isTerminal():
                break
            
            print(f'white({WHITE_ITERATION}) turn')
            if len(c_state.getValidMoves(-1)) > 0:
                c_state.makeMove(WhiteMCTS.getBestMove(c_state, -1, ITERATIONLIMIT, turn), -1)
                turn += 1
            else:
                print('white has mo available move')
                
            c_state.print()
                
        winner = c_state.getWinner()
        if winner == 1:
            BlackWin += 1
            print('black win')
        elif winner == -1:
            WhiteWin += 1
            print('white win')
        else:
            Draw += 1
            print('draw')
            
        print(f'black({BLACK_ITERATION}) win : {BlackWin}')
        print(f'white({WHITE_ITERATION}) win : {WhiteWin}')
        print(f'Draw : {Draw}')
        print(f'time elasped : {time() - startTime}')
    
    
if __name__ == '__main__':
    st_state = GameState()
    
    print(f'black iteration:{BLACK_ITERATION}')
    print(f'white iteration:{WHITE_ITERATION}')
    
    cmp()
    