import numpy as np
import random as rand
import subprocess
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk
from math import sqrt,log

CHANNEL = 32
BLOCKNUM = 10
BOARDSIZE = 8

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cnn = resCNN()
cnn.load_state_dict(torch.load(r'D:/Desktop/yanxue/rescnn.pth'))
cnn.to(device)
cnn.eval()

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

UCT_CONSTANT = sqrt(2)
PUCT_CONSTANT = 1
class MCTSNode:
    def __init__(self, state:GameState, player, type):
        self.state:GameState = state.copy()
        self.parent:MCTSNode = None
        self.children = []
        self.unexploredMoves = state.getValidMoves(player)
        self.player = player
        self.type = type
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
            child = MCTSNode(newState, -self.player, self.type)
        else:
            child = MCTSNode(newState, self.player, self.type)
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

    def uct(self, player):
        exploitation = self.v / self.n
        exploration = UCT_CONSTANT * sqrt(log(self.parent.n) / self.n)
        if player != self.parent.player:
            exploitation = -exploitation
        return exploitation + exploration
    
    def select(self, player):
        if self.type == 1:
            return max(self.children, key=lambda c: c.uct(player))
        elif self.type == 2:
            return max(self.children, key=lambda c: c.puct(player))
    
    def backpropagate(self, v):
        self.n += 1
        self.v += v
        if self.parent:
            self.parent.backpropagate(v)

    def randomPlayout(self, rootPlayer, fieldKnowledge):
        currentPlayer = self.player
        state = self.state.copy()
        while not state.isTerminal():
            moves = state.getValidMoves(currentPlayer)
            if len(moves) <= 0:
                currentPlayer = -currentPlayer
                continue

            randomIndex = rand.randint(0, len(moves) - 1)
            move = moves[randomIndex]
            if fieldKnowledge == 1:
                if (move[0] <= 1 or move[0] >= 6) and (move[1] <= 1 or move[1] >= 6):
                    randomIndex = rand.randint(0, len(moves) - 1)
                    move = moves[randomIndex]
                for i in (0,7):
                    for j in (0,7):
                        if state.isValid((i, j), currentPlayer):
                            move = (i, j)

            state.makeMove(move, currentPlayer)
            currentPlayer = -currentPlayer
        
        winner = state.getWinner()
        if winner == rootPlayer:
            self.backpropagate(1)
        elif winner == -rootPlayer:
            self.backpropagate(-1)
        else:
            self.backpropagate(0)
            
class CNNMCTS:
    def __init__(self):
        return
    
    # -2 : py mcts with knowledge
    # -1 : py mcts without knowledge
    #  0 : cpp mcts without knowledge
    #  1 : cpp mcts with knowledge
    #  2 : py cnn-mcts
    #  3 : cpp cnn-mcts
    def getBestMove(self, state, player, timeIterations, knowledge, turn):
        if knowledge == 2:
            return self.CNNMCTSBestMove(state, player, timeIterations)
        if knowledge < 0:
            return self.rawMCTSBestMove(state, player, timeIterations, 1-knowledge)
        if knowledge == 3:
            mcts = subprocess.Popen(r'./rawCNNMCTS.exe', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            input = ''
            for i in range(8):
                for j in range(8):
                    input += str(state.board[i, j]) + ' '
            input += str(player) + ' ' + str(timeIterations) + ' ' + str(turn) + '\n'
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
                    policyOutput, valueOutput = cnn(stateInput.to(device))
                    policyOutput = F.softmax(policyOutput, dim=-1)
                    input = str(policyOutput.detach().to('cpu').numpy()) + str(round(float(valueOutput[0]), 7)) + '\n'
                    mcts.stdin.write(input.encode())
                    mcts.stdin.flush()
            
                output = mcts.stdout.readline()
                output = tuple(map(float, output.decode().strip().split(' ')))

            print((int(output[-2]), int(output[-1])))
            return (int(output[-2]), int(output[-1]))
        if  knowledge == 0 or knowledge == 1:
            input = ''
            for i in range(8):
                for j in range(8):
                    input += str(state.board[i, j]) + ' '
            input += str(player) + ' ' + str(timeIterations) + ' ' + str(knowledge) + ' 0'
            result = subprocess.run(['./rawMCTS.exe'], input=input.encode('utf-8'), stdout=subprocess.PIPE)
            stdout = tuple(map(int, result.stdout.decode('utf-8').strip().split(' ')))
            print((stdout[-2], stdout[-1]))
            return (stdout[-2], stdout[-1])

    def CNNMCTSBestMove(self, state, player, timeIterations):
        rootNode = MCTSNode(state, player, 2)
        # for i in range(BOARDSIZE):
        #     print(rootNode.policyPredict[i*BOARDSIZE:i*BOARDSIZE+7])
        # print(rootNode.valuePredict)
        
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
        # print(len(rootNode.children))
        for child in rootNode.children:
            # print(child.state.history[-1], child.p, child.n, child.v, child.v / child.n)
            if child.n > bestChild.n:
                bestChild = child
        return bestChild.state.history[-1]


    def rawMCTSBestMove(self, state, player, timeIterations, fieldKnowledge):
        rootNode = MCTSNode(state, player, 1)

        for i in range(timeIterations):
            node = rootNode

            while len(node.unexploredMoves) == 0 and node.state.isTerminal() == False:
                if len(node.children) > 0:
                    node = node.select(player)
                else:
                    break
            
            if len(node.unexploredMoves) > 0 and node.state.isTerminal() == False:
                node = node.expand()

            if node != None and node.state.isTerminal() == False:
                node.randomPlayout(rootNode.player, fieldKnowledge)

            if node != None and node.state.isTerminal() == True:
                winner = node.state.getWinner()
                if winner == rootNode.player:
                    node.backpropagate(1)
                elif winner == -rootNode.player:
                    node.backpropagate(-1)
                else:
                    node.backpropagate(0)
        
        # print(len(rootNode.children))
        bestChild = rootNode.children[0]
        for child in rootNode.children:
            # print(child.state.history[-1], child.n, child.v, child.v / child.n)
            if child.n > bestChild.n:
                bestChild = child
        return bestChild.state.history[-1]

if __name__ == '__main__':
    c_state = GameState()

    def humanMove(player):
        read = input()
        move = tuple(map(int, read.strip().split(' ')))
        if move[0] == -1:
            return move
        while move not in c_state.getValidMoves(player):
            print("Invalid move! retry")
            read = input()
            move = tuple(map(int, read.strip().split(' ')))
        return move

    MCTS = CNNMCTS()

    while 1:
        tmp = int(input("set start board?"))
        if tmp == 1:
            print("set start board(b1w-1)")
            for i in range(8):
                read = input()
                row = tuple(map(int, read.strip().split(' ')))
                for j in range(8):
                    c_state.board[i,j] = row[j]

        blackLimit = 0
        blackKnowledge = 0
        whiteLimit = 0
        whiteKnowledge = 0
        blackPlayer = int(input("Black is human(0) or AI(1)"))
        if blackPlayer == 1:
            blackLimit = int(input())
            blackKnowledge = int(input())
        whitePlayer = int(input("White is human(0) or AI(1)"))
        if whitePlayer == 1:
            whiteLimit = int(input())
            whiteKnowledge = int(input())
        
        c_state.print()
        isSurrender = False
        turn = 1
        while not c_state.isTerminal():
            if len(c_state.getValidMoves(1)) > 0:
                print("Black turns!")
                if blackPlayer == 0:
                    move = humanMove(1)
                    if move[0] == -1:
                        print("Black surrender")
                        print("White win!")
                        isSurrender = True
                        break
                    c_state.makeMove(move, 1)
                else:
                    print("AI thinking...")
                    startTime = time()
                    c_state.makeMove(MCTS.getBestMove(c_state, 1, blackLimit, blackKnowledge, turn), 1)
                    turn += 1
                    print(time() - startTime)
            else:
                print("oops! Black have no available moves")
            
            c_state.print()
            if c_state.isTerminal():
                break
            
            if len(c_state.getValidMoves(-1)) > 0:
                print("White turns!")
                if whitePlayer == 0:
                    move = humanMove(-1)
                    if move[0] == -1:
                        print("White surrender")
                        print("Black win!")
                        isSurrender = True
                        break
                    c_state.makeMove(move, -1)
                else:
                    print("AI thinking...")
                    startTime = time()
                    c_state.makeMove(MCTS.getBestMove(c_state, -1, whiteLimit, whiteKnowledge, turn), -1)
                    turn += 1
                    print(time() - startTime)
            else:
                print("oops! White have no available moves")
            c_state.print()
        if isSurrender == False:
            print("Black score:", c_state.getScore(1))
            print("White score:", c_state.getScore(-1))
            winner = c_state.getWinner()
            if winner == 1:
                print("Black win!")
            elif winner == -1:
                print("White win!")
            else:
                print("Draw!")
        tmp = int(input("again?"))
        if tmp != 1:
            break
        c_state.__init__()