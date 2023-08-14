
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
        print(f'{cnt} / {DATASIZE}\r', end='')
        
    print('')
