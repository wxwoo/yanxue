
import copy
import math
import random

class Node:
    def __init__(self, state):
        self.state = state # 当前节点的游戏状态
        self.parent = None # 父节点
        self.children = [] # 子节点
        self.wins = 0 # 该节点获胜的次数
        self.visits = 0 # 该节点被访问的次数
        self.untried_actions = self.get_legal_actions() # 该节点未扩展的动作集合

    def get_legal_actions(self):
        # 获取当前游戏状态下所有合法的动作
        actions = []
        for x in range(8):
            for y in range(8):
                if self.state[x][y] == 0:
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        nx, ny = x + dx, y + dy
                        if nx < 0 or nx >= 8 or ny < 0 or ny >= 8 or self.state[nx][ny] == 0:
                            continue
                        while 0 <= nx < 8 and 0 <= ny < 8 and self.state[nx][ny] == -self.state[x][y]:
                            nx, ny = nx + dx, ny + dy
                        if 0 <= nx < 8 and 0 <= ny < 8 and self.state[nx][ny] == self.state[x][y]:
                            actions.append((x, y))
                            break
        return actions

    def expand(self, action):
        # 根据给定的动作扩展子节点
        new_state = copy.deepcopy(self.state)
        x, y = action
        new_state[x][y] = 1
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= 8 or ny < 0 or ny >= 8 or new_state[nx][ny] == 0:
                continue
            if new_state[nx][ny] == -1:
                while 0 <= nx < 8 and 0 <= ny < 8 and new_state[nx][ny] == -1:
                    nx, ny = nx + dx, ny + dy
                if 0 <= nx < 8 and 0 <= ny < 8 and new_state[nx][ny] == 1:
                    tx, ty = x + dx, y + dy
                    while tx != nx or ty != ny:
                        new_state[tx][ty] = 1
                        tx, ty = tx + dx, ty + dy
            elif new_state[nx][ny] == 1:
                continue
        child = Node(new_state)
        child.parent = self
        child.untried_actions = child.get_legal_actions()
        self.children.append(child)
        self.untried_actions.remove(action)
        return child

    def is_terminal(self):
        # 判断当前节点是否为终止节点
        for x in range(8):
            for y in range(8):
                if self.state[x][y] == 0:
                    return False
        return True
    def get_result(self):
        # 获取当前节点的结果（获胜次数）
        count = sum(row.count(1) for row in self.state)
        return count

class MCTS:
    def init(self, time_budget):
        self.time_budget = time_budget # 时间预算
        self.root = None # 根节点
    def select(self, node):
        # 选择最优的子节点进行扩展
        while not node.is_terminal():
            if node.untried_actions:
                return node.expand(random.choice(node.untried_actions))
            else:
                node = self.get_best_child(node)
        return node

    def simulate(self, node):
        # 模拟游戏，返回获胜次数
        state = copy.deepcopy(node.state)
        current_player = 1
        while not node.is_terminal():
            legal_actions = node.get_legal_actions()
            if not legal_actions:
                current_player = -current_player
                continue
            action = random.choice(legal_actions)
            x, y = action
            state[x][y] = current_player
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= 8 or ny < 0 or ny >= 8 or state[nx][ny] == 0:
                    continue
                if state[nx][ny] == -current_player:
                    while 0 <= nx < 8 and 0 <= ny < 8 and state[nx][ny] == -current_player:
                        nx, ny = nx + dx, ny + dy
                    if 0 <= nx < 8 and 0 <= ny < 8 and state[nx][ny] == current_player:
                        tx, ty = x + dx, y + dy
                        while tx != nx or ty != ny:
                            state[tx][ty] = current_player
                            tx, ty = tx + dx, ty + dy
                elif state[nx][ny] == current_player:
                    continue
            current_player = -current_player
        result = node.get_result()
        return result

    def backpropagate(self, node, result):
        # 更新节点的获胜次数和访问次数
        node.visits += 1
        node.wins += result
        if node.parent:
            self.backpropagate(node.parent, result)

    def get_best_child(self, node):
        # 获取最优的子节点
        best_child = None
        best_score = float('-inf')
        for child in node.children:
            score = child.wins / child.visits + math.sqrt(2 * math.log(node.visits) / child.visits)
            if score > best_score:
                best_child = child
                best_score = score
        return best_child

    def get_action(self, state):
        # 获取最优的动作
        if not self.root or self.root.state != state:
            self.root = Node(state)
        start_time = time.time()
        while time.time() - start_time < self.time_budget:
            node = self.select(self.root)
            result = self.simulate(node)
            self.backpropagate(node, result)
            best_child = self.get_best_child(self.root)
        return best_child.action
if name == 'main':
# 初始化游戏状态
    state = [[0] * 8 for _ in range(8)]
    state[3][3], state[4][4] = 1, 1
    state[3][4], state[4][3] = -1, -1
    # 初始化MCTS算法
    mcts = MCTS(time_budget=5)

    # 执行游戏
    current_player = 1
    while True:
        legal_actions = get_legal_actions(state, current_player)
        if not legal_actions:
            current_player = -current_player
            continue
        if current_player == 1:
            action = mcts.get_action(state)
            print(f'黑方选择了：({action[0]}, {action[1]})')
        else:
            action = input('白方选择：')
            x, y = map(int, action.split(','))
            action = (x, y)
        if action not in legal_actions:
            print('无效的动作，请重新选择。')
            continue
        x, y = action
        state[x][y] = current_player
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= 8 or ny < 0 or ny >= 8 or state[nx][ny] == 0:
                continue
            if state[nx][ny] == -current_player:
                while 0 <= nx < 8 and 0 <= ny < 8 and state[nx][ny] == -current_player:
                    nx, ny = nx + dx, ny + dy
                if 0 <= nx < 8 and 0 <= ny < 8 and state[nx][ny] == current_player:
                    tx, ty = x + dx, y + dy
                    while tx != nx or ty != ny:
                        state[tx][ty] = current_player
                        tx, ty = tx + dx, ty + dy
            elif state[nx][ny] == current_player:
                continue
        current_player = -current_player
        if is_terminal(state):
            break
    count = sum(row.count(1) for row in state)
    if count > 0:
        print('黑方获胜！')
    elif count < 0:
        print('白方获胜！')
    else:
        print('平局！')


