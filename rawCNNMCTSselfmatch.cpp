#include <algorithm>
#include <iostream>
#include <fstream>
#include <cctype>
#include <vector>
#include <random>
#include <cmath>
#include <ctime>
// #include <windows.h>

using namespace std;

// fstream fs("output.txt",ios::out);

const int BOARD_SIZE = 8;
const double alpha = 0.256;
const double epsilon = 0.25;
const int annealTurn = 10;

enum class Piece { EMPTY, BLACK, WHITE };

Piece togglePlayer(Piece player) {
    return (player == Piece::BLACK) ? Piece::WHITE : Piece::BLACK;
}

struct Position {
    int row;
    int col;
    double p;

    Position(int r = -1, int c = -1) : row(r), col(c), p(0.0) {}

    bool operator==(const Position& other) const {
        return row == other.row && col == other.col;
    }

    friend ostream& operator<<(ostream& os, const Position& position) {
        os << position.row << ' ' << position.col;
        return os;
    }

    friend bool operator<(const Position &a, const Position &b) {
        return a.p < b.p;
    }

};

int calc(const int& x, const int& y) {
    return x * BOARD_SIZE + y;
}

int calc(Position pos) {
    return pos.row * BOARD_SIZE + pos.col;
}

class Board {
public:
    Board() : last(Position(0, 0)) {
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if ((i == 3 && j == 3) || (i == 4 && j == 4)) {
                    m_board[i][j] = Piece::WHITE;
                }
                else if ((i == 3 && j == 4) || (i == 4 && j == 3)) {
                    m_board[i][j] = Piece::BLACK;
                }
                else {
                    m_board[i][j] = Piece::EMPTY;
                }
            }
        }
    }

    void init() {
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if ((i == 3 && j == 3) || (i == 4 && j == 4)) {
                    m_board[i][j] = Piece::WHITE;
                }
                else if ((i == 3 && j == 4) || (i == 4 && j == 3)) {
                    m_board[i][j] = Piece::BLACK;
                }
                else {
                    m_board[i][j] = Piece::EMPTY;
                }
            }
        }
    }

    vector<Position> getValidMoves(Piece player) const {
        vector<Position> moves;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (m_board[i][j] == Piece::EMPTY && isValidMove(i, j, player)) {
                    moves.push_back(Position(i, j));
                }
            }
        }
        return moves;
    }

    bool isValidMove(int row, int col, Piece player) const {

        if (row < 0 || row >=  BOARD_SIZE || col < 0 || col >= BOARD_SIZE)
            return false;

        if (m_board[row][col] != Piece::EMPTY) {
            return false;
        }

        Piece otherPlayer = togglePlayer(player);

        for (int dRow = -1; dRow <= 1; dRow++) {
            for (int dCol = -1; dCol <= 1; dCol++) {
                if (dRow == 0 && dCol == 0) {
                    continue;
                }

                int r = row + dRow;
                int c = col + dCol;
                int numFlips = 0;

                while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE && m_board[r][c] == otherPlayer) {
                    numFlips++;
                    r += dRow;
                    c += dCol;
                }

                if (numFlips > 0 && r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE && m_board[r][c] == player) {
                    return true;
                }
            }
        }

        return false;
    }

    void makeMove(Position move, Piece player) {
        makeMove(move.row, move.col, player);
    }

    void makeMove(int row, int col, Piece player) {
        last = Position(row, col);
        m_board[row][col] = player;

        Piece otherPlayer = togglePlayer(player);

        for (int dRow = -1; dRow <= 1; dRow++) {
            for (int dCol = -1; dCol <= 1; dCol++) {
                if (dRow == 0 && dCol == 0) {
                    continue;
                }

                int r = row + dRow;
                int c = col + dCol;
                int numFlips = 0;

                while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE && m_board[r][c] == otherPlayer) {
                    numFlips++;
                    r += dRow;
                    c += dCol;
                }

                if (numFlips > 0 && r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE && m_board[r][c] == player) {
                    r = row + dRow;
                    c = col + dCol;
                    while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE && m_board[r][c] == otherPlayer) {
                        m_board[r][c] = player;
                        r += dRow;
                        c += dCol;
                    }
                }
            }
        }
    }

    int getScore(Piece player) const {
        int score = 0;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (m_board[i][j] == player) {
                    score++;
                }
            }
        }
        return score;
    }

    bool isGameOver() const {
        return getValidMoves(Piece::BLACK).empty() && getValidMoves(Piece::WHITE).empty();
    }

    Piece getWinner() const {
        if (!isGameOver()) {
            return Piece::EMPTY;
        }

        int blackScore = getScore(Piece::BLACK);
        int whiteScore = getScore(Piece::WHITE);

        if (blackScore == whiteScore) {
            return Piece::EMPTY;
        }
        else if (blackScore > whiteScore) {
            return Piece::BLACK;
        }
        else {
            return Piece::WHITE;
        }
    }

    Position getLastMove() {
        return last;
    }

    friend ostream& operator<<(ostream& os, const Board& board) {
        os << "  ";
        for (int j = 0; j < BOARD_SIZE; j++) {
            os << j << " ";
        }
        os << endl;

        for (int i = 0; i < BOARD_SIZE; i++) {
            os << i << " ";
            for (int j = 0; j < BOARD_SIZE; j++) {
                switch (board.m_board[i][j]) {
                    case Piece::EMPTY:
                        os << ". ";
                        break;
                    case Piece::BLACK:
                        os << "# ";
                        break;
                    case Piece::WHITE:
                        os << "O ";
                        break;
                }
            }
            os << endl;
        }

        return os;
    }

// private:
    Piece m_board[BOARD_SIZE][BOARD_SIZE];
    Position last;
};

const double PUCT_CONSTANT = 1.0;

vector<double> dirichlet(const int& size) {
    random_device rd;
    mt19937 gen(rd() * time(NULL));
    vector<double> sample(size);
    double sum = 0;
    for (int i = 0; i < size; i++) {
        gamma_distribution<double> d(alpha, 1);
        sample[i] = d(gen);
        sum += sample[i];
    }
    for (int i = 0; i < sample.size(); i++) {
        sample[i] /= sum;
    }
    return sample;
}

class MCTSNode {
public:
    MCTSNode(MCTSNode* parent, Board board, Piece player, int isRoot = 0) 
        : m_parent(parent), m_board(board), m_player(player), m_n(0), m_v(0.0), m_p(0.0) {

        m_unexploredMoves = board.getValidMoves(m_player);

        cout<<0<<' ';
        for (int i = 0; i < BOARD_SIZE; ++i) {
            for (int j = 0; j < BOARD_SIZE; ++j) {
                if (m_board.m_board[i][j] == Piece::BLACK) {
                    cout<<1<<' ';
                }
                else {
                    cout<<0<<' ';
                }
            }
        }
        for (int i = 0; i < BOARD_SIZE; ++i) {
            for (int j = 0; j < BOARD_SIZE; ++j) {
                if (m_board.m_board[i][j] == Piece::WHITE) {
                    cout<<1<<' ';
                }
                else {
                    cout<<0<<' ';
                }
            }
        }
        if (player == Piece::BLACK) {
            for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i)
                cout<<1<<' ';
        }
        else {
            for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i)
                cout<<0<<' ';
        }
        cout<<endl;
        // char str[10000];
        // cin.getline(str,10000);
        // for (int i = 0; i < strlen(str); ++i)
        //     fs << str[i];
        // fs<<endl;
        // while(cin.peek()!=-1) {
        //     fs << char(cin.get());
        // }

        while(cin.peek()!=-1&&!isdigit(cin.peek()))
            cin.get();
        for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
            m_policyPredict[i] = 0.0;
            cin>>m_policyPredict[i];
            // fs<<m_policyPredict[i]<<' ';
        }
        // fs<<endl;
        while(cin.peek()!=-1&&(cin.peek()==']'||cin.peek()==' '))
            cin.get();
        cin>>m_valuePredict;
        // fs<<m_valuePredict<<endl<<endl;

        double sum = 0;
        for (int i = 0; i < m_unexploredMoves.size(); ++i) {
            sum += m_policyPredict[calc(m_unexploredMoves[i])];
        }
        for (int i = 0; i < m_unexploredMoves.size(); ++i) {
            m_policyPredict[calc(m_unexploredMoves[i])] /= sum;
        }

        if (isRoot == 1) {
            vector<double>distribution = dirichlet(m_unexploredMoves.size());
            for (int i = 0; i < m_unexploredMoves.size(); ++i) {
                // fs<<m_policyPredict[i]<<' ';
                m_policyPredict[calc(m_unexploredMoves[i])] = (1 - epsilon) * m_policyPredict[calc(m_unexploredMoves[i])] + epsilon * distribution[i];
                // fs<<m_policyPredict[i]<<endl;
            }
            // fs<<endl;
        }
        for (int i = 0; i < m_unexploredMoves.size(); ++i) {
            m_unexploredMoves[i].p = m_policyPredict[calc(m_unexploredMoves[i])];
        }
        sort(m_unexploredMoves.begin(), m_unexploredMoves.end());
    }

    ~MCTSNode() {
        for (MCTSNode* child : m_children) {
            delete child;
        }
    }

    MCTSNode* select() {
        MCTSNode* selectedChild = nullptr;
        double bestUCT = -INFINITY;

        for (MCTSNode* child : m_children) {
            double UCT = child->getPUCTValue();

            if (UCT > bestUCT) {
                selectedChild = child;
                bestUCT = UCT;
            }
        }

        return selectedChild;
    }

    MCTSNode* expand() {
        if (m_unexploredMoves.empty()) {
            return nullptr;
        }

        Position move = m_unexploredMoves.back();
        m_unexploredMoves.pop_back();

        Board newBoard = m_board;
        newBoard.makeMove(move, m_player);

        MCTSNode* child = nullptr;
        if(!newBoard.getValidMoves(togglePlayer(m_player)).empty())
            child = new MCTSNode(this, newBoard, togglePlayer(m_player));
        else
            child = new MCTSNode(this, newBoard, m_player);
        m_children.push_back(child);
        child->m_p = m_policyPredict[calc(move)];
        if (child->m_p < 0.1)
            child->m_p = 0.1;

        return child;
    }

    void backpropagate(double score) {
        ++m_n;
        m_v += score;

        if (m_parent != nullptr) {
            m_parent->backpropagate(score);
        }
    }

    double getPUCTValue() const {
        double Q = static_cast<double>(m_v) / m_n;
        double U = PUCT_CONSTANT * m_p * sqrt(m_parent->m_n - m_n + 1) / (m_n + 1);
        if (m_parent->m_player == Piece::WHITE)
            Q = -Q;
        return Q + U;
    }

// private:
    MCTSNode* m_parent;
    vector<MCTSNode*> m_children;
    Board m_board;
    Piece m_player;
    vector<Position> m_unexploredMoves;
    int m_n;
    double m_v;
    double m_p;
    double m_valuePredict;
    double m_policyPredict[64];
};

class MCTSSolver {
public:

    Position getBestMove(const Board& board, const Piece& player,const int& searchDepth, const int& iterationTimes, const int& fieldKnowledge, const int& displayInfo, const int& turn) {
        // srand(time(NULL));

        MCTSNode rootNode(nullptr, board, player, 1);

        for (int i = 0; i < iterationTimes; ++i) {
            MCTSNode* node = &rootNode;
            int dep = 0;

            while (dep < searchDepth && !node->m_board.isGameOver())
            {
                // Selection phase
                while (node->m_unexploredMoves.empty() && !node->m_board.isGameOver()) {
                    if (!node->m_children.empty()) {
                        node = node->select();
                        ++dep;
                    }
                    else {
                        break;
                    }
                }

                // Expansion phase
                if (node->m_unexploredMoves.size() > 0 && !node->m_board.isGameOver()) {
                    node = node->expand();
                    ++dep;
                }
            }

            // Backpropagation phase
            if (node != nullptr) {
                if (node->m_board.isGameOver() == true) {
                    Piece winner = node->m_board.getWinner();
                    if (winner == Piece::BLACK) {
                        node->backpropagate(1);
                    }
                    else if (winner == Piece::WHITE) {
                        node->backpropagate(-1);
                    }
                    else {
                        node->backpropagate(0);
                    }
                }
                else
                    node->backpropagate(node->m_valuePredict);
            }
        }

        double policyDistributuion[BOARD_SIZE*BOARD_SIZE];
        fill(policyDistributuion, policyDistributuion+BOARD_SIZE*BOARD_SIZE, 0.0);

        cout << 1 << ' ';
        if (displayInfo == 1) 
            cout << rootNode.m_children.size()<<endl;
        MCTSNode* bestChild = rootNode.m_children[0];
        for (MCTSNode* child : rootNode.m_children) {
            if(displayInfo == 1)
                cout << child->m_board.getLastMove() <<' '<< child->m_n<<' '<<child->m_v<<' '<<child->m_v*1.0/child->m_n<<endl;

            if (turn <= annealTurn)
                // policyDistributuion[calc(child->m_board.getLastMove())] = child->m_n;
                policyDistributuion[calc(child->m_board.getLastMove())] = child->m_n * 1.0 / rootNode.m_n;

            if (child->m_n > bestChild->m_n) {
                bestChild = child;
            }
        }
        if (turn > annealTurn)
            policyDistributuion[calc(bestChild->m_board.getLastMove())] = 1.0;
            // policyDistributuion[calc(bestChild->m_board.getLastMove())] = iterationTimes;

        for (int i = 0; i < BOARD_SIZE*BOARD_SIZE; ++i) {
            cout<<policyDistributuion[i]<<' ';
        }

        return bestChild->m_board.getLastMove();
    }
};

MCTSSolver MCTS;
Board c_board;
int tmp, iterationTimes, searchDepth, knowledge = 1, displayInfo = 0, isTerminate;
Piece currentPlayer;
Position bestMove;
int main()
{
    cin.sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);

    cin >> searchDepth >> iterationTimes;
    // cin >> isTerminate;

    // while (isTerminate == 0) {
    while (1) {
        c_board.init();
        currentPlayer = Piece::BLACK;
        for (int turn = 1; c_board.isGameOver() == false; ++turn) {
            if (c_board.getValidMoves(currentPlayer).size() == 0) {
                currentPlayer = togglePlayer(currentPlayer);
                --turn;
                continue;
            }
            bestMove = MCTS.getBestMove(c_board, currentPlayer, searchDepth, iterationTimes, knowledge, displayInfo, turn);
            // if (5 <= turn && turn <= 54) {
            for (int i = 0; i < BOARD_SIZE; ++i) {
                for (int j = 0; j < BOARD_SIZE; ++j) {
                    if (c_board.m_board[i][j] == Piece::BLACK) {
                        cout<<1<<' ';
                    }
                    else {
                        cout<<0<<' ';
                    }
                }
            }
            for (int i = 0; i < BOARD_SIZE; ++i) {
                for (int j = 0; j < BOARD_SIZE; ++j) {
                    if (c_board.m_board[i][j] == Piece::WHITE) {
                        cout<<1<<' ';
                    }
                    else {
                        cout<<0<<' ';
                    }
                }
            }
            if (currentPlayer == Piece::BLACK) {
                for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i)
                    cout<<1<<' ';
            }
            else {
                for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i)
                    cout<<0<<' ';
            }
            // }
            cout<<endl;
            cout<<2<<' '<<turn<<endl;

            c_board.makeMove(bestMove, currentPlayer);
            currentPlayer = togglePlayer(currentPlayer);
        }
        cout<<-1<<' ';
        Piece winner = c_board.getWinner();
        if (winner == Piece::BLACK)
            cout<<1;
        else if (winner == Piece::WHITE)
            cout<<-1;
        else if (winner == Piece::EMPTY)
            cout<<0;
        cout<<endl;
        // cin >> isTerminate;
    }
    // fstream fs;
    // fs.open("output.txt",ios::out);
    // c_board = Board();
    // cout<<0<<' ';
    // for (int i = 0; i < BOARD_SIZE; ++i) {
    //     for (int j = 0; j < BOARD_SIZE; ++j) {
    //         if (c_board.m_board[i][j] == Piece::BLACK) {
    //             cout<<1<<' ';
    //         }
    //         else {
    //             cout<<0<<' ';
    //         }
    //     }
    // }
    // for (int i = 0; i < BOARD_SIZE; ++i) {
    //     for (int j = 0; j < BOARD_SIZE; ++j) {
    //         if (c_board.m_board[i][j] == Piece::BLACK) {
    //             cout<<1<<' ';
    //         }
    //         else {
    //             cout<<0<<' ';
    //         }
    //     }
    // }
    // if (currentPlayer == Piece::BLACK) {
    //     for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i)
    //         cout<<1<<' ';
    // }
    // else {
    //     for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i)
    //         cout<<0<<' ';
    // }
    // cout<<endl;
    // double x = 0;
    // while(cin.peek()!=-1&&!isdigit(cin.peek()))
    //     cin.get();
    // for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
    //     cin>>x;
    //     fs << x << ' ';
    // }
    // fs<<endl;
    // while(cin.peek()!=-1&&!isdigit(cin.peek()))
    //     cin.get();
    // cin>>x;
    // fs<<x<<endl;
    return 0;
}