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
const double alpha = 0.156;
const double epsilon = 0.25;

enum class Piece { EMPTY, BLACK, WHITE };

Piece togglePlayer(Piece player) {
    return (player == Piece::BLACK) ? Piece::WHITE : Piece::BLACK;
}

struct Position {
    int row;
    int col;

    Position(int r = -1, int c = -1) : row(r), col(c) {}

    bool operator==(const Position& other) const {
        return row == other.row && col == other.col;
    }

    friend ostream& operator<<(ostream& os, const Position& position) {
        os << position.row << ' ' << position.col;
        return os;
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

        bool valid = false;
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
                    valid = true;
                }
            }
        }

        return valid;
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

vector<double> dirichlet() {
    random_device rd;
    mt19937 gen(rd() * time(NULL));
    vector<double> sample(BOARD_SIZE * BOARD_SIZE);
    double sum = 0;
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
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
        : m_parent(parent), m_board(board), m_player(player), m_unexploredMoves(board.getValidMoves(m_player)), m_n(0), m_v(0.0), m_p(0.0) {
            
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

        if (isRoot == 1) {
            vector<double>distribution = dirichlet();
            for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
                // fs<<m_policyPredict[i]<<' ';
                m_policyPredict[i] = (1 - epsilon) * m_policyPredict[i] + epsilon * distribution[i];
                // fs<<m_policyPredict[i]<<endl;
            }
            // fs<<endl;
        }
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

        return child;
    }

    void backpropagate(double score) {
        m_n++;
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

double annealParameter(const int& turn) {
    if(turn <= 10)
        return 1.0;
    return max(exp(1.0 * (10 - turn)), 1e-8);
}


class MCTSSolver {
public:

    Position getBestMove(const Board& board, const Piece& player, const int& searchDepth, const int& timeIterations, const int& fieldKnowledge, const int& displayInfo, const int& turn) {
        // srand(time(NULL));

        MCTSNode rootNode(nullptr, board, player, 1);

        for (int i = 0; i < timeIterations; ) {
            MCTSNode* node = &rootNode;
            int dep = 0;

            while (dep < searchDepth && !node->m_board.isGameOver())
            {
                // Selection phase
                while (node->m_unexploredMoves.empty() && !node->m_board.isGameOver()) {
                    if (!node->m_children.empty()) {
                        node = node->select();
                        ++dep;
                        ++i;
                    }
                    else {
                        break;
                    }
                }

                // Expansion phase
                if (node->m_unexploredMoves.size() > 0 && !node->m_board.isGameOver()) {
                    node = node->expand();
                    ++dep;
                    ++i;
                }
            }

            // Backpropagation phase
            if (node != nullptr) {
                if (node->m_board.isGameOver() == true) {
                    Piece winner = node->m_board.getWinner();
                    if (winner == Piece::BLACK) {
                        node->backpropagate(1);
                    }
                    else if (winner == Piece::WHITE){
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
        // double temperature = annealParameter(turn);
        // fs << temperature << endl;

        // cout << 1 << ' ';
        if (displayInfo == 1) 
            cout << rootNode.m_children.size()<<endl;
        MCTSNode* bestChild = rootNode.m_children[0];
        for (MCTSNode* child : rootNode.m_children) {
            if(displayInfo == 1)
                cout << child->m_board.getLastMove() <<' '<< child->m_n<<' '<<child->m_v<<' '<<child->m_v*1.0/child->m_n<<endl;

            // policyDistributuion[calc(child->m_board.getLastMove())] = pow(child->m_n, 1.0 / temperature) / pow((long double)(timeIterations), (long double)(1.0 / temperature));
            // fs << calc(child->m_board.getLastMove()) <<' '<< policyDistributuion[calc(child->m_board.getLastMove())]<<endl;

            if (child->m_n > bestChild->m_n) {
                bestChild = child;
            }
        }
        // policyDistributuion[calc(bestChild->m_board.getLastMove())] = 1.0;
        // for (int i = 0; i < BOARD_SIZE*BOARD_SIZE; ++i) {
        //     cout<<policyDistributuion[i]<<' ';
        // }

        return bestChild->m_board.getLastMove();
    }
    
    MCTSNode* lastMove;
};

MCTSSolver MCTS;
Board c_board;
int tmp, roundLimit, searchDepth = 10, knowledge = 1, displayInfo = 0, turn;
Piece currentPlayer;
Position bestMove;
int main()
{
    cin.sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            cin>>tmp;
            if (tmp == -1) {
                c_board.m_board[i][j] = Piece::WHITE;
            }
            else if (tmp == 1) {
                c_board.m_board[i][j] = Piece::BLACK;
            }
            else {
                c_board.m_board[i][j] = Piece::EMPTY;
            }
        }
    }
    cin>>tmp>>roundLimit>>turn;
    if (tmp == -1)
        currentPlayer = Piece::WHITE;
    else
        currentPlayer = Piece::BLACK;
    cout<<-1<<' '<<MCTS.getBestMove(c_board, currentPlayer, searchDepth, roundLimit, knowledge, displayInfo, turn)<<endl;
    return 0;
}