#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <windows.h>

using namespace std;

const int BOARD_SIZE = 8;

enum class Piece { EMPTY, BLACK, WHITE };

Piece togglePlayer(Piece player) {
    return (player == Piece::BLACK) ? Piece::WHITE : Piece::BLACK;
}

struct Position {
    int row;
    int col;

    Position(int r, int c) : row(r), col(c) {}

    bool operator==(const Position& other) const {
        return row == other.row && col == other.col;
    }

    friend ostream& operator<<(ostream& os, const Position& position) {
        os << position.row << ' ' << position.col;
        return os;
    }

};

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

const double UCT_CONSTANT = sqrt(2);

class MCTSNode {
public:
    MCTSNode(MCTSNode* parent, Board board, Piece player) 
        : m_parent(parent), m_board(board), m_player(player), m_unexploredMoves(board.getValidMoves(m_player)), m_n(0), m_v(0) {}

    ~MCTSNode() {
        for (MCTSNode* child : m_children) {
            delete child;
        }
    }

    MCTSNode* select(Piece player) {
        MCTSNode* selectedChild = nullptr;
        double bestUCT = -INFINITY;

        for (MCTSNode* child : m_children) {
            double UCT = child->getUCTValue(player);

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

        return child;
    }

    void backpropagate(double score) {
        m_n++;
        m_v += score;

        if (m_parent != nullptr) {
            m_parent->backpropagate(score);
        }

        // if (m_parent != nullptr) {
        //     if (m_parent->m_player == winner)
        //         m_parent->backpropagate(score, winner);
        //     else
        //         m_parent->backpropagate(-score, winner);
        // }

        // if (m_parent != nullptr) {
        //     if (m_parent->m_player == winner)
        //         m_parent->backpropagate(1, winner);
        //     else
        //         m_parent->backpropagate(0, winner);
        // }
    }

    double getUCTValue(Piece player) const {
        if (m_n == 0) {
            return INFINITY;
        }

        double exploitation = static_cast<double>(m_v) / m_n;
        double exploration = UCT_CONSTANT * sqrt(log(m_parent->m_n) / m_n);
        if(player != m_parent->m_player)
            exploitation = -exploitation;

        return exploitation + exploration;
    }

    void simulateRandomPlayout(Piece rootPlayer, int fieldKnnowledge) {
        Piece currentPlayer = m_player;
        Board board = m_board;

        while (!board.isGameOver()) {
            vector<Position> moves = board.getValidMoves(currentPlayer);

            if (moves.empty()) {
                currentPlayer = togglePlayer(currentPlayer);
                continue;
            }

            int randomIndex = rand() % moves.size();
            Position move = moves[randomIndex];
            if (fieldKnnowledge == 1) {
                if ((move.row <= 1 || move.row >= 6) && (move.col <= 1 || move.col >= 6)) {
                    randomIndex = rand() % moves.size();
                    move = moves[randomIndex];
                }
                for (int i = 0; i < BOARD_SIZE; i+=7) {
                    for (int j = 0; j < BOARD_SIZE; j+=7) {
                        if (board.isValidMove(i, j, currentPlayer)) {
                            move = (Position){i, j};
                        }
                    }
                }
            }
            
            board.makeMove(move, currentPlayer);
            currentPlayer = togglePlayer(currentPlayer);
        }

        Piece winner = board.getWinner();

        if (winner == Piece::EMPTY) {
            backpropagate(0);
        }
        else if (winner == rootPlayer) {
            backpropagate(1);
        }
        else {
            backpropagate(-1);
        }

        // if (winner == rootPlayer) {
        //     backpropagate(1);
        // }
        // else {
        //     backpropagate(0);
        // }
    }

// private:
    MCTSNode* m_parent;
    vector<MCTSNode*> m_children;
    Board m_board;
    Piece m_player;
    vector<Position> m_unexploredMoves;
    int m_n;
    int m_v;
};

class MCTSSolver {
public:

    Position getBestMove(const Board& board, const Piece& player, const int& timeIterations, const int& fieldKnowledge, const int& displayInfo) {
        srand(time(NULL));

        MCTSNode rootNode(nullptr, board, player);

        for (int i = 0; i < timeIterations; i++) {
            MCTSNode* node = &rootNode;

            // Selection phase
            while (node->m_unexploredMoves.empty() && !node->m_board.isGameOver()) {
                if (!node->m_children.empty()) {
                    node = node->select(player);
                }
                else {
                    break;
                }
            }

            // Expansion phase
            if (node->m_unexploredMoves.size() > 0 && !node->m_board.isGameOver()) {
                node = node->expand();
            }
    
            // Simulation phase
            if (node != nullptr && !node->m_board.isGameOver()) {
                node->simulateRandomPlayout(rootNode.m_player, fieldKnowledge);
            }

            // Backpropagation phase
            if (node != nullptr && node->m_board.isGameOver()) {
                Piece winner = node->m_board.getWinner();
                if (winner == rootNode.m_player) {
                    node->backpropagate(1);
                }
                else if (winner == Piece::EMPTY) {
                    node->backpropagate(0);
                }
                else {
                    node->backpropagate(-1);
                }
            }
        }
        if (displayInfo == 1)
            cout << rootNode.m_children.size()<<endl;
        MCTSNode* bestChild = rootNode.m_children[0];
        for (MCTSNode* child : rootNode.m_children) {
            if(displayInfo == 1)
                cout << child->m_board.getLastMove() <<' '<< child->m_n<<' '<<child->m_v<<' '<<child->m_v*1.0/child->m_n<<endl;
            if (child->m_n > bestChild->m_n) {
                bestChild = child;
            }
        }

        return bestChild->m_board.getLastMove();
    }

};

MCTSSolver MCTS;
Board c_board;
Position humanMove(Piece player)
{
    int x,y;
    char ch = ' ';
    while(1) {
        while(cin.peek()!=-1&&!isdigit(cin.peek()))
            ch = cin.get();
        cin>>x;
        if (ch == '-')
            x = -x;
        if(x==-1) {
            return (Position){-1,-1};
        }
        while(cin.peek()!=-1&&!isdigit(cin.peek()))
            cin.get();
        cin>>y;
        if(c_board.isValidMove(x, y, player)) 
            break;
        cout<<"Invalid move! retry"<<endl;
    }
    return (Position){x, y};
}

int blackPlayer,whitePlayer,blackLimit,whiteLimit,blackKnowledge,whiteKnowledge,debugInfo,tmp;
int main()
{
    cin.sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    start:;
    cout<<"set start board?"<<endl;
    cin>>tmp;
    if (tmp == 1) {
        cout<<"set start board(b1w-1)"<<endl;
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
    }
    else if (tmp == -1) {
        cout<<"dataset generate mode"<<endl;
        cout<<"round limit:"<<endl;
        cin>>blackLimit;
        cout<<"simulate round"<<endl;
        int simulateRound;
        cin>>simulateRound;
        ofstream out("dataset.out");
        srand(time(NULL));
        while(simulateRound--) {
            Piece currentPlayer = Piece::BLACK;
            int getDataFrom = rand() % 31 + 20;
            cout<<simulateRound + 1<<" round(s) left..."<<endl;
            for (int _ = 1; !c_board.isGameOver(); ++_) {
                vector<Position> moves = c_board.getValidMoves(currentPlayer);

                if (moves.empty()) {
                    currentPlayer = togglePlayer(currentPlayer);
                    --_;
                    continue;
                }
                
                Position move = MCTS.getBestMove(c_board, currentPlayer, blackLimit, 1, 0);
                
                if (_ == getDataFrom) {
                    for (int i = 0; i < BOARD_SIZE; ++i) {
                        for (int j = 0; j < BOARD_SIZE; ++j) {
                            if (c_board.m_board[i][j] == Piece::BLACK) {
                                out<<1<<' ';
                            }
                            else {
                                out<<0<<' ';
                            }
                        }
                    }
                    out<<endl;
                    for (int i = 0; i < BOARD_SIZE; ++i) {
                        for (int j = 0; j < BOARD_SIZE; ++j) {
                            if (c_board.m_board[i][j] == Piece::WHITE) {
                                out<<1<<' ';
                            }
                            else {
                                out<<0<<' ';
                            }
                        }
                    }
                    out<<endl;
                    if (currentPlayer == Piece::BLACK) {
                        out<<1<<endl;
                    }
                    else {
                        out<<0<<endl;
                    }
                    for (int i = 0; i < BOARD_SIZE; ++i) {
                        for (int j = 0; j < BOARD_SIZE; ++j) {
                            if (i == move.row && j == move.col) {
                                out<<1<<' ';
                            }
                            else {
                                out<<0<<' ';
                            }
                        }
                    }
                    out<<endl;
                }
                c_board.makeMove(move, currentPlayer);
                currentPlayer = togglePlayer(currentPlayer);
            }
            Piece winner = c_board.getWinner();
            if (winner == Piece::BLACK) {
                out<<1<<endl;
            }
            else if (winner == Piece::WHITE) {
                out<<-1<<endl;
            }
            else {
                out<<0<<endl;
            }
            c_board = (Board){};
        }
        return 0;
    }
    cout<<"Black is human(0) or AI(1,simulate round limit,if use field knowledge):"<<endl;
    cin>>blackPlayer;
    if (blackPlayer == 1) {
        cin>>blackLimit>>blackKnowledge;
    }
    cout<<"White is human(0) or AI(1,simulate round limit,if use field knowledge):"<<endl;
    cin>>whitePlayer;
    if (whitePlayer == 1) {
        cin>>whiteLimit>>whiteKnowledge;
    }
    cout<<"show debug info"<<endl;
    cin>>debugInfo;
    bool isSurrender = false;
    cout<<c_board;
    while(!c_board.isGameOver()) {
        if(c_board.getValidMoves(Piece::BLACK).size() != 0) {
            cout<<"Black turns!"<<endl;
            if (blackPlayer == 1) {
                cout<<"AI thinking..."<<endl;
                c_board.makeMove(MCTS.getBestMove(c_board, Piece::BLACK, blackLimit, blackKnowledge, debugInfo), Piece::BLACK);
            }
            else {
                Position move = humanMove(Piece::BLACK);
                if (move == (Position){-1,-1}) {
                    cout<<"Black surrender"<<endl<<"White Win!"<<endl;
                    isSurrender = true;
                    break;
                }
                c_board.makeMove(move, Piece::BLACK);
            }
            // system("cls");
            cout<<c_board;
            cout<<"Last Move: B "<<c_board.last<<endl;
            if(c_board.isGameOver())
                break;
        }
        else {
            cout<<"oops! Black have no available moves"<<endl;
        }
        
        if(c_board.getValidMoves(Piece::WHITE).size() != 0) {
            cout<<"White turns!"<<endl;
            if (whitePlayer ==1 ) {
                cout<<"AI thinking..."<<endl;
                c_board.makeMove(MCTS.getBestMove(c_board, Piece::WHITE, whiteLimit, whiteKnowledge, debugInfo), Piece::WHITE);
            }
            else {
                Position move = humanMove(Piece::WHITE);
                if (move == (Position){-1,-1}) {
                    cout<<"White surrender"<<endl<<"Black Win!"<<endl;
                    isSurrender = true;
                    break;
                }
                c_board.makeMove(move, Piece::WHITE);
            }
            // system("cls");
            cout<<c_board;
            cout<<"Last Move: W "<<c_board.last<<endl;
            if(c_board.isGameOver())
                break;
        }
        else {
            cout<<"oops! White have no available moves"<<endl;
        }
    }

    if (!isSurrender) {
        int blackScore = c_board.getScore(Piece::BLACK);
        int whiteScore = c_board.getScore(Piece::WHITE);
        cout<<"black score: "<<blackScore<<endl;
        cout<<"white score: "<<whiteScore<<endl;

        if (blackScore == whiteScore) {
            cout<<"Draw!"<<endl;
        }
        else if (blackScore > whiteScore) {
            cout<<"Black win!"<<endl;
        }
        else {
            cout<<"White win!"<<endl;
        }
    }
    cout<<"again?"<<endl;
    cin>>tmp;
    if (tmp == 1) {
        system("cls");
        c_board = (Board){};
        goto start;
    }
    return 0;
}