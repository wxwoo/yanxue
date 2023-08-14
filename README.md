train_py.py: 训练程序，生成数据之后对神经网络进行训练，保存在rescnn.pth   
cnn-mcts.py: 可以运行的黑白棋AI，支持基于py/cpp的纯MCTS决策、神经网络-MCTS结合决策，可以自我对弈、人机对弈    
MCTS.cpp: 基于cpp的MCTS，可以自我对战    
mcts.py: 基于py的mcts    
rescnn.py: 覆盖rescnn.pth，生成一个空的神经网络，输出神经网络架构    
rawcnn.py: 从rescnn.pth加载神经网络模型，测试神经网络输出    
train.py: 基于cpp训练神经网络，需要rawCNNMCTSselfmatch.cpp辅助    
rawCNNMCTS.py/rawCNNMCTSselfmatch.py/rawMCTS.py: 辅助cnn-mcts.py/train.py，基于windows10环境    