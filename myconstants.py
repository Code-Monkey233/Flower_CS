import torch
DEVICE = torch.device("cpu")
N_FEATURES = 9
INPUT_DIM = N_FEATURES #check，此处特征值的设定不一定合理
OUTPUT_DIM = 1
HIDDEN_DIM = 64
LAYER_DIM = 3
BATCH_SIZE = 64
DROPOUT = 0.2
EPOCH = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
# FL Settings  训练7轮，客户端数目为5
ROUND = 5
NUM_CLIENTS = 2
#from myconstants import *  #本篇中的所有常量引用来源
# Models chosen from rnn, lstm #定义了LSTM模型的基本结构
DATASETS = ["102.csv", "1162.csv"]
MODEL = "lstm"
MODEL_PARAMS = {"input_dim": INPUT_DIM,
                "hidden_dim": HIDDEN_DIM,
                "layer_dim": LAYER_DIM,
                "output_dim": OUTPUT_DIM,
                "dropout_prob": DROPOUT}