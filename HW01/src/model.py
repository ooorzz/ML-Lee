import torch.nn as nn


class NeuralNet(nn.Module):
    """A simple fully-connected deep neural network."""

    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        # 输入层 93×64 -- RELU -- 输出层 64×1
        # 两个全连接层，用RELU作为激活函数
        # 但这看起来不deep啊，就是一层，用64个RELU来近似函数
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # 用均方差来作为loss函数
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, x):
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        return self.criterion(pred, target)
