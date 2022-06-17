import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
from torchsummary import summary


# import pretty_errors
class PointNetBase(nn.Module):
    def __init__(self):
        super(PointNetBase, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 40)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.max_pool = nn.MaxPool1d(10000, 10000)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)


class trans_matrix(PointNetBase):
    def __init__(self, num_features):
        super(trans_matrix, self).__init__()
        self.num_features = num_features
        self.fc_t = nn.Linear(256, num_features * num_features)
        self.t_conv1 = nn.Conv1d(num_features, 64, 1)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.relu(self.bn1(self.t_conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        # x = torch.max(x, 2, keepdim=True)[0]  # 横向每10000个元素去最大一个, 这样每行保留一个元素, 结果就是1024个特征
        x = self.max_pool(x)
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc_t(x)  # 这一层不需要激活

        identity_matrix = Variable(torch.from_numpy(np.eye(self.num_features, dtype=np.float32).flatten())).repeat(
            batch_size, 1)

        if x.is_cuda:
            identity_matrix = identity_matrix.cuda()

        x = x + identity_matrix
        x = x.view(-1, self.num_features, self.num_features)
        return x


class PointNet(PointNetBase):
    def __init__(self):
        super(PointNet, self).__init__()
        self.trans_matrix_3 = trans_matrix(3)
        self.trans_matrix_64 = trans_matrix(64)

    def forward(self, x):
        # TODO: use functions in __init__ to build network
        # trans_matrix_3 = self.trans_matrix_3(x)
        # x = x.permute(0, 2, 1)
        # x = torch.bmm(x, trans_matrix_3)
        # x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))  # out=64
        # trans_matrix_64 = self.trans_matrix_64(x)
        # x = x.permute(0, 2, 1)
        # x = torch.bmm(x, trans_matrix_64)  # out=64
        # x = x.permute(0, 2, 1)
        x = self.relu(self.bn2(self.conv2(x)))  # out = 128
        x = self.relu(self.bn3(self.conv3(x)))  # out = 1024
        # x = torch.max(x, 2, keepdim=True)[0]
        x = self.max_pool(x)
        x = x.view(-1, 1024)

        # 全连接层
        x = self.relu(self.bn4(self.fc1(x)))  # out=512
        x = self.relu(self.bn5(self.dropout(self.fc2(x))))  # out=256
        x = self.fc3(x)  # out=40

        return x


if __name__ == "__main__":
    net = PointNet()
    sim_data = Variable(torch.rand(3, 3, 10000))
    print(sim_data.size())
    out = net(sim_data)
    print('gfn', out.size())
    # net.cuda()
    # summary(net, (3, 10000))
