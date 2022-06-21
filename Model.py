import torch

class Net(torch.nn.Module):# 继承 torch 的 Module
    def __init__(self, n_input, hidden_size, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        # self.layer1 = torch.nn.Linear(n_input, hidden_size* 2)  #
        # self.layer2 = torch.nn.Linear(hidden_size* 2, hidden_size * 4)  #
        # self.layer3 = torch.nn.Linear(hidden_size * 4, hidden_size* 2)
        # self.layer4 = torch.nn.Linear(hidden_size* 2, n_output)
        # self.sigmod = torch.nn.Sigmoid()

        self.layer1 = torch.nn.Linear(n_input, n_input * 2)  #
        self.layer2 = torch.nn.Linear(n_input * 2, n_input * 4)  #
        self.layer3 = torch.nn.Linear(n_input * 4, n_input * 2)
        self.layer4 = torch.nn.Linear(n_input * 2, n_output)
        self.sigmod = torch.nn.Sigmoid()

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        x = self.layer1(x)
        x = torch.relu(x)      #
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        x = torch.relu(x)
        x = self.layer4(x)
        # x = self.sigmod(x)
        return x
