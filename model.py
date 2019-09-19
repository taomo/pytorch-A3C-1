from torch import nn
from tcn import TemporalConvNet

'''
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        return self.linear(y1.transpose(1, 2))
'''

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        # self.linear = nn.Linear(num_channels[-1], output_size)
        
        self.critic_linear = nn.Linear(num_channels[-1] * 10, 1)
        self.actor_linear = nn.Linear(num_channels[-1] * 10, output_size)
        self.init_weights()

    def init_weights(self):
        self.critic_linear.weight.data.normal_(0, 0.01)
        self.actor_linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        # print(y1.size())
        # print(y1.transpose(1, 2).size())
        # print(self.critic_linear(y1.view(-1)).size())
        # print(self.critic_linear(y1.view(-1)))
        # return self.linear(y1.transpose(1, 2))
        return self.critic_linear(y1.view(-1)) , self.actor_linear(y1.view(-1))#, self.actor_linear(y1.transpose(1, 2))