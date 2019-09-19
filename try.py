from torch import nn
import torch

from tcn import TemporalConvNet
from continuous_A3C import TCN



sys = TCN(input_size = 3, output_size = 1, num_channels = [25, 25, 30], kernel_size = 8, dropout = 0)
# loss_function = nn.MSELoss()
# optimizer = optim.SGD(sys.parameters(), lr=0.01)

lstm_input = torch.randn(1, 3, 1)
mu, sigma, values  = sys(lstm_input)
# print(lstm_input)
print(mu, sigma, values)