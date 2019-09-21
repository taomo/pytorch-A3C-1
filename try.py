from torch import nn
import torch

from tcn import TemporalConvNet
from continuous_A3C import TCN

import gym
from utils import v_wrap, set_init, push_and_pull, record


batch = 1
seq_len = 1

# sys = TCN(input_size = 3, output_size = 1, num_channels = [25, 25, 30], kernel_size = 8, dropout = 0)
# # loss_function = nn.MSELoss()
# # optimizer = optim.SGD(sys.parameters(), lr=0.01)

# lstm_input = torch.randn(1, 3, 1)
# mu, sigma, values  = sys(lstm_input)
# # print(lstm_input)
# print(mu, sigma, values)


env_id  = 'VibrationEnv-v0'

env = gym.make(env_id)  #Pendulum  VibrationEnv


input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]

env.reset()

s_, r, done, _ = env.step(1)

# s_ = torch.from_numpy(s_.reshape([batch, input_size,seq_len])).float()
# s_ = s_.reshape([1, input_size,-1]).float()
# s_ = s_.float()
print(s_)
sys = TCN(input_size, output_size, num_channels = [25, 25, 30], kernel_size = 8, dropout = 0)

# lstm_input = torch.randn(1, input_size, 1)
# mu, sigma, values  = sys(lstm_input)
# print(lstm_input)
# print(mu, sigma, values)

v_wrap(s_, batch, input_size, seq_len)

mu, sigma, values  = sys(v_wrap(s_, batch, input_size, seq_len))
print(mu, sigma, values)


# a = sys.choose_action(v_wrap(s_[None, :]))
a = sys.choose_action(v_wrap(s_, batch, input_size, seq_len))
print(a)