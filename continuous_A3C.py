"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""

import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gym
import math, os
os.environ["OMP_NUM_THREADS"] = "1"

import gym_Vibration
from tcn import TemporalConvNet


UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000
MAX_EP_STEP = 200

env_id  = 'Pendulum-v0'

env = gym.make(env_id)  #Pendulum  VibrationEnv
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]

input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]

num_channels = [1]
kernel_size = 2
dropout = 0


'''
class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a1 = nn.Linear(s_dim, 200)
        self.mu = nn.Linear(200, a_dim)
        self.sigma = nn.Linear(200, a_dim)
        self.c1 = nn.Linear(s_dim, 100)
        self.v = nn.Linear(100, 1)
        set_init([self.a1, self.mu, self.sigma, self.c1, self.v])
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        a1 = F.relu6(self.a1(x))
        mu = 2 * F.tanh(self.mu(a1))
        sigma = F.softplus(self.sigma(a1)) + 0.001      # avoid 0
        c1 = F.relu6(self.c1(x))
        values = self.v(c1)
        return mu, sigma, values

    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
        return m.sample().numpy()

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()
        return total_loss
'''


class TCN(nn.Module):
    # def __init__(self, s_dim, a_dim):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()

        s_dim = input_size
        a_dim = output_size

        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        # self.linear = nn.Linear(num_channels[-1], output_size)        
        # self.critic_linear = nn.Linear(num_channels[-1], 1)
        # self.actor_linear = nn.Linear(num_channels[-1] * 10, output_size)

        self.s_dim = s_dim
        self.a_dim = a_dim
        # self.a1 = nn.Linear(s_dim, 200)
        self.mu = nn.Linear(num_channels[-1], a_dim)
        self.sigma = nn.Linear(num_channels[-1], a_dim)

        self.c1 = nn.Linear(s_dim, 100)
        self.v = nn.Linear(100, 1)

        self.distribution = torch.distributions.Normal

        self.init_weights()

    def init_weights(self):
        self.mu.weight.data.normal_(0, 0.01)
        self.sigma.weight.data.normal_(0, 0.01)
        self.c1.weight.data.normal_(0, 0.01)
        self.v.weight.data.normal_(0, 0.01)

    def forward(self, x):
        '''
        a1 = F.relu6(self.a1(x))
        mu = 2 * F.tanh(self.mu(a1))
        sigma = F.softplus(self.sigma(a1)) + 0.001      # avoid 0
        c1 = F.relu6(self.c1(x))
        values = self.v(c1)
        return mu, sigma, values
        '''

        a1 = F.relu6(self.tcn(x))
        # print(a1.size())
        mu = 2 * torch.tanh(self.mu(a1.transpose(1, 2)))
        sigma = F.softplus(self.sigma(a1.transpose(1, 2))) + 0.001      # avoid 0
        
        c1 = F.relu6(self.c1(x.transpose(1, 2)))
        values = self.v(c1)

        return mu, sigma, values


    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
        return m.sample().numpy()

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        # self.lnet = Net(N_S, N_A)           # local network
        self.lnet = TCN(input_size, output_size, num_channels, kernel_size, dropout)
        self.env = gym.make(env_id).unwrapped    # VibrationEnv  Pendulum

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            for t in range(MAX_EP_STEP):
                if self.name == 'w0':
                    self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a.clip(-2, 2))
                if t == MAX_EP_STEP - 1:
                    done = True
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append((r+8.1)/8.1)    # normalize

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1

        self.res_queue.put(None)


if __name__ == "__main__":
    # gnet = Net(N_S, N_A)        # global network
    gnet = TCN(input_size, output_size, num_channels, kernel_size, dropout)
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=0.0002)  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    # workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i = 0)]
    # w = Worker(gnet, opt, global_ep, global_ep_r, res_queue, 0)
    # w.start()
    # print(1000)
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
