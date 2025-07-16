# DQN
1. concept
采取行动a，获得及时奖励R, 得到下一状态S_,依据该状态中可能得所有actions,获得一个可能得最大的Q:
$$Q_{target} = R + \gamma*Q_{\_}(S_{\_}) \\
Q_{eval} = Q(S) \\
loss = loss_{\_}fn(Q_{target}, Q_{eval})
一定步数后：
_ = Q
$$

```python
# Q值计算
        q_eval = self.eval_net(s).gather(1, a.unsqueeze(1)).squeeze()
        q_next = self.target_net(s_).detach().max(1)[0]
        q_target = r + self.gamma * q_next
        
        # 反向传播
        loss = self.loss_fn(q_eval, q_target)
```

2. 使用DQN代替Q_table:
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class DQN(nn.Module):
    def __init__(self, n_actions, n_features):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_features, 10)
        self.fc2 = nn.Linear(10, n_actions)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc1.weight, 0, 0.3)
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.normal_(self.fc2.weight, 0, 0.3)
        nn.init.constant_(self.fc2.bias, 0.1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, s, a, r, s_):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (s, a, r, s_)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_ = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(np.array(a)),
            torch.FloatTensor(np.array(r)),
            torch.FloatTensor(np.array(s_))
        )
        
class DQNAgent:
    def __init__(self, n_actions, n_features, lr=0.01, gamma=0.9, epsilon=0.9, 
                 replace_target_iter=300, memory_size = 500):
        self.eval_net = DQN(n_actions, n_features)
        self.target_net = DQN(n_actions, n_features)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        
        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.memory = ReplayBuffer(memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.replace_target_iter = replace_target_iter
        self.cost_his = []
        self.learn_step_counter = 0

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            with torch.no_grad():
                q_values = self.eval_net(torch.FloatTensor(state))
            return q_values.argmax().item()
        else:
            return np.random.randint(self.n_actions)

    def learn(self, batch_size=32):
        # 目标网络更新逻辑
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        
        # 经验采样
        s, a, r, s_ = self.memory.sample(batch_size)
        
        # Q值计算
        q_eval = self.eval_net(s).gather(1, a.unsqueeze(1)).squeeze()
        q_next = self.target_net(s_).detach().max(1)[0]
        q_target = r + self.gamma * q_next
        
        # 反向传播
        loss = self.loss_fn(q_eval, q_target)
        self.cost_his.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1
    
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()   

```