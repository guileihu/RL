# Q_learning

1. concenpt
采取行动a，获得及时奖励R, 得到下一状态S_,依据该状态中可能得所有actions,获得一个可能得最大的Q:
$$Q_{target} = R + \gamma*Q(S_,a_{max}) \\
Q(S,a) = Q(S,a) + \alpha * (Q_{target} - Q(S,a))
$$

2. code
./Q_Learning_maze

3. 关键代码：
- choose action:以epsilon概率选择最优决策，epsilon可以由q_table的丰富而降低随机决策的概率
    ``` python
    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action
    ```
- Q table update:
    ``` python
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
    ```