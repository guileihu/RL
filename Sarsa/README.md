# Sarsa

1. concenpt
采取行动a，获得及时奖励R, 得到下一状态S_,依据该状态选中下一状态需要的a_,得到Q：
$$a_ = choose_action(S_) \\
Q_{target} = R + \gamma*Q(S_,a_) \\
Q(S,a) = Q(S,a) + \alpha * (Q_{target} - Q(S,a))
$$

2. code
./Sarsa_maze

3. 与Q-learning不同：
- Q table update:
    ``` python
    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_] # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
    ```

- step: #########!!!!!!处不同
    ```python
     # RL choose action based on observation
        action = RL.choose_action(str(observation)) #######!!!!!!!!!!!

        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_)) #########!!!!!!

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation and action
            observation = observation_
            action = action_ ############# !!!!!!!!!!!!!!!
    ```