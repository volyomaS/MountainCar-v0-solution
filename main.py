import numpy as np
import pandas as pd
import gym
import matplotlib.pyplot as plt

n_pos = 19
n_vel = 15
iter_max = 2000

initial_lr = 1
min_lr = 0.003
gamma = 1
t_max = 201
eps = 0.05
factor = 0.9

if __name__ == '__main__':
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    res = []
    q_table = np.zeros((n_pos, n_vel, 3))
    for i in range(iter_max):
        eta = max(min_lr, initial_lr * (factor ** (i//25)))
        obs = env.reset()
        max_p = -1
        for j in range(t_max):
            if i % 500 == 0:
                env.render()
            p, v = int(round(obs[0]*10+6)), int(round(obs[1]*100+7))
            max_p = max(max_p, obs[0])
            if np.random.uniform(0, 1) < eps:
                action = np.random.choice(env.action_space.n)
            else:
                m = max(q_table[p][v])
                for k in range(len(q_table[p][v])):
                    if q_table[p][v][k] == m:
                        action = k
            obs, reward, done, info = env.step(action)
            p_, v_ = int(round(obs[0]*10+6)), int(round(obs[1]*100+7))
            q_table[p][v][action] = q_table[p][v][action] + eta * (reward + gamma*np.max(q_table[p_][v_]) - q_table[p][v][action])
            if obs[0] >= 0.5:
                print("You win on %d iteration with %d steps" % (i+1, j+1))
            if done:
                break
        res.append(max_p)
    res = pd.DataFrame(res)
    res.plot()
    plt.show()
    # plt.savefig("result.png")
