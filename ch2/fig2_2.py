import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm, trange 
import random 

seed = 42
k = 10
time_step = 1000
amount_of_games = 2000
np.random.seed(seed)
actions = [i for i in range(k)]

def bandit(q_star, action): 
    mean = q_star[action]
    return np.random.normal(loc=mean)


def game(epsilon=0):
    reward_experiments = [] 
    action_experiments = [] 
    for i in trange(2000): 
        q_star = np.random.normal(size=k) 
        # init Q, N 
        Q = [0 for _ in range(k)] 
        N = [0 for _ in  range(k)] 
        
        action_history = [-1 for _ in range(1000)] 
        reward_history = [0 for _ in range(1000)] 

        for j in range(1000): 
            a = np.argmax(Q) 
            if random.random() < epsilon:
                a = np.random.choice(actions) 
            r = bandit(q_star, a) 

            N[a] += 1
            x = Q[a] + (1/N[a]) * (r - Q[a])
            Q[a] = x  

            reward_history[j] = r 
            action_history[j] = a

        reward_experiments.append(reward_history)
        action_experiments.append(action_history)

    # plot average reward over 1k run for 2k experiments 
    reward_experiments_arr = np.array(reward_experiments)
    return reward_experiments_arr

if __name__ == '__main__': 
    print(f'Greedy Method')
    reward_exps_greedy = game(0) 
    print(f'Epsilon 0.1')
    reward_exps_ep1 = game(0.1) 
    print(f'Epsilon 0.01')
    reward_exps_ep01 = game(0.01) 

    plt.plot([i for i in range(1, 1000+1)], reward_exps_greedy.mean(axis=0), label=f'greedy')
    plt.plot([i for i in range(1, 1000+1)], reward_exps_ep1.mean(axis=0), label=f'epsilon=0.1')
    plt.plot([i for i in range(1, 1000+1)], reward_exps_ep01.mean(axis=0), label=f'epsilon=0.01')
    plt.legend()
    plt.show()


