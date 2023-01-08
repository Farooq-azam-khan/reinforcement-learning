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


def game(initial_Q_val=0, epsilon=0):
    reward_experiments = [] 
    action_experiments = [] 

    for i in trange(2000): 
        q_star = np.random.normal(size=k) 
        optimal_action = q_star.argmax() #TOOD: make % optimal action list 
        # init Q, N 
        Q = [initial_Q_val for _ in range(k)] 
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
    return np.array(reward_experiments)

if __name__ == '__main__': 
    print(f'Q1 = 5 (greedy)')
    q5_greedy = game(initial_Q_val=5, epsilon=0) 
    print(f'Q1 = 0, ep=0.1 ')
    q0_ep01 = game(initial_Q_val=0, epsilon=0.1)

    plt.plot(q5_greedy.mean(axis=0), label=f'greedy, Q1 = 5 for all actions')
    plt.plot(q0_ep01.mean(axis=0), label=f'ep0.1, Q1 = 0 for all actions')
    plt.legend()
    plt.show()


