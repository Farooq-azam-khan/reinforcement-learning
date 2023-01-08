'''
Design and conduct an experiment to demonstrate the
difficulties that sample-average methods have for nonstationary problems. Use a modified
version of the 10-armed testbed in which all the q_star(a) start out equal and then take
independent random walks (say by adding a normally distributed increment with mean
zero and standard deviation 0.01 to all the q_star(a) on each step). Prepare plots like
Figure 2.2 for an action-value method using sample averages, incrementally computed,
and another action-value method using a constant step-size parameter, α = 0.1. Use
ε = 0.1 and longer runs, say of 10,000 steps.
'''
import numpy as np 
from tqdm import trange
k = 10 
q_stars = np.zeros(k) 
total_steps = 100
alpha = 0.1 
epsilon = 0.1 
for step in range(total_steps): 
    q_stars += np.random.normal(loc=0.0, scale=0.01) 

def bandit(q_star, action): 
    mean = q_star[action]
    return np.random.normal(loc=mean)


def game(alpha_fn, actions, epsilon=0, total_simulations=2000, total_steps_per_simulation=1000):
    reward_experiments = [] 
    action_experiments = [] 
    for i in trange(total_simulations): 
        q_star = np.ones(k) 
        # init Q, N 
        Q = [0 for _ in range(k)] 
        N = [0 for _ in  range(k)] 
        
        action_history = [-1 for _ in range(total_steps_per_simulation)] 
        reward_history = [0 for _ in range(total_steps_per_simulation)] 

        for step in range(total_steps_per_simulation): 
            q_star += np.random.normal(loc=0, scale=0.01)
            a = np.argmax(Q) 
            if np.random.rand() < epsilon:
                a = np.random.choice(actions) 
            r = bandit(q_star, a) 

            N[a] += 1
            x = Q[a] + alpha_fn(N[a]) * (r - Q[a])
            Q[a] = x  

            reward_history[step] = r 
            action_history[step] = a

        reward_experiments.append(reward_history)
        action_experiments.append(action_history)

    # plot average reward over 1k run for 2k experiments 
    reward_experiments_arr = np.array(reward_experiments)
    return reward_experiments_arr
import matplotlib.pyplot as plt 
if __name__ == '__main__': 
    actions = np.arange(k)
    reward_experiments = game(alpha_fn=lambda _: alpha, actions=actions, epsilon=epsilon) 
    plt.plot([i for i in range(1, 1000+1)], reward_experiments.mean(axis=0), label=f'wandering q_star')
    plt.legend()
    plt.show()



