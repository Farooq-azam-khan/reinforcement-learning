import matplotlib.pyplot as plt
import numpy as np 
seed = 42
k = 10
time_step = 1000

np.random.seed(seed)
q_star = np.random.normal(size=k) 
rewards_dist = [] 
for q in q_star: 
    rewards_dist.append(np.random.normal(loc=q, size=1000))
rewards_dist = np.vstack(rewards_dist) 

Q = [0 for _ in range(k)]
N = [0 for _ in range(k)]

import random 
actions = [i for i in range(k)] 

def run(time_step, epsilon=0):
    print(f'{epsilon=}')
    total_reward = 0 
    for _ in range(time_step): 
        a = np.argmax(Q) 
        if random.random() < epsilon:
            a = np.random.choice(actions) 
        #print(f'selected action: {a=}', end=' ') 
        r = rewards_dist[a][0] 
        #print(f'and got reward: {r=}')
        N[a] += 1
        x = Q[a] + (1/N[a]) * (r - Q[a])
        Q[a] = x  
        total_reward += r 
    print(f'{Q=}')
    print(f'{total_reward=:.2f}, average_reward={total_reward/time_step:.2f}')
print(f'{q_star=}') 
run(time_step)
run(time_step, 0.1)
run(time_step, 0.01)
