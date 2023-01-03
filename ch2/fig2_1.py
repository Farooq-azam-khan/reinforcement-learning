import matplotlib.pyplot as plt
import numpy as np 
seed = 42
k = 10

np.random.seed(seed)
q_star = np.random.normal(size=k) 
rewards_dist = [] 
for q in q_star: 
    rewards_dist.append(np.random.normal(loc=q, size=1000))
rewards_dist = np.vstack(rewards_dist) 
plt.violinplot(rewards_dist.transpose())
plt.scatter([i for i in range(1,k+1)], q_star) 
plt.xlabel("Action")
plt.ylabel("Reward distribution")
plt.show()
