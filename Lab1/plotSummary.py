import numpy as np
import matplotlib.pyplot as plt
from banditsOperations import run_eGreedy, run_greedyInitialization, run_UCB

# Note the '.0' on every numerator so Python 2 does NOT do integer division!
eps_values = [1.0/128, 1.0/64, 1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0/2]
opt_values = [1.0/4, 1.0/2, 1.0, 2.0, 4.0, 8.0]
ucb_values = [1.0/16, 1.0/8, 1.0/4, 1.0/2, 1.0, 2.0, 4.0]

print("Running E-Greedy sweep...")
egreedy_scores = []
for eps in eps_values:
    avg_curve = run_eGreedy(epsilon=eps, runs=100, iterations=1000)
    egreedy_scores.append(np.mean(avg_curve))

print("Running Optimistic Greedy sweep...")
opt_scores = []
for q0 in opt_values:
    avg_curve = run_greedyInitialization(initial_Q=q0, runs=100, iterations=1000)
    opt_scores.append(np.mean(avg_curve))

print("Running UCB sweep...")
ucb_scores = []
for c_val in ucb_values:
    avg_curve = run_UCB(c=c_val, runs=100, iterations=1000)
    ucb_scores.append(np.mean(avg_curve))

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(eps_values, egreedy_scores, label=r'$\epsilon$-greedy ($\epsilon$)', marker='o', color='blue')
plt.plot(opt_values, opt_scores, label=r'Optimistic Greedy ($Q_0$)', marker='s', color='orange')
plt.plot(ucb_values, ucb_scores, label=r'UCB ($c$)', marker='^', color='green')

# Python 2 / ROS 1 Matplotlib log-scale syntax
plt.xscale('log', basex=2)

# Set x limits explicitly so all points from 1/128 (2^-7) to 8 (2^3) are visible
plt.xlim(1.0/256, 16.0)

plt.xlabel('Hyperparameter value (log scale $2^x$)')
plt.ylabel('Average reward over first 1000 steps')
plt.title('Summary Comparison of Bandit Algorithms Across Hyperparameters')
plt.legend(loc='upper left')
plt.grid(True, which="both", ls="--", alpha=0.4)

# Adjust layout so labels don't clip
plt.tight_layout()

plt.savefig("summary_comparison.png", dpi=300)
plt.show()