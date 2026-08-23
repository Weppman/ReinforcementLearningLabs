import matplotlib.pyplot as plt
from banditsOperations import run_eGreedy, run_greedyInitialization, run_UCB

# Deliverable 1 Settings: epsilon=0.1, Q1=5, c=2
egreedy_data = run_eGreedy(epsilon=0.1)
optimistic_data = run_greedyInitialization(initial_Q=5)
ucb_data = run_UCB(c=2)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(egreedy_data, label=r"$\epsilon$-Greedy ($\epsilon=0.1$)")
plt.plot(optimistic_data, label=r"Optimistic Greedy ($Q_1=5$)")
plt.plot(ucb_data, label=r"UCB ($c=2$)")

plt.xlabel("Steps")
plt.ylabel("Average Reward (Over 100 Runs)")
plt.title("10-Arm Bandit Algorithm Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()