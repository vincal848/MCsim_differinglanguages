"""SUPERSEDED. Kept for reference only -- this file is not part of the package
and is known to be incorrect.

Known defects:

1. Same drift/discounting intent as the other two scripts -- except this one
   does no pricing at all, so there is nothing here to check against
   Black-Scholes. It only plots.
2. No functions -- everything runs at import time, so nothing here is
   callable from a test or from run.py.
3. Calls plt.show() twice, which blocks and opens windows; a script that only
   plots should at least let the caller choose to save instead.
4. Depends on seaborn for a histogram+KDE that matplotlib's hist() plus the
   exact lognormal density does equally well without the extra dependency --
   see plots.terminal_hist.

Replaced by mc.py (simulation and pricing) and plots.py (paths_fan and
terminal_hist).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(123) 
n_simulations = 10000
n_steps = 252  
S0 = 100  
mu = 0.05  
sigma = 0.2  
dt = 1 / n_steps


simulated_paths = np.zeros((n_steps, n_simulations))
simulated_paths[0, :] = S0


for i in range(1, n_steps):
    Z = np.random.normal(size=n_simulations)
    simulated_paths[i, :] = simulated_paths[i - 1, :] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

final_prices = simulated_paths[-1, :]


plt.figure(figsize=(10, 6))
plt.plot(simulated_paths[:, :10])
plt.xlabel("Time Steps")
plt.ylabel("Stock Price")
plt.title("Monte Carlo Simulated Stock Price Paths")
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(final_prices, bins=50, kde=True)
plt.xlabel("Final Stock Price")
plt.ylabel("Frequency")
plt.title("Distribution of Final Stock Prices from Monte Carlo Simulations")
plt.show()
