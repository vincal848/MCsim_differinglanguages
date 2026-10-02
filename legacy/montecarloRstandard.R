# SUPERSEDED. Kept for reference only -- this file is not part of the package
# and is known to be incorrect.
#
# Known defects, each with a named regression test:
#
# 1. Paths are simulated with real-world drift mu = 0.05 but discounted at the
#    risk-free rate r = 0.03. That is not risk-neutral pricing. At these
#    parameters it overprices the K=105 call by about 15% (1.05 vs a
#    Black-Scholes price of 7.13, tens of standard errors away). Pinned by
#    test_mc_price_matches_black_scholes_within_3_se in tests/test_mc.py,
#    which the replacement passes and this script would not.
# 2. simulated_paths[1, ] is set to S0 and the loop only fills rows 2:n_steps,
#    so only n_steps - 1 = 251 steps are actually simulated, while the
#    discount factor uses exp(-r * n_steps * dt) = exp(-r * T) for the full
#    year. The number of steps simulated and the horizon discounted over
#    disagree.
# 3. Builds a 10000 x 252 = 2,520,000-row payoff_data frame that nothing in
#    the script reads.
# 4. Strike prices (95, 105) do not match the C++ version's (105) or the
#    Python version, which does not price at all -- the three "same model"
#    scripts price different things.
#
# Replaced by mc.R (functions) and run.py / mc.py, the Python driver that also
# runs this file's replacement and the C++ replacement for comparison.

set.seed(123)
n_simulations <- 10000
n_steps <- 252 
S0 <- 100 
mu <- 0.05 
sigma <- 0.2 
dt <- 1/n_steps
strike_prices <- c(95, 105) 
risk_free_rate <- 0.03 


simulated_paths <- matrix(0, nrow=n_steps, ncol=n_simulations)
simulated_paths[1, ] <- S0


for (i in 2:n_steps) {
  Z <- rnorm(n_simulations)
  simulated_paths[i, ] <- simulated_paths[i - 1, ] * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
}


option_prices <- c()
payoffs <- matrix(0, nrow=n_simulations, ncol=length(strike_prices))
for (k in 1:length(strike_prices)) {
  strike_price <- strike_prices[k]
  total_payoff <- 0
  for (j in 1:n_simulations) {
    final_price <- simulated_paths[n_steps, j]
    payoff <- max(final_price - strike_price, 0)
    payoffs[j, k] <- payoff
    total_payoff <- total_payoff + payoff
  }
  average_payoff <- total_payoff / n_simulations
  option_price <- average_payoff * exp(-risk_free_rate * n_steps * dt)
  option_prices <- c(option_prices, option_price)
}


for (i in 1:length(strike_prices)) {
  cat("Estimated Call Option Price for Strike Price", strike_prices[i], ":", option_prices[i], "\n")
}


payoff_data <- data.frame(Simulation=rep(1:n_simulations, each=n_steps),
                          TimeStep=rep(1:n_steps, n_simulations),
                          Price=as.vector(simulated_paths),
                          Payoff=rep(payoffs[, 1], each=n_steps))



matplot(simulated_paths[, 1:10], type="l", col=1:10, lty=1, xlab="Time Steps", ylab="Stock Price", main="Monte Carlo Simulated Stock Price Paths for R")
