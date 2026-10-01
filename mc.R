# Monte Carlo pricer for European options under risk-neutral GBM. Mirrors mc.py
# exactly -- same model, same step-by-step simulation, same table format -- so
# run.py's `compare` subcommand can run this file, the C++ binary and the Python
# implementation on identical parameters and diff the output.
#
# Run directly:
#   Rscript mc.R --s0 100 --r 0.03 --sigma 0.2 --T 1 --n-steps 252 --n-paths 10000 \
#       --strikes 95,100,105 --seed 123 [--antithetic] [--json out.json]

simulate_terminal <- function(S0, r, sigma, T, n_steps, n_paths, seed = NULL, antithetic = FALSE) {
  if (!(S0 > 0 && sigma > 0 && T > 0 && n_steps > 0 && n_paths > 0)) {
    stop("S0, sigma, T, n_steps and n_paths must be strictly positive")
  }
  if (antithetic && n_paths %% 2 != 0) {
    stop("antithetic sampling needs an even n_paths")
  }
  if (!is.null(seed)) set.seed(seed)

  dt <- T / n_steps
  drift <- (r - 0.5 * sigma^2) * dt
  vol <- sigma * sqrt(dt)

  if (antithetic) {
    half <- n_paths / 2
    Z <- matrix(rnorm(half * n_steps), nrow = half, ncol = n_steps)
    log_plus <- rowSums(drift + vol * Z)
    log_minus <- rowSums(drift - vol * Z)
    ST <- numeric(n_paths)
    ST[seq(1, n_paths, by = 2)] <- S0 * exp(log_plus)
    ST[seq(2, n_paths, by = 2)] <- S0 * exp(log_minus)
  } else {
    Z <- matrix(rnorm(n_paths * n_steps), nrow = n_paths, ncol = n_steps)
    log_terminal <- rowSums(drift + vol * Z)
    ST <- S0 * exp(log_terminal)
  }
  ST
}

price_european <- function(S0, strikes, r, sigma, T, n_steps, n_paths, seed = NULL,
                            antithetic = FALSE) {
  if (length(strikes) == 0 || any(strikes <= 0)) {
    stop("strikes must be a non-empty list of strictly positive numbers")
  }
  ST <- simulate_terminal(S0, r, sigma, T, n_steps, n_paths, seed, antithetic)
  discount <- exp(-r * T)

  rows <- vector("list", length(strikes))
  for (i in seq_along(strikes)) {
    K <- strikes[i]
    call_pay <- pmax(ST - K, 0)
    put_pay <- pmax(K - ST, 0)

    if (antithetic) {
      call_sample <- (call_pay[seq(1, n_paths, by = 2)] + call_pay[seq(2, n_paths, by = 2)]) / 2
      put_sample <- (put_pay[seq(1, n_paths, by = 2)] + put_pay[seq(2, n_paths, by = 2)]) / 2
    } else {
      call_sample <- call_pay
      put_sample <- put_pay
    }

    n_eff <- length(call_sample)
    rows[[i]] <- list(
      K = K,
      call = discount * mean(call_sample),
      call_se = discount * sd(call_sample) / sqrt(n_eff),
      put = discount * mean(put_sample),
      put_se = discount * sd(put_sample) / sqrt(n_eff)
    )
  }
  rows
}

print_table <- function(rows) {
  cat("K call call_se put put_se\n")
  for (row in rows) {
    cat(sprintf("%.6f %.6f %.6f %.6f %.6f\n",
                row$K, row$call, row$call_se, row$put, row$put_se))
  }
}

write_json <- function(rows, path) {
  entries <- vapply(rows, function(row) {
    sprintf('  {"K": %.6f, "call": %.6f, "call_se": %.6f, "put": %.6f, "put_se": %.6f}',
            row$K, row$call, row$call_se, row$put, row$put_se)
  }, character(1))
  writeLines(paste0("[\n", paste(entries, collapse = ",\n"), "\n]"), path)
}

# ---- CLI ---------------------------------------------------------------
# Guarded so that source("mc.R") -- from a test, or from another R script --
# only defines the functions above and does not also run a simulation.
if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)

  get_arg <- function(flag, default) {
    i <- which(args == flag)
    if (length(i) == 0) return(default)
    args[i + 1]
  }
  has_flag <- function(flag) flag %in% args

  s0 <- as.numeric(get_arg("--s0", "100"))
  r <- as.numeric(get_arg("--r", "0.03"))
  sigma <- as.numeric(get_arg("--sigma", "0.2"))
  Tt <- as.numeric(get_arg("--T", "1"))
  n_steps <- as.integer(get_arg("--n-steps", "252"))
  n_paths <- as.integer(get_arg("--n-paths", "10000"))
  strikes <- as.numeric(strsplit(get_arg("--strikes", "95,100,105"), ",")[[1]])
  seed <- as.integer(get_arg("--seed", "123"))
  antithetic <- has_flag("--antithetic")
  json_path <- get_arg("--json", NA)

  rows <- price_european(s0, strikes, r, sigma, Tt, n_steps, n_paths, seed, antithetic)
  print_table(rows)

  if (!is.na(json_path)) {
    write_json(rows, json_path)
  }
}
