// Monte Carlo pricer for European options under risk-neutral GBM. Mirrors mc.py
// and mc.R exactly -- same model, same step-by-step simulation, same table
// format -- so run.py's `compare` subcommand can diff all three against each
// other and against Black-Scholes.
//
// Standard library only.
//
// Build:
//   g++ -O2 -std=c++17 -o mc mc.cpp
// Run:
//   ./mc --s0 100 --r 0.03 --sigma 0.2 --T 1 --n-steps 252 --n-paths 10000 \
//        --strikes 95,100,105 --seed 123 [--antithetic] [--json out.json]

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Row {
    double K, call, call_se, put, put_se;
};

std::vector<double> simulate_terminal(double S0, double r, double sigma, double T,
                                       int n_steps, int n_paths, unsigned seed,
                                       bool antithetic) {
    if (!(S0 > 0 && sigma > 0 && T > 0 && n_steps > 0 && n_paths > 0)) {
        throw std::invalid_argument("S0, sigma, T, n_steps and n_paths must be strictly positive");
    }
    if (antithetic && n_paths % 2 != 0) {
        throw std::invalid_argument("antithetic sampling needs an even n_paths");
    }

    std::mt19937 gen(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    double dt = T / n_steps;
    double drift = (r - 0.5 * sigma * sigma) * dt;
    double vol = sigma * std::sqrt(dt);

    std::vector<double> ST(static_cast<size_t>(n_paths));

    if (antithetic) {
        int half = n_paths / 2;
        for (int i = 0; i < half; ++i) {
            double log_plus = 0.0, log_minus = 0.0;
            for (int t = 0; t < n_steps; ++t) {
                double z = normal(gen);
                log_plus += drift + vol * z;
                log_minus += drift - vol * z;
            }
            ST[2 * i] = S0 * std::exp(log_plus);
            ST[2 * i + 1] = S0 * std::exp(log_minus);
        }
    } else {
        for (int i = 0; i < n_paths; ++i) {
            double log_terminal = 0.0;
            for (int t = 0; t < n_steps; ++t) {
                log_terminal += drift + vol * normal(gen);
            }
            ST[i] = S0 * std::exp(log_terminal);
        }
    }
    return ST;
}

double mean_of(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

double stddev_of(const std::vector<double>& v, double m) {
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / static_cast<double>(v.size() - 1));
}

Row price_one_strike(const std::vector<double>& ST, double K, double discount,
                      bool antithetic) {
    size_t n_paths = ST.size();
    std::vector<double> call_pay(n_paths), put_pay(n_paths);
    for (size_t i = 0; i < n_paths; ++i) {
        call_pay[i] = std::max(ST[i] - K, 0.0);
        put_pay[i] = std::max(K - ST[i], 0.0);
    }

    std::vector<double> call_sample, put_sample;
    if (antithetic) {
        size_t half = n_paths / 2;
        call_sample.resize(half);
        put_sample.resize(half);
        for (size_t i = 0; i < half; ++i) {
            call_sample[i] = (call_pay[2 * i] + call_pay[2 * i + 1]) / 2.0;
            put_sample[i] = (put_pay[2 * i] + put_pay[2 * i + 1]) / 2.0;
        }
    } else {
        call_sample = call_pay;
        put_sample = put_pay;
    }

    double call_mean = mean_of(call_sample);
    double put_mean = mean_of(put_sample);
    double call_sd = stddev_of(call_sample, call_mean);
    double put_sd = stddev_of(put_sample, put_mean);
    double n_eff = static_cast<double>(call_sample.size());

    Row row;
    row.K = K;
    row.call = discount * call_mean;
    row.call_se = discount * call_sd / std::sqrt(n_eff);
    row.put = discount * put_mean;
    row.put_se = discount * put_sd / std::sqrt(n_eff);
    return row;
}

std::string get_arg(const std::vector<std::string>& args, const std::string& flag,
                     const std::string& def) {
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == flag && i + 1 < args.size()) return args[i + 1];
    }
    return def;
}

bool has_flag(const std::vector<std::string>& args, const std::string& flag) {
    return std::find(args.begin(), args.end(), flag) != args.end();
}

std::vector<double> parse_strikes(const std::string& s) {
    std::vector<double> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(std::stod(item));
    return out;
}

void write_json(const std::vector<Row>& rows, const std::string& path) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "unable to open " << path << " for writing\n";
        return;
    }
    f << std::fixed << std::setprecision(6);
    f << "[\n";
    for (size_t i = 0; i < rows.size(); ++i) {
        const Row& row = rows[i];
        f << "  {\"K\": " << row.K << ", \"call\": " << row.call
          << ", \"call_se\": " << row.call_se << ", \"put\": " << row.put
          << ", \"put_se\": " << row.put_se << "}";
        f << (i + 1 < rows.size() ? ",\n" : "\n");
    }
    f << "]\n";
}

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);

    try {
        double S0 = std::stod(get_arg(args, "--s0", "100"));
        double r = std::stod(get_arg(args, "--r", "0.03"));
        double sigma = std::stod(get_arg(args, "--sigma", "0.2"));
        double T = std::stod(get_arg(args, "--T", "1"));
        int n_steps = std::stoi(get_arg(args, "--n-steps", "252"));
        int n_paths = std::stoi(get_arg(args, "--n-paths", "10000"));
        std::vector<double> strikes = parse_strikes(get_arg(args, "--strikes", "95,100,105"));
        unsigned seed = static_cast<unsigned>(std::stoul(get_arg(args, "--seed", "123")));
        bool antithetic = has_flag(args, "--antithetic");
        std::string json_path = get_arg(args, "--json", "");

        std::vector<double> ST = simulate_terminal(S0, r, sigma, T, n_steps, n_paths, seed,
                                                     antithetic);
        double discount = std::exp(-r * T);

        std::vector<Row> rows;
        for (double K : strikes) rows.push_back(price_one_strike(ST, K, discount, antithetic));

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "K call call_se put put_se\n";
        for (const Row& row : rows) {
            std::cout << row.K << " " << row.call << " " << row.call_se << " "
                      << row.put << " " << row.put_se << "\n";
        }

        if (!json_path.empty()) write_json(rows, json_path);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
