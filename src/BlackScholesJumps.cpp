// BlackScholesJumps.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <BlackScholesJumps.h>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <random>
#include <numeric>
#include <chrono>

#include <EigenRand/EigenRand>

using namespace std::chrono;



ArrayXXd gen_normdist_array(size_t seed, size_t rows, size_t cols, double mu, double sigma) {
    Rand::P8_mt19937_64 urng{ seed };
    Rand::NormalGen<double> norm_gen{ mu, sigma };
    ArrayXXd mat = norm_gen.template generate<ArrayXXd>(rows, cols, urng);
    return mat;
}

PathsStruct generate_paths(size_t seed, size_t num_of_paths, size_t num_of_steps, double S0, size_t T, double muJ, double sigmaJ, double r) {
    ArrayXXd X(num_of_paths, num_of_steps + 1);
    ArrayXXd S(num_of_paths, num_of_steps + 1);
    X.col(0) = std::log(S0);
    S.col(0) = S0;
    ArrayXXd Z = gen_normdist_array(seed, num_of_paths, num_of_steps, 0.0, 1.0);
    ArrayXXd J = gen_normdist_array(seed + 100, num_of_paths, num_of_steps, muJ, sigmaJ);
    const double dt = static_cast<double>(T) / num_of_steps;
    for (size_t i = 0; i < num_of_steps; ++i) {
        X.col(i + 1) = X.col(i) + ((r - 0.5 * J.col(i).square()) * dt) + (J.col(i) * std::sqrt(dt) * Z.col(i));
    }
    S = X.exp();
    return PathsStruct{ J,S };
}

template<typename S, typename Discount, typename K, typename Sigma>
auto BSCall(const S& S0, const Discount& discount, const K& k, const Sigma& sigma) {
    //note that sigma includes sqrt(T-t) term so in vanilla BS sigma is equal to volatility*sqrt(T-t)
    //Discount is exp(-r*(T-t))
    double s = sqrt(2.0);
    auto d1 = log(S0 / (discount * k)) / (sigma)+sigma * .5;
    return S0 * (.5 + .5 * erf(d1 / s)) - k * discount * (.5 + .5 * (erf((d1 - sigma) / s)));

}

template<typename S, typename Discount, typename K, typename Sigma>
auto BSPut(const S& S0, const Discount& discount, const K& k, const Sigma& sigma) {
    double s = sqrt(2.0);
    auto d1 = log(S0 / (discount * k)) / (sigma)+sigma * .5;
    return S0 * (.5 * erf(d1 / s) - .5) + k * discount * (.5 - .5 * (erf((d1 - sigma) / s)));
}

double EUOptionPriceMC(const OptionType option_type, const Eigen::ArrayXd& S, double K, double T, double r) {
    if (option_type == OptionType::CALL) {
        return std::exp(-r * T) * (S - K).max(0.0).mean();
    }
    else return std::exp(-r * T) * (K - S).max(0.0).mean();
}

double CallOptionCondExpectation(size_t num_of_paths, size_t T, double S0, double K, const Eigen::ArrayXd& J, double r) {
    ArrayXd results(num_of_paths);
    for (size_t i = 0; i < num_of_paths; ++i) {
        double sigma = J[i];
        results[i] = BSCall(S0, std::exp(-r * T), K, sigma * std::sqrt(T));
    }
    return results.mean();
}
