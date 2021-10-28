#pragma once
#include <Eigen/Dense>
#include <Eigen/Core>

using namespace Eigen;

enum class OptionType { CALL, PUT };

struct PathsStruct {
    ArrayXXd J;
    ArrayXXd S;
};
PathsStruct generate_paths(size_t seed, size_t num_of_paths, size_t num_of_steps, double S0, size_t T, double muJ, double sigmaJ, double r);
double EUOptionPriceMC(const OptionType option_type, const Eigen::ArrayXd& S, double K, double T, double r);
double CallOptionCondExpectation(size_t num_of_paths, size_t T, double S0, double K, const Eigen::ArrayXd& J, double r);
