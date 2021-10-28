#include "catch.hpp"
#include <../src/BlackScholesJumps.h>
#include <tbb/parallel_for.h>
#include <mutex>
#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <numeric>



using namespace std::chrono;

TEST_CASE("parallel for") {

    const size_t num_of_steps = 500;
    const size_t T = 5;
    const double S0 = 100.0;
    const double K = 80.0;
    const double muJ = 0.3;
    const double sigmaJ = 0.005;
    const double r = 0.0;
    const size_t initSeed = 106;
    std::vector<size_t> seeds;

    auto start = high_resolution_clock::now();

    const ArrayXi NGrid = ArrayXi::LinSpaced(10, 100, 10000);
    const size_t num_of_runs = NGrid.size();
    seeds.resize(num_of_runs);
    std::iota(seeds.begin(), seeds.end(), initSeed);
    ArrayXd resultsMC(num_of_runs), resultCondExp(num_of_runs);
    tbb::parallel_for(size_t(0), num_of_runs, [&](size_t i) {
        //std::cout << "running loop for " << NGrid[i] << "\n";
        const PathsStruct paths = generate_paths(seeds[i], NGrid[i], num_of_steps, S0, T, muJ, sigmaJ, r);
        const ArrayXd terminalS = paths.S.col(num_of_steps);
        const ArrayXd terminalJ = paths.J.col(num_of_steps - 1);
        resultsMC[i] = EUOptionPriceMC(OptionType::CALL, terminalS, 80.0, 5.0, 0.0);
        resultCondExp[i] = CallOptionCondExpectation(NGrid[i], T, S0, K, terminalJ, r);
    });

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "It took " << duration.count() / 1000.0 << " seconds" << "\n";

    std::cout << "--- Monte Carlo Option Price Convergence zig-zag pattern ----" << "\n";
    std::cout << resultsMC << "\n";
    std::cout << "--- Conditional Expectation Convergence wow! almost a straight line ----" << "\n";
    std::cout << resultCondExp << "\n";
}
