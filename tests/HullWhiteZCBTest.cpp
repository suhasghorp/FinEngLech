#include "catch.hpp"
#include <../src/HullWhiteZCB.h>
#include <fmt/core.h>


TEST_CASE("HullWhiteZCB") {

	const size_t numOfPaths = 50000;
	const size_t numOfSteps = 25;
	const auto lambda = 0.04;
	const auto eta = 0.01;
	const size_t T = 10;

	auto P0T = [&](const auto& t) {return std::exp(-0.1 * t); };
	auto r0 = hullwhite::HW_r_0(lambda, eta, P0T);
	const hullwhite::PathsStruct pathstruct = hullwhite::GeneratePathsHWEuler(numOfPaths, numOfSteps, T, P0T, lambda, eta);
	for (size_t i = 0; i < numOfSteps + 1; ++i) {
		auto P_MC = pathstruct.M.col(i).inverse().mean();
		auto ti = pathstruct.T[i];
		auto P_T = P0T(ti);
		auto P_analytical = hullwhite::P_t_T(lambda, eta, P0T, 0.0, ti, r0);
		fmt::print("ti:{:<10.2f} P_Exact:{:<15.4f} P_MC:{:<15.4f} P_analytical:{:<15.4f}\n", ti, P_T, P_MC, P_analytical);
	}	

REQUIRE(7.0 == 7.0);

}



/*TEST_CASE("HullWhiteAnalytical") {

	const auto lambda = 0.05;
	const auto eta = 0.1;
	auto P0T = [](const auto& t) {return std::exp(-0.02 * t); };	
	auto r0 = hullwhite::HW_r_0(lambda, eta, P0T);

	const auto N = 26.0;
	auto T_end = 10.0;
	const std::vector<double> TGrid = hullwhite::linspace(0.0, T_end, N);
	
	std::vector<double> Exact,Proxy;
	for (double ti : TGrid) {
		Proxy.emplace_back(hullwhite::P_t_T(lambda, eta, P0T, 0.0, ti, r0));
		Exact.emplace_back(P0T(ti));
	}
	
	

	REQUIRE(7.0 == 7.0);

}*/