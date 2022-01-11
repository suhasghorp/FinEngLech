#include <catch.hpp>
#include <FinEngLech/IRSwap.h>
#include <fmt/core.h>
#include <tuple>
#include <FinEngLech/Utils.h>
#include <FinEngLech/YieldCurve.h>
#include <libInterpolate/AnyInterpolator.hpp>

using namespace lecture6;

TEST_CASE("YieldCurveTest", "[Lecture 6]") {

	std::vector<double> maturities{ 1.0,2.0,3.0,5.0,7.0,10.0,20.0,30.0 };
	std::vector<double> r0{ 0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01 };
	const double tolerance = 1.0e-15;
	auto print = [](const double& n) { std::cout << " " << n; };
	const std::vector<double> expected{ 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 };
	
	std::vector<std::shared_ptr<IRSwap>> instruments{};
	instruments.reserve(8);
	instruments.emplace_back(std::make_shared<IRSwap>(OptionTypeSwap::Payer, 1.0, 0.04 / 100.0, 0.0, 0.0, 1.0));
	instruments.emplace_back(std::make_shared<IRSwap>(OptionTypeSwap::Payer, 1.0, 0.16 / 100.0, 0.0, 0.0, 2.0));
	instruments.emplace_back(std::make_shared<IRSwap>(OptionTypeSwap::Payer, 1.0, 0.31 / 100.0, 0.0, 0.0, 3.0));
	instruments.emplace_back(std::make_shared<IRSwap>(OptionTypeSwap::Payer, 1.0, 0.81 / 100.0, 0.0, 0.0, 5.0));
	instruments.emplace_back(std::make_shared<IRSwap>(OptionTypeSwap::Payer, 1.0, 1.28 / 100.0, 0.0, 0.0, 7.0));
	instruments.emplace_back(std::make_shared<IRSwap>(OptionTypeSwap::Payer, 1.0, 1.62 / 100.0, 0.0, 0.0, 10.0));
	instruments.emplace_back(std::make_shared<IRSwap>(OptionTypeSwap::Payer, 1.0, 2.22 / 100.0, 0.0, 0.0, 20.0));
	instruments.emplace_back(std::make_shared<IRSwap>(OptionTypeSwap::Payer, 1.0, 2.30 / 100.0, 0.0, 0.0, 30.0));

	/******  LINEAR INTERPOLATION ******/

	_1D::AnyInterpolator<double> linear_interp = _1D::LinearInterpolator<double>();	
	lecture6::YieldCurve yc_linear(linear_interp,instruments,maturities,r0, 1.0e-15);
	yc_linear.multivariateNewtonRaphson();
	std::vector<double> r = yc_linear.getR();
	std::cout << "Zero rates - linear interpolation" << "\n";
	std::for_each(r.cbegin(), r.cend(), print);
	std::cout << "\n";
	auto vals = yc_linear.evalInstruments(r);
	REQUIRE_THAT(vals, Catch::Approx(expected).margin(1.e-5));

	/******  SPLINE INTERPOLATION ******/

	_1D::AnyInterpolator<double> spline_interp = _1D::CubicSplineInterpolator<double>();
	lecture6::YieldCurve yc_spline(spline_interp, instruments, maturities, r0, 1.0e-15);
	yc_spline.multivariateNewtonRaphson();
	r = yc_spline.getR();
	std::cout << "Zero rates - spline interpolation" << "\n";
	std::for_each(r.cbegin(), r.cend(), print);
	std::cout << "\n";
	vals = yc_spline.evalInstruments(r);
	REQUIRE_THAT(vals, Catch::Approx(expected).margin(1.e-5));

};