#ifndef __HULLWHITE_H_INCLUDED__
#define __HULLWHITE_H_INCLUDED__

#include <cmath>
#include <numeric>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <EigenRand/EigenRand>
#include <boost/math/quadrature/trapezoidal.hpp>

using namespace Eigen;
using boost::math::quadrature::trapezoidal;


namespace hullwhite {

	template <typename T>
	std::vector<T> linspace(T start, T end, int num)
	{
		std::vector<T> linspaced;
		if (0 != num){
			if (1 == num){
				linspaced.push_back(static_cast<T>(start));
			}else{
				double delta = (end - start) / (num - 1);
				for (auto i = 0; i < (num - 1); ++i){
					linspaced.push_back(static_cast<T>(start + delta * i));
				}
				// ensure that start and end are exactly the same as the input
				linspaced.push_back(static_cast<T>(end));
			}
		}
		return linspaced;
	}

	struct PathsStruct {
		ArrayXXd M;
		ArrayXd T;
	};

	ArrayXXd GenNormDistArray(size_t rows, size_t cols, double mu, double sigma) {
		std::random_device rd;
		//Rand::P8_mt19937_64 urng{ rd()};
		Rand::P8_mt19937_64 urng{ 42 };
		Rand::NormalGen<double> norm_gen{ mu, sigma };
		ArrayXXd mat = norm_gen.template generate<ArrayXXd>(rows, cols, urng);
		return mat;
	}

	

	
	auto forward(
		const double& t,
		const double& dt,
		const std::function<double(double)>& P0T){

		return -(log(P0T(t + dt)) - log(P0T(t - dt))) / (2 * dt);				
	}

	auto HW_r_0(
		const double& lambda,
		const double& eta,
		const std::function<double(double)>& P0T) {
		return forward(0.00001, 0.0001, P0T);
	}

	template <typename MeanRevertParameter, typename Volatility, typename MarketCurve, typename Time>
	auto HW_A(
		const MeanRevertParameter& lambda,
		const Volatility& eta,
		const MarketCurve& P0T,
		const Time& T1, const Time& T2
	) {
		auto tau = T2 - T1;
		//const ArrayXi zGrid = ArrayXi::LinSpaced(0.0, tau, 250);
		auto B_r = [&](const auto& t) {return 1.0 / lambda * (std::exp(-lambda * t) - 1.0); };
		auto theta = [&](const auto& t) {
			return 1.0 / lambda * (forward(t + 0.0001, 0.0001, P0T) - forward(t - 0.0001, 0.0001, P0T)) / (2.0 * 0.0001) + forward(t, 0.0001, P0T) + eta * eta / (2.0 * lambda * lambda) * (1.0 - std::exp(-2.0 * lambda * t));
		};
		auto f = [&](const auto& t) { return theta(T2-t) * B_r(t); };
		auto temp1 = lambda * trapezoidal(f, 0.0, tau);
		auto temp2 = eta * eta / (4.0 * pow(lambda, 3.0)) * (std::exp(-2.0 * lambda * tau) * (4 * std::exp(lambda * tau) - 1.0) - 3.0) + eta * eta * tau / (2.0 * lambda * lambda);
		return temp1 + temp2;
	}

	template <typename MeanRevertParameter, typename Volatility, typename Time>
	auto HW_B(
		const MeanRevertParameter& lambda,
		const Volatility& eta,
		const Time& T1, const Time& T2
	) {
		return 1.0 / lambda * (std::exp(-lambda * (T2 - T1)) - 1.0);
	}

	template <typename MeanRevertParameter, typename Volatility, typename MarketCurve, typename Time, typename ShortRate>
	auto P_t_T(
		const MeanRevertParameter& lambda,
		const Volatility& eta,
		const MarketCurve& P0T,
		const Time& T1, const Time& T2, const ShortRate& rT1
	) {
		
		if (T1 < T2) {
			auto B_r = HW_B(lambda, eta, T1, T2);
			auto A_r = HW_A(lambda, eta, P0T, T1, T2);
			return std::exp(A_r + B_r * rT1);
		}
		else return 1.0;
	}


	template <typename NumOfPaths, typename NumOfSteps, typename Time, typename MarketCurve, typename MeanRevertParam, typename Volatility>
	auto GeneratePathsHWEuler(
		const NumOfPaths& numOfPaths,
		const NumOfSteps& numOfSteps,
		const Time& T,
		const MarketCurve& P0T,
		const MeanRevertParam& lambda,
		const Volatility& eta
	) {

		auto deltaT = 0.01;

		auto r0 = forward(0.01, deltaT, P0T);

		auto dt = (double)T / numOfSteps;

		auto theta = [&](const auto& t) {
			return 1.0 / lambda * (forward(t + dt, deltaT, P0T) - forward(t - dt, deltaT, P0T)) / (2.0 * 0.0001) + forward(t, deltaT, P0T) + eta * eta / (2.0 * lambda * lambda) * (1.0 - std::exp(-2.0 * lambda * t));
		};

		ArrayXXd W(numOfPaths, numOfSteps + 1); W.setZero();
		ArrayXXd R(numOfPaths, numOfSteps + 1); R.setZero(); R.col(0) = r0;
		ArrayXXd M(numOfPaths, numOfSteps + 1); M.setZero(); M.col(0) = 1.0;
		ArrayXXd Z = GenNormDistArray(numOfPaths, numOfSteps, 0.0, 1.0);
		ArrayXd time(numOfSteps + 1);time.setZero();

		

		for (size_t i = 0; i < numOfSteps; ++i) {
			W.col(i + 1) = W.col(i) + std::sqrt(dt) * Z.col(i);
			R.col(i + 1) = R.col(i) + lambda * (theta(time[i]) - R.col(i)) * dt + eta * (W.col(i + 1) - W.col(i));
			M.col(i + 1) = M.col(i) * exp((R.col(i + 1) + R.col(i)) * 0.5 * dt);
			time[i + 1] = time[i] + dt;
		}

		return PathsStruct { M, time };
		
	}


}


#endif