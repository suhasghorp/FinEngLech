#ifndef __HULLWHITE2F_H_INCLUDED__
#define __HULLWHITE2F_H_INCLUDED__

#include <cmath>
#include <vector>
#include <numeric>
#include <Eigen/Core>
#include <EigenRand/EigenRand>
#include <boost/math/quadrature/trapezoidal.hpp>

using namespace Eigen;
using boost::math::quadrature::trapezoidal;


namespace hullwhite2f {

	template <typename T>
	std::vector<T> linspace(T start, T end, int num)
	{
		std::vector<T> linspaced;
		if (0 != num){
			if (1 == num){
				linspaced.push_back(static_cast<T>(start));
			}else{
				const double delta = (end - start) / (num - 1);
				for (auto i = 0; i < (num - 1); ++i){
					linspaced.push_back(static_cast<T>(start + delta * i));
				}
				// ensure that start and end are exactly the same as the input
				linspaced.push_back(static_cast<T>(end));
			}
		}
		return linspaced;
	}

	Eigen::ArrayXd F(const Eigen::ArrayXd& column){
		Eigen::ArrayXd abc;
		abc << 1.0, 2.0, 3.0;
		return column + abc;
	}

	inline ArrayXXd GenNormDistArray(size_t rows, size_t cols, double mu, double sigma) {
		ArrayXXd mat(rows, cols); mat.setZero();
		std::vector<size_t> seeds(cols);
		std::iota(seeds.begin(), seeds.end(), 42);	
		
		std::for_each(mat.colwise().begin(), mat.colwise().end(), [&seeds, mu, sigma, rows, i = 0](const auto& column) mutable {
			Rand::P8_mt19937_64 urng{ seeds.at(i) };
			Rand::NormalGen<double> norm_gen{ mu, sigma };
			column < norm_gen.generate<ArrayXd>(rows, 1, urng);
			++i;
		});
		return mat;
	}
	
	inline auto forward(
		const double& t,
		const double& dt,
		const std::function<double(double)>& P0T){

		return -(log(P0T(t + dt)) - log(P0T(t - dt))) / (2 * dt);				
	}

	inline auto HW_r_0(
		const double& lambda,
		const double& eta,
		const std::function<double(double)>& P0T) {
		return forward(0.001, 0.01, P0T);
	}

	template <typename MeanRevertParameter, typename Volatility, typename MarketCurve, typename Time>
	auto HW_A(
		const MeanRevertParameter& lambda,
		const Volatility& eta,
		const MarketCurve& P0T,
		const Time& T1, const Time& T2
	) {
		auto tau = T2 - T1;
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

	template <typename Lambda1, typename Lambda2, typename Eta1, typename Eta2, typename Rho, typename MarketCurve, typename T1, typename T2, typename xT1, typename yT1>
	auto P_t_T(
		const Lambda1& lambda1, 
		const Lambda2& lambda2, 
		const Eta1& eta1, 
		const Eta2& eta2, 
		const Rho& rho, 
		const MarketCurve& P0T, 
		const T1& t1, const T2& t2, const xT1& xt1, const yT1& yt1) {

		/*auto V = [&](const T1 t, const T1 T) {
			return pow(eta1, 2.0) / pow(lambda1, 2.0) * ((T - t) + 2.0 / lambda1 * std::exp(-lambda1 * (T - t)) - 1.0 / (2.0 * lambda1) * std::exp(-2.0 * lambda1 * (T - t)) - 3.0 / (2.0 * lambda1)) +
				pow(eta2, 2.0) / pow(lambda2, 2.0) * ((T - t) + 2.0 / lambda2 * std::exp(-lambda2 * (T - t)) - 1.0 / (2.0 * lambda2) * std::exp(-2.0 * lambda2 * (T - t)) - 3.0 / (2.0 * lambda2)) +
				2.0 * rho * eta1 * eta2 / (lambda1 * lambda2) * (T - t + 1.0 / lambda1 * (std::exp(-lambda1 * (T - t)) - 1.0) + 1.0 / lambda2 * (std::exp(-lambda2 * (T - t)) - 1.0) - 1.0 / (lambda1 + lambda2) * (std::exp(-(lambda1 + lambda2) * (T - t)) - 1.0));
		};*/

		auto V = [&](const T1 t, const T1 T) {
			return pow(eta1, 2.0) / pow(lambda1, 2.0) * ((t2 - t1) + 2.0 / lambda1 * std::exp(-lambda1 * (t2 - t1)) - 1.0 / (2.0 * lambda1) * std::exp(-2.0 * lambda1 * (t2 - t1)) - 3.0 / (2.0 * lambda1)) +
				pow(eta2, 2.0) / pow(lambda2, 2.0) * ((t2 - t1) + 2.0 / lambda2 * std::exp(-lambda2 * (t2 - t1)) - 1.0 / (2.0 * lambda2) * std::exp(-2.0 * lambda2 * (t2 - t1)) - 3.0 / (2.0 * lambda2)) +
				2.0 * rho * eta1 * eta2 / (lambda1 * lambda2) * (t2 - t1 + 1.0 / lambda1 * (std::exp(-lambda1 * (t2 - t1)) - 1.0) + 1.0 / lambda2 * (std::exp(-lambda2 * (t2 - t1)) - 1.0) - 1.0 / (lambda1 + lambda2) * (std::exp(-(lambda1 + lambda2) * (t2 - t1)) - 1.0));
		};

		auto intPhi = -log(P0T(t2) / P0T(t1) * std::exp(-0.5 * (V(0, t2) - V(0, t1))));

		double A = 1.0 / lambda1 * (1.0 - std::exp(-lambda1 * (t2 - t1)));
		double B = 1.0 / lambda2 * (1.0 - std::exp(-lambda2 * (t2 - t1)));

		return std::exp(-intPhi - A * xt1 - B * yt1 + 0.5 * V(t1, t2));

	}


	template <typename NumOfPaths, typename NumOfSteps, typename Time, typename MarketCurve, typename MeanRevertParam1, typename MeanRevertParam2, typename Volatility1, typename Volatility2, typename Correlation>
	ArrayXXd GeneratePathsHWEuler(
		const NumOfPaths& numOfPaths,
		const NumOfSteps& numOfSteps,
		const Time& T,
		const MarketCurve& P0T,
		const MeanRevertParam1& lambda1,
		const MeanRevertParam2& lambda2,
		const Volatility1& eta1,
		const Volatility2& eta2,
		const Correlation& rho
	) {

		auto deltaT = 0.01;

		auto r0 = forward(0.01, deltaT, P0T);

		auto phi = [&](const auto t) {
			return r0 + pow(eta1, 2.0) / (2.0 * pow(lambda1, 2.0)) * (1.0 - std::exp(-lambda1 * t)) * (1.0 - std::exp(-lambda1 * t))
				+ pow(eta2, 2.0) / (2.0 * pow(lambda2, 2.0)) * (1.0 - std::exp(-lambda2 * t)) * (1.0 - std::exp(-lambda2 * t))
				+ rho * eta1 * eta2 / (lambda1 * lambda2) * (1.0 - std::exp(-lambda1 * t)) * (1.0 - std::exp(-lambda2 * t));
		};		

		auto dt = static_cast<double>(T) / numOfSteps;

		ArrayXXd Z1 = GenNormDistArray(numOfPaths, numOfSteps, 0.0, 1.0);
		ArrayXXd Z2 = GenNormDistArray(numOfPaths, numOfSteps, 0.0, 1.0);
		ArrayXXd W1(numOfPaths, numOfSteps + 1); W1.setZero();
		ArrayXXd W2(numOfPaths, numOfSteps + 1); W2.setZero();		
		ArrayXXd X(numOfPaths, numOfSteps + 1); X.setZero(); 
		ArrayXXd Y(numOfPaths, numOfSteps + 1); Y.setZero();
		ArrayXXd R(numOfPaths, numOfSteps + 1); R.setZero(); R.col(0) = phi(0);
		ArrayXXd M(numOfPaths, numOfSteps + 1); M.setZero(); M.col(0) = 1.0;
		
		ArrayXd time(numOfSteps + 1); time.setZero();

		for (size_t i = 0; i < numOfSteps; ++i) {
			Z2.col(i) = rho * Z1.col(i) + std::sqrt(1.0 - pow(rho, 2)) * Z2.col(i);
			W1.col(i + 1) = W1.col(i) + std::sqrt(dt) * Z1.col(i);
			W2.col(i + 1) = W2.col(i) + std::sqrt(dt) * Z2.col(i);

			X.col(i + 1) = X.col(i) - lambda1 * X.col(i) * dt + eta1 * (W1.col(i + 1) - W1.col(i));
			Y.col(i + 1) = Y.col(i) - lambda2 * Y.col(i) * dt + eta2 * (W2.col(i + 1) - W2.col(i));
			time[i + 1] = time[i] + dt;
			R.col(i + 1) = X.col(i + 1) + Y.col(i + 1) + phi(time[i + 1]);	
			M.col(i + 1) = M.col(i) * exp((R.col(i + 1) + R.col(i)) * 0.5 * dt);
		}

		//return PathsStruct{ M, time };
		return M;
	}
}
#endif