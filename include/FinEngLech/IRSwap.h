#ifndef __IRSWAP_H_INCLUDED__
#define __IRSWAP_H_INCLUDED__

#include <cmath>
#include <vector>
#include <numeric>
#include <optional>
#include <Eigen/Core>
#include <EigenRand/EigenRand>
#include <boost/math/quadrature/trapezoidal.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/adaptors.hpp>
#include <FinEngLech/Utils.h>
#include <FinEngLech/Discount.h>

using namespace Eigen;
using boost::math::quadrature::trapezoidal;


namespace lecture6 {

	enum class OptionTypeSwap { Payer = -1, Receiver = 1};

	class IRSwap {
	public:
		IRSwap(OptionTypeSwap optionTypeSwap, double notional, double strike, double t, double ti, double tm);
		
		double evaluate(const std::vector<double>& ri, std::shared_ptr<Discount> discounter,
			std::optional<std::shared_ptr<Discount>> forwarder) const ;

		void setT(double t) noexcept { t = t;}
		void setTi(double ti) noexcept { ti = ti;}
		void setTm(double tm) noexcept {tm = tm;}
	private:
		OptionTypeSwap optionTypeSwap{OptionTypeSwap::Payer};
		double notional{1.0};
		double strike {0.0};
		double t{0.0};
		double ti{0.0};
		double tm{0.0};
		int n {0};

	};
}

#endif
