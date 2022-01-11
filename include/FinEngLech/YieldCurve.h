#pragma once
#include <memory>
#include <vector>
#include <FinEngLech/IRSwap.h>
#include <libInterpolate/Interpolate.hpp>
#include <libInterpolate/AnyInterpolator.hpp>

namespace lecture6 {

	class YieldCurve {
	public:

		YieldCurve(const YieldCurve&) = delete;
		YieldCurve(const YieldCurve&&) = delete;
		
		YieldCurve(_1D::AnyInterpolator<double>& interpolator, 
			std::vector<std::shared_ptr<lecture6::IRSwap>>& instruments, 
			std::vector<double>& maturities, 
			std::vector<double>& r,
			double tolerance);			

		void multivariateNewtonRaphson() ;

		std::vector<double> evalInstruments(const std::vector<double>& r) ;

		
		MatrixXd jacobian() ;

		const std::vector<double> getR() const;

	private:
		std::vector<double> maturities;
		std::vector<std::shared_ptr<lecture6::IRSwap>> instruments;		
		std::vector<double> r;
		double tolerance;
		_1D::AnyInterpolator<double> interpolator;
		std::shared_ptr<Discount> discounter;
		std::shared_ptr<Discount> forwarder;
		
	};
}


