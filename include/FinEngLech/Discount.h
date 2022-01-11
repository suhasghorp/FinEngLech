#pragma once

#include <memory>
#include <libInterpolate/Interpolators/_1D/LinearInterpolator.hpp>
#include <libInterpolate/Interpolators/_1D/CubicSplineInterpolator.hpp>
#include <libInterpolate/Interpolators/_1D/MonotonicInterpolator.hpp>
#include <libInterpolate/Interpolators/_1D/AnyInterpolator.hpp>
#include <iostream>

namespace lecture6 {
	class Discount {
	public:
		explicit Discount(const _1D::AnyInterpolator<double>& interpolator) : interpolator{interpolator}{}
		double getDiscFactor(double t) { 
			return std::exp(-interpolator(t) * t); }


	private:
		_1D::AnyInterpolator<double> interpolator;
	};
}
