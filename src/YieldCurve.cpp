#include <memory>
#include <vector>
#include <FinEngLech/YieldCurve.h>
#include <FinEngLech/IRSwap.h>
#include <libInterpolate/Interpolate.hpp>
#include <libInterpolate/AnyInterpolator.hpp>
#include <FinEngLech/Discount.h>

namespace lecture6 {
	
	YieldCurve::YieldCurve(_1D::AnyInterpolator<double>& interpolator, std::vector<std::shared_ptr<lecture6::IRSwap>>& instruments,
		std::vector<double>& maturities,
		std::vector<double>& r,
		double tolerance) :
		interpolator{interpolator},instruments(instruments), maturities(maturities), r(r), tolerance{tolerance}{
			
	}

	void YieldCurve::multivariateNewtonRaphson() {

		double err = 10e10; //not const, we will change it
	
		while (err > tolerance) {			
			auto values = evalInstruments(r);
			VectorXd valuesVector = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(values.data(), values.size());
			MatrixXd J = jacobian();
			MatrixXd J_inverse = J.inverse();
			VectorXd errM = - J_inverse * valuesVector; 
			std::transform(begin(r), end(r), errM.begin(), begin(r), std::plus<double>());
			err = errM.norm();
		}
	}

	std::vector<double> YieldCurve::evalInstruments(const std::vector<double>& rin) {
		std::vector<double> values;
		auto m_interp{maturities};
		auto r_interp{rin};
		m_interp.insert(m_interp.begin(), 0.0);
		r_interp.insert(r_interp.begin(), r_interp[0]);
		interpolator.setData(m_interp.size(), m_interp.data(), r_interp.data());
		discounter = std::make_shared<Discount>(interpolator);
		for (std::shared_ptr<IRSwap>& swap : instruments) {
			values.push_back(swap->evaluate(rin, discounter, std::nullopt));
		}
		return values;
	}

	MatrixXd YieldCurve::jacobian() {
		const double eps = 1e-05;
		const int n = instruments.size();
		MatrixXd J = MatrixXd::Zero(n,n);
		std::vector<double> dv(n, 0.0);
		auto vals = evalInstruments(r);
		std::vector<double> r_up {r};
		for (size_t index = 0; index < r.size(); ++index){
			
			r_up[index] = r[index] + eps;
			std::vector<double> vals_up = evalInstruments(r_up);
			r_up[index] = r[index];
			for (size_t i = 0; i < vals.size(); ++i) {
				dv[i] = (vals_up[i] - vals[i])/eps;
			}
			J.col(index) = Eigen::Map<Eigen::VectorXd>(dv.data(), dv.size());			
		}
		return J;
	}

	const std::vector<double> YieldCurve::getR() const {
		return r;
	}
}

	

