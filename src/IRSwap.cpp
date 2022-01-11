#include <FinEngLech/IRSwap.h>
#include <iostream>

namespace lecture6 {

	IRSwap::IRSwap(OptionTypeSwap optionTypeSwap, double notional, double strike, double t, double ti, double tm) :
		optionTypeSwap{ optionTypeSwap }, notional{ notional }, strike{ strike }, t{ t }, ti{ ti }, tm{ tm }{
		n = static_cast<int>(4 * tm);
	}

	
	double IRSwap::evaluate(const std::vector<double>& ri, std::shared_ptr<Discount> discounter,
		std::optional <std::shared_ptr<Discount>> forwarder) const {
		double ti_new{ 0.0 };
		const auto tigrid = utils::linspace(ti, tm, n);
		const auto tau = tigrid[1] - tigrid[0];
		std::vector<double> prevTi{}, nextTi{};
		std::copy_if(tigrid.begin(), tigrid.end(), std::back_inserter(prevTi), [this](double x) {return x < t; });
		if (prevTi.size() > 0)
			ti_new = prevTi.back();
		std::copy_if(tigrid.begin(), tigrid.end(), std::back_inserter(nextTi), [this](double x) {return x > t; });

		const auto filtered_grid = boost::copy_range<std::vector<double>>(tigrid | boost::adaptors::filtered([this](double x) {return x > t; }));
		auto lamb = [&](double sum, double tii){
			return sum + (tau * discounter->getDiscFactor(tii)); };
		auto temp = std::accumulate(filtered_grid.begin(), filtered_grid.end(), 0.0, lamb);			

		const auto P_t_Ti = discounter->getDiscFactor(ti);
		const auto P_t_Tm = discounter->getDiscFactor(tm);

		return optionTypeSwap == OptionTypeSwap::Payer ?
			notional * ((P_t_Ti - P_t_Tm) - strike * temp) :
			notional * (strike * temp - (P_t_Ti - P_t_Tm));
	}

}

