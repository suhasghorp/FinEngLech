#pragma once

#include <vector>

namespace utils {

	template <typename T>
	std::vector<T> linspace(T start, T end, int num)
	{
		std::vector<T> linspaced;
		if (0 != num) {
			if (1 == num) {
				linspaced.push_back(static_cast<T>(start));
			}
			else {
				const double delta = (end - start) / (num - 1);
				for (auto i = 0; i < (num - 1); ++i) {
					linspaced.push_back(static_cast<T>(start + delta * i));
				}
				// ensure that start and end are exactly the same as the input
				linspaced.push_back(static_cast<T>(end));
			}
		}
		return linspaced;
	}

	


}
