#ifndef BAYMAR_OLS_SIMULATOR_H
#define BAYMAR_OLS_SIMULATOR_H

#include "./forecaster.h"

namespace baymar {

class MarSimulator;

class MarSimulator : public bvhar::MultistepForecastRun<Eigen::MatrixXd, Eigen::MatrixXd> {
public:
	MarSimulator(
		int num_iter, int num_burn,
		int lag,
		const Eigen::MatrixXd& init,
		const Eigen::MatrixXd& row_coef, const Eigen::MatrixXd& col_coef,
		const Eigen::MatrixXd& row_sig, const Eigen::MatrixXd& col_sig,
		unsigned int seed
	)
	: num_iter(num_iter), num_burn(num_burn), num_row(row_coef.cols()), res(num_iter) {
		Eigen::MatrixXd error_mean = Eigen::MatrixXd::Zero(num_row, col_coef.cols());
		auto dgp_updater = std::make_unique<MatGaussianErrorGenerator>(error_mean, row_sig, col_sig, seed);
		generator = std::make_unique<MarForecaster>(num_iter + num_burn, init, lag, row_coef, col_coef, NULLOPT, std::move(dgp_updater));
	}
	virtual ~MarSimulator() = default;

	std::vector<Eigen::MatrixXd> returnDgp() {
		Eigen::MatrixXd pred = generator->doForecast();
		for (int i = 0; i < num_iter; ++i) {
			res[i] = pred.middleRows(num_row * (num_burn + i), num_row);
		}
		return res;
	}

private:
	int num_iter, num_burn, num_row;
	std::vector<Eigen::MatrixXd> res;
	std::unique_ptr<MarForecaster> generator;
};

} // namespace baymar

#endif // BAYMAR_OLS_SIMULATOR_H