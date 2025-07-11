#ifndef BAYMAR_OLS_SIMULATOR_H
#define BAYMAR_OLS_SIMULATOR_H

#include "./forecaster.h"

namespace baymar {

class MarSimulator;
class MdfmSimulator;
// class MdfmVecSimulator;
class MdfmMarSimulator;

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

class MdfmSimulator : public bvhar::MultistepForecastRun<Eigen::MatrixXd, Eigen::MatrixXd> {
public:
	MdfmSimulator(
		int num_iter, int num_burn,
		const Eigen::MatrixXd& init,
		const Eigen::MatrixXd& row_coef, const Eigen::MatrixXd& col_coef,
		const Eigen::MatrixXd& row_sig, const Eigen::MatrixXd& col_sig,
		unsigned int seed
	)
	: num_iter(num_iter), num_burn(num_burn),
		num_row(row_coef.rows()), num_col(col_coef.rows()),
		nrow_factor(row_coef.cols()), ncol_factor(col_coef.cols()),
		row_coef(row_coef), col_coef(col_coef),
		// pred(Eigen::MatrixXd::Zero(num_row * num_iter, num_col)),
		pred(Eigen::MatrixXd::Zero(num_row, num_col)),
		factor_mat(Eigen::MatrixXd::Zero(num_iter * nrow_factor, ncol_factor)),
		res(num_iter) {
		Eigen::MatrixXd error_mean = Eigen::MatrixXd::Zero(num_row, num_col);
		dgp_updater = std::make_unique<MatGaussianErrorGenerator>(error_mean, row_sig, col_sig, seed);
		// generator = std::make_unique<MarForecaster>(num_iter + num_burn, init, lag, row_coef, col_coef, NULLOPT, std::move(dgp_updater));
		// int num_init = init.rows() / nrow_factor;
		// generator = std::make_unique<MatExogenForecaster>(0, init, num_init, num_row, num_col);
	}
	virtual ~MdfmSimulator() = default;

	LIST returnDgp() {
		generateFactor();
		generator = std::make_unique<MatExogenForecaster>(0, factor_mat, factor_mat.rows() / nrow_factor, num_row, num_col);
		generator->updateCoef(row_coef, col_coef);
		for (int i = 0; i < num_iter; ++i) {
			pred.setZero();
			generator->appendForecast(pred, 0);
			res[i] = pred;
		}
		return CREATE_LIST(
			NAMED("y") = WRAP(res),
			NAMED("factor") = factor_mat
		);
	}

protected:
	int num_iter, num_burn, num_row, num_col, nrow_factor, ncol_factor;
	Eigen::MatrixXd row_coef, col_coef, pred;
	Eigen::MatrixXd factor_mat;
	std::vector<Eigen::MatrixXd> res;
	std::unique_ptr<MatGaussianErrorGenerator> dgp_updater;
	std::unique_ptr<MatExogenForecaster> generator;

	virtual void generateFactor() = 0;
};

class MdfmMarSimulator : public MdfmSimulator {
public:
	MdfmMarSimulator(
		int num_iter, int num_burn,
		int lag,
		const Eigen::MatrixXd& init,
		const Eigen::MatrixXd& row_coef, const Eigen::MatrixXd& col_coef, const Eigen::MatrixXd& row_sig, const Eigen::MatrixXd& col_sig,
		const Eigen::MatrixXd& factor_init,
		const Eigen::MatrixXd& fac_row_coef, const Eigen::MatrixXd& fac_col_coef, const Eigen::MatrixXd& fac_row_sig, const Eigen::MatrixXd& fac_col_sig,
		unsigned int seed
	)
	: MdfmSimulator(num_iter, num_burn, init, row_coef, col_coef, row_sig, col_sig, seed) {
		// nrow_factor(fac_row_coef.cols()), ncol_factor(fac_col_coef.cols()),
		// factor_mat(Eigen::MatrixXd::Zero(num_iter * nrow_factor, ncol_factor)) {
		Eigen::MatrixXd error_mean = Eigen::MatrixXd::Zero(nrow_factor, ncol_factor);
		auto factor_dgp_updater = std::make_unique<MatGaussianErrorGenerator>(error_mean, fac_row_sig, fac_col_sig, seed);
		factor_generator = std::make_unique<MarForecaster>(num_iter + num_burn, init, lag, fac_row_coef, fac_col_coef, NULLOPT, std::move(factor_dgp_updater));
	}
	virtual ~MdfmMarSimulator() = default;

private:
	// int nrow_factor, ncol_factor;
	Eigen::MatrixXd factor_mat;
	std::unique_ptr<MarForecaster> factor_generator;

	void generateFactor() override {
		factor_mat = factor_generator->doForecast(); // F_1, ..., F_{num_iter + num_burn}
	}
};

} // namespace baymar

#endif // BAYMAR_OLS_SIMULATOR_H