#ifndef BAYMAR_OLS_SIMULATOR_H
#define BAYMAR_OLS_SIMULATOR_H

#include "./forecaster.h"
#include <bvhar/ols>

namespace baymar {

class MarSimulator;
class MdfmSimulator;
class MdfmVecSimulator;
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
		generator = std::make_unique<MarForecaster>(num_iter + num_burn, init, lag, row_coef, col_coef, BVHAR_NULLOPT, std::move(dgp_updater));
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
		const Eigen::MatrixXd& row_coef, const Eigen::MatrixXd& col_coef,
		const Eigen::MatrixXd& row_sig, const Eigen::MatrixXd& col_sig,
		unsigned int seed
	)
	: num_iter(num_iter), num_burn(num_burn),
		num_row(row_coef.cols()), num_col(col_coef.cols()),
		nrow_factor(row_coef.rows()), ncol_factor(col_coef.rows()),
		row_coef(row_coef), col_coef(col_coef),
		pred(Eigen::MatrixXd::Zero(num_row, num_col)),
		factor_mat(num_iter), res(num_iter) {
		Eigen::MatrixXd error_mean = Eigen::MatrixXd::Zero(num_row, num_col);
		dgp_updater = std::make_unique<MatGaussianErrorGenerator>(error_mean, row_sig, col_sig, seed);
		// generator = std::make_unique<MarForecaster>(num_iter + num_burn, init, lag, row_coef, col_coef, BVHAR_NULLOPT, std::move(dgp_updater));
		// int num_init = init.rows() / nrow_factor;
		// generator = std::make_unique<MatExogenForecaster>(0, init, num_init, num_row, num_col);
	}
	virtual ~MdfmSimulator() = default;

	BVHAR_LIST returnDgp() {
		generateFactor();
		// Eigen::MatrixXd factor_acc = std::accumulate(
		// 	factor_mat.begin() + 1, factor_mat.end(), factor_mat[0],
		// 	[](const Eigen::MatrixXd& acc, const Eigen::MatrixXd& curr) {
		// 		Eigen::MatrixXd concat_mat(acc.rows() + curr.rows(), acc.cols());
		// 		concat_mat << acc,
		// 									curr;
		// 		return concat_mat;
		// 	}
		// );
		// generator = std::make_unique<MatExogenForecaster>(0, factor_acc, factor_acc.rows() / nrow_factor, num_row, num_col);
		// generator->updateCoef(row_coef, col_coef);
		for (int i = 0; i < num_burn; ++i) {
			pred.setZero();
			dgp_updater->appendError(pred);
		}
		for (int i = 0; i < num_iter; ++i) {
			// generator->appendForecast(pred, i);
			pred = row_coef.transpose() * factor_mat[i] * col_coef;
			dgp_updater->appendError(pred);
			res[i] = pred;
		}
		return BVHAR_CREATE_LIST(
			BVHAR_NAMED("y") = BVHAR_WRAP(res),
			BVHAR_NAMED("factor") = BVHAR_WRAP(factor_mat)
		);
	}

protected:
	int num_iter, num_burn, num_row, num_col, nrow_factor, ncol_factor;
	Eigen::MatrixXd row_coef, col_coef, pred;
	std::vector<Eigen::MatrixXd> factor_mat;
	std::vector<Eigen::MatrixXd> res;
	std::unique_ptr<MatGaussianErrorGenerator> dgp_updater;
	std::unique_ptr<MatExogenForecaster> generator;

	virtual void generateFactor() = 0;
};

class MdfmVecSimulator : public MdfmSimulator {
public:
	MdfmVecSimulator(
		int num_iter, int num_burn,
		int lag,
		const Eigen::MatrixXd& row_coef, const Eigen::MatrixXd& col_coef, const Eigen::MatrixXd& row_sig, const Eigen::MatrixXd& col_sig,
		const Eigen::MatrixXd& factor_init,
		const Eigen::MatrixXd& factor_coef, const Eigen::MatrixXd& factor_sig,
		unsigned int seed
	)
	: MdfmSimulator(num_iter, num_burn, row_coef, col_coef, row_sig, col_sig, seed),
		factor_draw(Eigen::MatrixXd::Zero(num_iter, factor_coef.cols())),
		ncol_factor(col_coef.rows()) {
		factor_generator = std::make_unique<bvhar::OlsSimulator>(
			num_iter, num_burn, lag,
			factor_init, factor_coef, factor_sig, 2, seed
		);
	}
	virtual ~MdfmVecSimulator() = default;

protected:
	void generateFactor() override {
		factor_draw = factor_generator->returnDgp();
		for (int i = 0; i < num_iter; ++i) {
			factor_mat[i] = bvhar::unvectorize(factor_draw.row(i).transpose(), ncol_factor);
		}
	}

private:
	std::unique_ptr<bvhar::OlsSimulator> factor_generator;
	Eigen::MatrixXd factor_draw;
	int ncol_factor;
};

class MdfmMarSimulator : public MdfmSimulator {
public:
	MdfmMarSimulator(
		int num_iter, int num_burn,
		int lag,
		const Eigen::MatrixXd& row_coef, const Eigen::MatrixXd& col_coef, const Eigen::MatrixXd& row_sig, const Eigen::MatrixXd& col_sig,
		const Eigen::MatrixXd& factor_init,
		const Eigen::MatrixXd& fac_row_coef, const Eigen::MatrixXd& fac_col_coef, const Eigen::MatrixXd& fac_row_sig, const Eigen::MatrixXd& fac_col_sig,
		unsigned int seed
	)
	: MdfmSimulator(num_iter, num_burn, row_coef, col_coef, row_sig, col_sig, seed) {
		Eigen::MatrixXd error_mean = Eigen::MatrixXd::Zero(nrow_factor, ncol_factor);
		factor_generator = std::make_unique<MarSimulator>(
			num_iter, num_burn, lag,
			factor_init, fac_row_coef, fac_col_coef,
			fac_row_sig, fac_col_sig, seed
		);
	}
	virtual ~MdfmMarSimulator() = default;

protected:
	void generateFactor() override {
		factor_mat = factor_generator->returnDgp();
	}

private:
	std::unique_ptr<MarSimulator> factor_generator;
};

} // namespace baymar

#endif // BAYMAR_OLS_SIMULATOR_H