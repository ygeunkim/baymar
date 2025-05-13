#ifndef BAYMAR_BAYES_MNIW_FORECASTER_H
#define BAYMAR_BAYES_MNIW_FORECASTER_H

#include "./mniw.h"
#include "../../math/design.h"

namespace baymar {

class MatMniwForecaster;
class MatMniwForecastRun;
template <bool isUpdate> class MatMniwOutForecastRun;
template <bool isUpdate> class MatMniwRollForecastRun;
template <bool isUpdate> class MatMniwExpandForecastRun;

class MatMniwForecaster : public bvhar::BayesForecaster<Eigen::MatrixXd, Eigen::MatrixXd> {
public:
	MatMniwForecaster(const MatMniwRecords& records, int step, const Eigen::MatrixXd& y, int num_data, int lag, unsigned int seed)
	: bvhar::BayesForecaster<Eigen::MatrixXd, Eigen::MatrixXd>(step, y, lag, records.row_coef_record.rows(), seed),
		mat_record(std::make_unique<MatMniwRecords>(records)),
		num_row(y.rows() / num_data), num_col(y.cols()), nrow_row_coef(num_row * lag), nrow_col_coef(num_col * lag),
		row_coef(Eigen::MatrixXd::Zero(nrow_row_coef, num_row)),
		row_sig_lower(Eigen::MatrixXd::Identity(num_row, num_row)),
		col_coef(Eigen::MatrixXd::Zero(nrow_col_coef, num_col)),
		col_sig_lower(Eigen::MatrixXd::Identity(num_col, num_col)),
		error_mat(Eigen::MatrixXd::Zero(num_row, num_col)) {
		initLagged();
	}
	virtual ~MatMniwForecaster() = default;
	
	Eigen::MatrixXd getLastForecast() override {
		return this->doForecast().bottomRows(num_row);
	}

	Eigen::MatrixXd getLastForecast(const Eigen::MatrixXd& valid_vec) override {
		return this->doForecast(valid_vec).bottomRows(num_row);
	}

protected:
	std::unique_ptr<MatMniwRecords> mat_record;
	int num_row, num_col, nrow_row_coef, nrow_col_coef;
	Eigen::MatrixXd row_coef, row_sig_lower, col_coef, col_sig_lower, error_mat;

	void initLagged() override {
		last_pvec = build_dense_design(response, lag);
		point_forecast = Eigen::MatrixXd::Zero(num_row, num_col);
		pred_save = Eigen::MatrixXd::Zero(step * num_row, num_sim * num_col);
		tmp_vec = last_pvec.block(num_row, num_col, num_row * (lag - 1), num_col * (lag - 1));
	}

	void initRecursion(const Eigen::MatrixXd& obs_vec) override {
		last_pvec = obs_vec;
		point_forecast = obs_vec.topLeftCorner(num_row, num_col);
		tmp_vec = obs_vec.bottomRightCorner(num_row * (lag - 1), num_col * (lag - 1));
	}

	void setRecursion() override {
		last_pvec.bottomRightCorner(num_row * (lag - 1), num_col * (lag - 1)) = tmp_vec;
		last_pvec.topLeftCorner(num_row, num_col) = point_forecast;
	}

	void updatePred(const int h, const int i) override {
		computeMean();
		updateVariance();
		// point_forecast += error_mat;
		pred_save.block(h * num_row, i * num_col, num_row, num_col) = point_forecast + error_mat;
	}

	void updateRecursion() override {
		tmp_vec = last_pvec.topLeftCorner(num_row * (lag - 1), num_col * (lag - 1));
	}

	void computeMean() {
		// point_forecast = row_coef.transpose() * last_pvec.sparseView() * col_coef;
		point_forecast.setZero();
	// #ifdef _OPENMP
	// 	#pragma omp parallel for reduction(+:point_forecast)
	// #endif
		for (int i = 0; i < lag; ++i) {
			point_forecast += row_coef.middleRows(i * num_row, num_row).transpose() * last_pvec.block(i * num_row, i * num_col, num_row, num_col) * col_coef.middleRows(i * num_col, num_col);
		}
	}

	void updateParams(const int i) override {
		row_coef = bvhar::unvectorize(mat_record->row_coef_record.row(i).transpose(), num_row);
		col_coef = bvhar::unvectorize(mat_record->col_coef_record.row(i).transpose(), num_col);
		fill_lower(row_sig_lower, mat_record->row_sigma_record.row(i).transpose());
		fill_lower(col_sig_lower, mat_record->col_sigma_record.row(i).transpose());
	}

	void updateVariance() {
		for (int j = 0; j < num_col; ++j) {
			for (int i = 0; i < num_row; ++i) {
				error_mat(i, j) = bvhar::normal_rand(rng);
			}
		}
		error_mat = row_sig_lower * error_mat * col_sig_lower.transpose();
	}

	void updateLpl(int h, const Eigen::MatrixXd& valid_vec) override {}
};

inline std::vector<std::unique_ptr<MatMniwForecaster>> initialize_matmniwforecaster(
	int num_chains, int lag, int step, const Eigen::MatrixXd& y, int num_data,
	LIST& fit_record, Eigen::Ref<const Eigen::VectorXi> seed_chain, int nthreads
) {
	PY_LIST row_coef_record = fit_record["A_record"];
	PY_LIST row_sigma_record = fit_record["SigmaR_record"];
	PY_LIST col_coef_record = fit_record["B_record"];
	PY_LIST col_sigma_record = fit_record["SigmaC_record"];
	std::vector<std::unique_ptr<MatMniwForecaster>> forecaster(num_chains);
	for (int i = 0; i < num_chains; ++i) {
		MatMniwRecords mat_record(
			CAST<Eigen::MatrixXd>(row_coef_record[i]),
			CAST<Eigen::MatrixXd>(row_sigma_record[i]),
			CAST<Eigen::MatrixXd>(col_coef_record[i]),
			CAST<Eigen::MatrixXd>(col_sigma_record[i])
		);
		forecaster[i] = std::make_unique<MatMniwForecaster>(mat_record, step, y, num_data, lag, static_cast<unsigned int>(seed_chain[i]));
	}
	return forecaster;
}

class MatMniwForecastRun : public bvhar::McmcForecastRun<Eigen::MatrixXd, Eigen::MatrixXd> {
public:
	MatMniwForecastRun(
		int num_chains, int lag, int step, const Eigen::MatrixXd& y, int num_data,
		LIST& fit_record, const Eigen::VectorXi& seed_chain, int nthreads
	)
	: bvhar::McmcForecastRun<Eigen::MatrixXd, Eigen::MatrixXd>(num_chains, lag, step, nthreads) {
		auto temp_forecaster = initialize_matmniwforecaster(num_chains, lag, step, y, num_data, fit_record, seed_chain, nthreads);
		for (int i = 0; i < num_chains; ++i) {
			forecaster[i] = std::move(temp_forecaster[i]);
		}
	}
	virtual ~MatMniwForecastRun() = default;
};

template <bool isUpdate = true>
class MatMniwOutForecastRun : public bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate> {
public:
	MatMniwOutForecastRun(
		const Eigen::MatrixXd& y, int num_data, int lag,
		int num_chains, int num_iter, int num_burn, int thin, LIST& fit_record,
		LIST& param_coef_sig, LIST_OF_LIST& coef_sig_init,
		LIST& row_prior, LIST_OF_LIST& row_init, const int row_prior_type,
		LIST& col_prior, LIST_OF_LIST& col_init, const int col_prior_type,
		int step, const Eigen::MatrixXd& y_test,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads
	)
	: bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>(
			num_data, lag, num_chains, num_iter, num_burn, thin, step, y_test, false,
			seed_chain, seed_forecast, display_progress, nthreads
		),
		num_row(y.rows() / num_data), num_col(y.cols()), nrow_row_coef(num_row * lag), nrow_col_coef(num_col * lag) {}
	virtual ~MatMniwOutForecastRun() = default;

protected:
	int num_row, num_col, nrow_row_coef, nrow_col_coef;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::num_window;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::num_test;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::num_horizon;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::step;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::lag;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::num_chains;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::num_iter;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::num_burn;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::thin;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::nthreads;
	// using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::get_lpl;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::display_progress;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::seed_forecast;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::roll_mat;
	// using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::roll_y0;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::y_test;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::model;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::forecaster;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::out_forecast;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::lpl_record;

	Eigen::MatrixXd getValid() override {
		return y_test.bottomRows(num_row);
	}

	void initialize(
		const Eigen::MatrixXd& y, LIST& fit_record,
		LIST& param_coef_sig, LIST_OF_LIST& coef_sig_init,
		LIST& row_prior, LIST_OF_LIST& row_init, const int row_prior_type,
		LIST& col_prior, LIST_OF_LIST& col_init, const int col_prior_type,
		const Eigen::MatrixXi& seed_chain
	) {
		initData(y);
		// initForecaster(fit_record);
		using is_mcmc = std::integral_constant<bool, isUpdate>;
		if (is_mcmc::value) {
			// initMcmc(
			// 	param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
			// 	seed_chain
			// );
			for (int window = 0; window < num_horizon; ++window) {
				std::vector<Eigen::MatrixXd> response = marmatrix_to_vector(roll_mat[window], num_row, lag);
				std::vector<Eigen::SparseMatrix<double>> design = build_mar_design(roll_mat[window], num_window, lag);
				auto temp_mcmc = initialize_matmcmc(
					num_chains, num_iter - num_burn, design, response,
					param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type,
					col_prior, col_init, col_prior_type,
					seed_chain.row(window)
				);
				for (int i = 0; i < num_chains; ++i) {
					model[window][i] = std::move(temp_mcmc[i]);
				}
			}
			auto temp_forecaster = initialize_matmniwforecaster(num_chains, lag, step, roll_mat[0], num_window, fit_record, seed_forecast, nthreads);
			for (int i = 0; i < num_chains; ++i) {
				forecaster[0][i] = std::move(temp_forecaster[i]);
			}
		} else {
			for (int window = 0; window < num_horizon; ++window) {
				auto temp_forecaster = initialize_matmniwforecaster(num_chains, lag, step, roll_mat[window], num_window, fit_record, seed_forecast, nthreads);
				for (int i = 0; i < num_chains; ++i) {
					forecaster[window][i] = std::move(temp_forecaster[i]);
				}
			}
		}
	}

	virtual void initData(const Eigen::MatrixXd& y) = 0;

	// virtual std::vector<Eigen::SparseMatrix<double>> buildDesign(int window) = 0;

	// void initMcmc(
	// 	LIST& param_coef_sig, LIST_OF_LIST& coef_sig_init,
	// 	LIST& row_prior, LIST_OF_LIST& row_init, const int row_prior_type,
	// 	LIST& col_prior, LIST_OF_LIST& col_init, const int col_prior_type,
	// 	const Eigen::MatrixXi& seed_chain
	// ) {
	// 	for (int window = 0; window < num_horizon; ++window) {
	// 		std::vector<Eigen::MatrixXd> response = marmatrix_to_vector(roll_mat[window], num_row, lag);
	// 		std::vector<Eigen::SparseMatrix<double>> design = build_mar_design(roll_mat[window], num_window, lag);
	// 		auto temp_mcmc = initialize_matmcmc(
	// 			num_chains, num_iter - num_burn, design, response,
	// 			param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type,
	// 			col_prior, col_init, col_prior_type,
	// 			seed_chain.row(window)
	// 		);
	// 		for (int i = 0; i < num_chains; ++i) {
	// 			model[window][i] = std::move(temp_mcmc[i]);
	// 		}
	// 	}
	// }

	// void initForecaster(LIST& fit_record) {
	// 	using is_mcmc = std::integral_constant<bool, isUpdate>;
	// 	if (is_mcmc::value) {
	// 		auto temp_forecaster = initialize_matmniwforecaster(num_chains, lag, step, roll_mat[0], num_window, fit_record, seed_forecast, nthreads);
	// 		for (int i = 0; i < num_chains; ++i) {
	// 			forecaster[0][i] = std::move(temp_forecaster[i]);
	// 		}
	// 	} else {
	// 		for (int window = 0; window < num_horizon; ++window) {
	// 			auto temp_forecaster = initialize_matmniwforecaster(num_chains, lag, step, roll_mat[window], num_window, fit_record, seed_forecast, nthreads);
	// 			for (int i = 0; i < num_chains; ++i) {
	// 				forecaster[window][i] = std::move(temp_forecaster[i]);
	// 			}
	// 		}
	// 	}
	// }

	void updateForecaster(int window, int chain) override {
		auto* mcmc_mniw = dynamic_cast<McmcMatMniw*>(model[window][chain].get());
		MatMniwRecords mniw_record = mcmc_mniw->returnStructRecords(0, thin);
		forecaster[window][chain] = std::make_unique<MatMniwForecaster>(mniw_record, step, roll_mat[window], num_window, lag, static_cast<unsigned int>(seed_forecast[chain]));
	}
};

template <bool isUpdate = true>
class MatMniwRollForecastRun : public MatMniwOutForecastRun<isUpdate> {
public:
	MatMniwRollForecastRun(
		const Eigen::MatrixXd& y, int num_data, int lag,
		int num_chains, int num_iter, int num_burn, int thin, LIST& fit_record,
		LIST& param_coef_sig, LIST_OF_LIST& coef_sig_init,
		LIST& row_prior, LIST_OF_LIST& row_init, const int row_prior_type,
		LIST& col_prior, LIST_OF_LIST& col_init, const int col_prior_type,
		int step, const Eigen::MatrixXd& y_test,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads
	)
	: MatMniwOutForecastRun<isUpdate>(
			y, num_data, lag,
			param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
			step, y_test, seed_chain, seed_forecast, display_progress, nthreads
		) {
		initialize(
			y, fit_record,
			param_coef_sig, coef_sig_init,
			row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
			seed_chain
		);
	}
	virtual ~MatMniwRollForecastRun() = default;

protected:
	using MatMniwOutForecastRun<isUpdate>::num_window;
	using MatMniwOutForecastRun<isUpdate>::num_row;
	using MatMniwOutForecastRun<isUpdate>::num_col;
	using MatMniwOutForecastRun<isUpdate>::num_test;
	using MatMniwOutForecastRun<isUpdate>::num_horizon;
	using MatMniwOutForecastRun<isUpdate>::roll_mat;
	using MatMniwOutForecastRun<isUpdate>::y_test;
	using MatMniwOutForecastRun<isUpdate>::initialize;

	void initData(const Eigen::MatrixXd y) override {
		Eigen::MatrixXd tot_mat((num_window + num_test) * num_row, num_col);
		tot_mat << y,
							 y_test;
		for (int i = 0; i < num_horizon; ++i) {
			roll_mat[i] = tot_mat.middleRows(i * num_row, num_window);
			// roll_y0[i] = roll_mat[i].bottomRows(num_window - num_row * lag);
		}
	}

	// std::vector<Eigen::SparseMatrix<double>> buildDesign(int window) override {
	// 	return build_mar_design(roll_mat[window], num_window, lag);
	// }
};

template <bool isUpdate = true>
class MatMniwExpandForecastRun : public MatMniwOutForecastRun<isUpdate> {
public:
	MatMniwExpandForecastRun(
		const Eigen::MatrixXd& y, int num_data, int lag,
		int num_chains, int num_iter, int num_burn, int thin, LIST& fit_record,
		LIST& param_coef_sig, LIST_OF_LIST& coef_sig_init,
		LIST& row_prior, LIST_OF_LIST& row_init, const int row_prior_type,
		LIST& col_prior, LIST_OF_LIST& col_init, const int col_prior_type,
		int step, const Eigen::MatrixXd& y_test,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads
	)
	: MatMniwOutForecastRun<isUpdate>(
			y, num_data, lag,
			param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
			step, y_test, seed_chain, seed_forecast, display_progress, nthreads
		) {
		initialize(
			y, fit_record,
			param_coef_sig, coef_sig_init,
			row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
			seed_chain
		);
	}
	virtual ~MatMniwExpandForecastRun() = default;

protected:
	using MatMniwOutForecastRun<isUpdate>::num_window;
	using MatMniwOutForecastRun<isUpdate>::num_row;
	using MatMniwOutForecastRun<isUpdate>::num_col;
	using MatMniwOutForecastRun<isUpdate>::num_test;
	using MatMniwOutForecastRun<isUpdate>::num_horizon;
	using MatMniwOutForecastRun<isUpdate>::roll_mat;
	using MatMniwOutForecastRun<isUpdate>::y_test;
	using MatMniwOutForecastRun<isUpdate>::initialize;

	void initData(const Eigen::MatrixXd y) override {
		Eigen::MatrixXd tot_mat((num_window + num_test) * num_row, num_col);
		tot_mat << y,
							 y_test;
		for (int i = 0; i < num_horizon; ++i) {
			roll_mat[i] = tot_mat.topRows(i * num_row + num_window);
		}
	}
};

} // namespace baymar

#endif // BAYMAR_BAYES_MNIW_FORECASTER_H