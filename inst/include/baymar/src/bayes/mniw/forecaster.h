#ifndef BAYMAR_BAYES_MNIW_FORECASTER_H
#define BAYMAR_BAYES_MNIW_FORECASTER_H

#include "./mniw.h"
#include "../../math/design.h"

namespace baymar {

class MatMniwExogenForecaster;
class MatMniwForecaster;
class MatMniwForecastRun;
template <bool isUpdate> class MatMniwOutForecastRun;
template <bool isUpdate> class MatMniwRollForecastRun;
template <bool isUpdate> class MatMniwExpandForecastRun;

class MatMniwExogenForecaster : public bvhar::ExogenForecaster<Eigen::MatrixXd, Eigen::MatrixXd> {
public:
	MatMniwExogenForecaster(int lag, const Eigen::MatrixXd& exogen, int num_exogen, int num_row, int num_col)
	: bvhar::ExogenForecaster<Eigen::MatrixXd, Eigen::MatrixXd>(lag, exogen),
		nrow_exogen(exogen.rows() / num_exogen), ncol_exogen(exogen.cols()),
		nrow_row_exogen((lag + 1) * nrow_exogen), nrow_col_exogen((lag + 1) * ncol_exogen),
		num_row(num_row), num_col(num_col),
		row_coef(nrow_row_exogen, num_row), col_coef(nrow_col_exogen, num_col) {
		BVHAR_DEBUG_LOG(debug_logger, "Constructor: num_exogen={}, num_row={}, num_col={}", num_exogen, num_row, num_col);
		// last_pvec = Eigen::MatrixXd::Zero((lag + 1) * nrow_exogen, (lag + 1) * ncol_exogen);
		// exogen: rbind(X_{T - s}, ..., X_{T + h})
		// last_pvec = x_(T + h), ..., x_(T + h - s)
		last_pvec = Eigen::MatrixXd::Zero(nrow_exogen, ncol_exogen);
	}
	virtual ~MatMniwExogenForecaster() = default;
	
	void appendForecast(Eigen::MatrixXd& point_forecast, const int h) override {
		BVHAR_DEBUG_LOG(debug_logger, "appendForecast(point_forecast, h) called");
		for (int i = 0; i < lag + 1; ++i) {
			last_pvec = exogen.middleRows((lag + h - i) * nrow_exogen, nrow_exogen); // x_(T + h - i)
			point_forecast += row_coef.middleRows(i * nrow_exogen, nrow_exogen).transpose() * last_pvec * col_coef.middleRows(i * ncol_exogen, ncol_exogen);
		}
	}

	void updateCoefmat(const Eigen::VectorXd& row_coef_record, const Eigen::VectorXd& col_coef_record, int nrow_row_coef, int nrow_col_coef) {
		BVHAR_DEBUG_LOG(debug_logger, "updateCoefmat() called");
		row_coef = bvhar::unvectorize(row_coef_record.segment(nrow_row_coef * num_row, nrow_row_exogen * num_row).transpose(), num_row);
		col_coef = bvhar::unvectorize(col_coef_record.segment(nrow_col_coef * num_col, nrow_col_exogen * num_col).transpose(), num_col);
	}

private:
	int nrow_exogen, ncol_exogen, nrow_row_exogen, nrow_col_exogen, num_row, num_col;
	Eigen::MatrixXd row_coef, col_coef;
};

class MatMniwForecaster : public bvhar::BayesForecaster<Eigen::MatrixXd, Eigen::MatrixXd> {
public:
	MatMniwForecaster(
		const MatMniwRecords& records, int step, const Eigen::MatrixXd& y, int num_data, int lag, unsigned int seed,
		Optional<std::unique_ptr<MatMniwExogenForecaster>> exogen_forecaster = NULLOPT
	)
	: bvhar::BayesForecaster<Eigen::MatrixXd, Eigen::MatrixXd>(step, y, lag, records.row_coef_record.rows(), seed),
		mat_record(std::make_unique<MatMniwRecords>(records)),
		num_row(y.rows() / num_data), num_col(y.cols()), nrow_row_coef(num_row * lag), nrow_col_coef(num_col * lag),
		row_coef(Eigen::MatrixXd::Zero(nrow_row_coef, num_row)),
		row_sig_lower(Eigen::MatrixXd::Identity(num_row, num_row)),
		col_coef(Eigen::MatrixXd::Zero(nrow_col_coef, num_col)),
		col_sig_lower(Eigen::MatrixXd::Identity(num_col, num_col)),
		error_mat(Eigen::MatrixXd::Zero(num_row, num_col)) {
		BVHAR_DEBUG_LOG(debug_logger, "MatMniwForecaster Constructor: step={}, num_data={}, lag={}", step, num_data, lag);
		initLagged();
		if (exogen_forecaster) {
			exogen_updater = std::move(*exogen_forecaster);
		}
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
	std::unique_ptr<MatMniwExogenForecaster> exogen_updater;
	int num_row, num_col, nrow_row_coef, nrow_col_coef;
	Eigen::MatrixXd row_coef, row_sig_lower, col_coef, col_sig_lower, error_mat;

	void initLagged() override {
		BVHAR_DEBUG_LOG(debug_logger, "initLagged() called");
		last_pvec = build_dense_design(response, lag);
		point_forecast = Eigen::MatrixXd::Zero(num_row, num_col);
		pred_save = Eigen::MatrixXd::Zero(step * num_row, num_sim * num_col);
		tmp_vec = last_pvec.block(num_row, num_col, num_row * (lag - 1), num_col * (lag - 1));
	}

	void initRecursion(const Eigen::MatrixXd& obs_vec) override {
		BVHAR_DEBUG_LOG(debug_logger, "initRecursion(obs_vec) called");
		last_pvec = obs_vec;
		point_forecast = obs_vec.topLeftCorner(num_row, num_col);
		tmp_vec = obs_vec.bottomRightCorner(num_row * (lag - 1), num_col * (lag - 1));
	}

	void setRecursion() override {
		BVHAR_DEBUG_LOG(debug_logger, "setRecursion() called");
		last_pvec.bottomRightCorner(num_row * (lag - 1), num_col * (lag - 1)) = tmp_vec;
		last_pvec.topLeftCorner(num_row, num_col) = point_forecast;
	}

	void updatePred(const int h, const int i) override {
		BVHAR_DEBUG_LOG(debug_logger, "updatePred(h={}, i={}) called", h, i);
		computeMean();
		updateVariance();
		// point_forecast += error_mat;
		if (exogen_updater) {
			exogen_updater->appendForecast(point_forecast, h);
		}
		pred_save.block(h * num_row, i * num_col, num_row, num_col) = point_forecast + error_mat;
	}

	void updateRecursion() override {
		BVHAR_DEBUG_LOG(debug_logger, "updateRecursion() called");
		tmp_vec = last_pvec.topLeftCorner(num_row * (lag - 1), num_col * (lag - 1));
	}

	void computeMean() {
		BVHAR_DEBUG_LOG(debug_logger, "computeMean() called");
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
		BVHAR_DEBUG_LOG(debug_logger, "updateParams(i={}) called", i);
		row_coef = bvhar::unvectorize(mat_record->row_coef_record.row(i).head(nrow_row_coef * num_row).transpose(), num_row);
		col_coef = bvhar::unvectorize(mat_record->col_coef_record.row(i).head(nrow_col_coef * num_col).transpose(), num_col);
		if (exogen_updater) {
			exogen_updater->updateCoefmat(mat_record->row_coef_record.row(i).transpose(), mat_record->col_coef_record.row(i).transpose(), nrow_row_coef, nrow_col_coef);
		}
		fill_lower(row_sig_lower, mat_record->row_sigma_record.row(i).transpose());
		fill_lower(col_sig_lower, mat_record->col_sigma_record.row(i).transpose());
	}

	void updateVariance() {
		BVHAR_DEBUG_LOG(debug_logger, "updateVariance() called");
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
	LIST& fit_record, Eigen::Ref<const Eigen::VectorXi> seed_chain, int nthreads,
	Optional<Eigen::MatrixXd> exogen = NULLOPT, Optional<int> exogen_lag = NULLOPT
) {
	// PY_LIST row_coef_record = fit_record["A_record"];
	// PY_LIST row_sigma_record = fit_record["SigmaR_record"];
	// PY_LIST col_coef_record = fit_record["B_record"];
	// PY_LIST col_sigma_record = fit_record["SigmaC_record"];
	STRING a_name = "A_record";
	STRING sigr_name = "SigmaR_record";
	STRING b_name = "B_record";
	STRING sigc_name = "SigmaC_record";
	std::vector<std::unique_ptr<MatMniwForecaster>> forecaster(num_chains);
	for (int i = 0; i < num_chains; ++i) {
		std::unique_ptr<MatMniwRecords> mat_record;
		Optional<std::unique_ptr<MatMniwExogenForecaster>> exogen_updater = NULLOPT;
		if (exogen) {
			STRING c_name = "C_record";
			STRING d_name = "D_record";
			exogen_updater = std::make_unique<MatMniwExogenForecaster>(*exogen_lag, *exogen, *exogen_lag + step, y.rows() / num_data, y.cols());
			initialize_matmniw_record(mat_record, i, fit_record, a_name, sigr_name, b_name, sigc_name, c_name, d_name);
		} else {
			initialize_matmniw_record(mat_record, i, fit_record, a_name, sigr_name, b_name, sigc_name);
		}
		forecaster[i] = std::make_unique<MatMniwForecaster>(
			*mat_record, step, y, num_data, lag, static_cast<unsigned int>(seed_chain[i]),
			std::move(exogen_updater)
		);
	}
	return forecaster;
}

class MatMniwForecastRun : public bvhar::McmcForecastRun<Eigen::MatrixXd, Eigen::MatrixXd> {
public:
	MatMniwForecastRun(
		int num_chains, int lag, int step, const Eigen::MatrixXd& y, int num_data,
		LIST& fit_record, const Eigen::VectorXi& seed_chain, int nthreads,
		Optional<Eigen::MatrixXd> exogen = NULLOPT, Optional<int> exogen_lag = NULLOPT
	)
	: bvhar::McmcForecastRun<Eigen::MatrixXd, Eigen::MatrixXd>(num_chains, lag, step, nthreads) {
		BVHAR_DEBUG_LOG(
			debug_logger,
			"MatMniwForecastRun Constructor: num_chains={}, lag={}, step={}, num_data={}, nthreads={}",
			num_chains, lag, step, num_data, nthreads
		);
		auto temp_forecaster = initialize_matmniwforecaster(
			num_chains, lag, step, y, num_data, fit_record, seed_chain, nthreads,
			exogen, exogen_lag
		);
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
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads,
		Optional<LIST> row_exogen_prior = NULLOPT, Optional<LIST_OF_LIST> row_exogen_init = NULLOPT, Optional<int> row_exogen_prior_type = NULLOPT,
		Optional<LIST> col_exogen_prior = NULLOPT, Optional<LIST_OF_LIST> col_exogen_init = NULLOPT, Optional<int> col_exogen_prior_type = NULLOPT,
		Optional<Eigen::MatrixXd> exogen = NULLOPT, Optional<int> exogen_lag = NULLOPT
	)
	: bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>(
			num_data, lag, num_chains, num_iter, num_burn, thin, step, y_test, false,
			seed_chain, seed_forecast, display_progress, nthreads,
			exogen_lag
		),
		num_row(y.rows() / num_data), num_col(y.cols()), nrow_row_coef(num_row * lag), nrow_col_coef(num_col * lag) {
		BVHAR_DEBUG_LOG(debug_logger, "MatMniwOutForecastRun Constructor: num_data={}, row_prior_type={}, col_prior_type={}", num_data, row_prior_type, col_prior_type);
		num_test /= num_row;
		num_horizon = num_test - step + 1;
		roll_mat.resize(num_horizon);
		model.resize(num_horizon);
		out_forecast.resize(num_horizon);
		lpl_record.resize(num_horizon, num_chains);
		lpl_record = Eigen::MatrixXd::Zero(num_horizon, num_chains);
		// for (int i = 0; i < num_horizon; ++i) {
		// 	model[i].resize(num_chains);
		// 	forecaster[i].resize(num_chains);
		// 	out_forecast[i].resize(num_chains);
		// }
	}
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
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::roll_exogen_mat;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::roll_exogen;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::lag_exogen;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::debug_logger;

	Eigen::MatrixXd getValid() override {
		BVHAR_DEBUG_LOG(debug_logger, "getValid() called");
		return y_test.bottomRows(num_row);
	}

	void initialize(
		const Eigen::MatrixXd& y, LIST& fit_record,
		LIST& param_coef_sig, LIST_OF_LIST& coef_sig_init,
		LIST& row_prior, LIST_OF_LIST& row_init, const int row_prior_type,
		LIST& col_prior, LIST_OF_LIST& col_init, const int col_prior_type,
		const Eigen::MatrixXi& seed_chain,
		Optional<LIST> row_exogen_prior = NULLOPT, Optional<LIST_OF_LIST> row_exogen_init = NULLOPT, Optional<int> row_exogen_prior_type = NULLOPT,
		Optional<LIST> col_exogen_prior = NULLOPT, Optional<LIST_OF_LIST> col_exogen_init = NULLOPT, Optional<int> col_exogen_prior_type = NULLOPT,
		Optional<Eigen::MatrixXd> exogen = NULLOPT, Optional<int> exogen_lag = NULLOPT
	) {
		BVHAR_DEBUG_LOG(debug_logger, "initialize(...) called");
		initData(y, exogen);
		// initForecaster(fit_record);
		using is_mcmc = std::integral_constant<bool, isUpdate>;
		if (is_mcmc::value) {
			// initMcmc(
			// 	param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
			// 	seed_chain
			// );
			Optional<int> exogen_rows = NULLOPT;
 			Optional<int> exogen_cols = NULLOPT;
			for (int window = 0; window < num_horizon; ++window) {
				std::vector<Eigen::MatrixXd> y_data = marmatrix_to_vector(roll_mat[window], num_row);
				std::vector<Eigen::MatrixXd> response = build_mar_response(y_data, lag);
				Optional<std::vector<Eigen::MatrixXd>> exogen_data = NULLOPT;
				if (lag_exogen) {
					int nrow_exogen = exogen->rows() / (num_window + num_test);
					exogen_data = marmatrix_to_vector(*(roll_exogen_mat[window]), nrow_exogen);
				}
				std::vector<Eigen::SparseMatrix<double>> design = lag_exogen ? build_mar_design(y_data, *exogen_data, lag, *lag_exogen) : build_mar_design(y_data, lag);
				if (lag_exogen) {
					exogen_rows = (*lag_exogen + 1) * (*exogen_data)[0].rows();
					exogen_cols = (*lag_exogen + 1) * (*exogen_data)[0].cols();
				}
				auto temp_mcmc = initialize_matmcmc(
					num_chains, num_iter - num_burn, design, response,
					param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type,
					col_prior, col_init, col_prior_type,
					seed_chain.row(window),
					row_exogen_prior, row_exogen_init, row_exogen_prior_type, exogen_rows,
					col_exogen_prior, col_exogen_init, col_exogen_prior_type, exogen_cols
				);
				auto temp_forecaster = initialize_matmniwforecaster(
					num_chains, lag, step, roll_mat[window], num_window, fit_record, seed_forecast, nthreads,
					roll_exogen[window], lag_exogen
				);
				for (int i = 0; i < num_chains; ++i) {
					model[window][i] = std::move(temp_mcmc[i]);
					forecaster[window][i] = std::move(temp_forecaster[i]);
				}
			}
		} else {
			auto temp_forecaster = initialize_matmniwforecaster(
				num_chains, lag, step, roll_mat[0], num_window, fit_record, seed_forecast, nthreads,
				roll_exogen[0], lag_exogen
			);
			for (int i = 0; i < num_chains; ++i) {
				forecaster[0][i] = std::move(temp_forecaster[i]);
			}
		}
	}

	virtual void initData(const Eigen::MatrixXd& y, Optional<Eigen::MatrixXd> exogen = NULLOPT) = 0;

	void updateForecaster(int window, int chain) override {
		BVHAR_DEBUG_LOG(debug_logger, "updateForecaster(window={}, chain={}) called", window, chain);
		auto* mcmc_mniw = dynamic_cast<McmcMatMniw*>(model[window][chain].get());
		MatMniwRecords mniw_record = mcmc_mniw->returnStructRecords(0, thin);
		Optional<std::unique_ptr<MatMniwExogenForecaster>> exogen_updater = NULLOPT;
		if (lag_exogen) {
			exogen_updater = std::make_unique<MatMniwExogenForecaster>(*lag_exogen, *(roll_exogen[window]), *lag_exogen + step, num_row, num_col);
		}
		forecaster[window][chain] = std::make_unique<MatMniwForecaster>(
			mniw_record, step, roll_mat[window], num_window, lag, static_cast<unsigned int>(seed_forecast[chain]),
			std::move(exogen_updater)
		);
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
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads,
		Optional<LIST> row_exogen_prior = NULLOPT, Optional<LIST_OF_LIST> row_exogen_init = NULLOPT, Optional<int> row_exogen_prior_type = NULLOPT,
		Optional<LIST> col_exogen_prior = NULLOPT, Optional<LIST_OF_LIST> col_exogen_init = NULLOPT, Optional<int> col_exogen_prior_type = NULLOPT,
		Optional<Eigen::MatrixXd> exogen = NULLOPT, Optional<int> exogen_lag = NULLOPT
	)
	: MatMniwOutForecastRun<isUpdate>(
			y, num_data, lag,
			num_chains, num_iter, num_burn, thin, fit_record,
			param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
			step, y_test, seed_chain, seed_forecast, display_progress, nthreads,
			row_exogen_prior, row_exogen_init, row_exogen_prior_type,
			col_exogen_prior, col_exogen_init, col_exogen_prior_type,
			exogen, exogen_lag
		) {
		BVHAR_DEBUG_LOG(debug_logger, "MatMniwRollForecastRun constructor");
		initialize(
			y, fit_record,
			param_coef_sig, coef_sig_init,
			row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
			seed_chain,
			row_exogen_prior, row_exogen_init, row_exogen_prior_type,
			col_exogen_prior, col_exogen_init, col_exogen_prior_type,
			exogen, exogen_lag
		);
	}
	virtual ~MatMniwRollForecastRun() = default;

protected:
	using MatMniwOutForecastRun<isUpdate>::num_window;
	using MatMniwOutForecastRun<isUpdate>::num_row;
	using MatMniwOutForecastRun<isUpdate>::num_col;
	using MatMniwOutForecastRun<isUpdate>::num_test;
	using MatMniwOutForecastRun<isUpdate>::num_horizon;
	using MatMniwOutForecastRun<isUpdate>::step;
	using MatMniwOutForecastRun<isUpdate>::roll_mat;
	using MatMniwOutForecastRun<isUpdate>::y_test;
	using MatMniwOutForecastRun<isUpdate>::initialize;
	using MatMniwOutForecastRun<isUpdate>::roll_exogen_mat;
	using MatMniwOutForecastRun<isUpdate>::roll_exogen;
	using MatMniwOutForecastRun<isUpdate>::lag_exogen;
	using MatMniwOutForecastRun<isUpdate>::debug_logger;

	void initData(const Eigen::MatrixXd& y, Optional<Eigen::MatrixXd> exogen = NULLOPT) override {
		BVHAR_DEBUG_LOG(debug_logger, "initData(y, ...) called");
		Eigen::MatrixXd tot_mat((num_window + num_test) * num_row, num_col);
		tot_mat << y,
							 y_test;
		for (int i = 0; i < num_horizon; ++i) {
			// roll_mat[i] = tot_mat.middleRows(i * num_row, num_window);
			roll_mat[i] = tot_mat.middleRows(i * num_row, num_window * num_row);
			// roll_y0[i] = roll_mat[i].bottomRows(num_window - num_row * lag);
		}
		if (lag_exogen) {
			int nrow_exogen = exogen->rows() / (num_window + num_test);
			BVHAR_DEBUG_LOG(debug_logger, "nrow_exogen={}", nrow_exogen);
			for (int i = 0; i < num_horizon; ++i) {
				roll_exogen_mat[i] = (*exogen).middleRows(i * nrow_exogen, num_window * nrow_exogen);
				roll_exogen[i] = (*exogen).middleRows((num_window - *lag_exogen + i) * nrow_exogen, (*lag_exogen + step) * nrow_exogen);
			}
		}
	}
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
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads,
		Optional<LIST> row_exogen_prior = NULLOPT, Optional<LIST_OF_LIST> row_exogen_init = NULLOPT, Optional<int> row_exogen_prior_type = NULLOPT,
		Optional<LIST> col_exogen_prior = NULLOPT, Optional<LIST_OF_LIST> col_exogen_init = NULLOPT, Optional<int> col_exogen_prior_type = NULLOPT,
		Optional<Eigen::MatrixXd> exogen = NULLOPT, Optional<int> exogen_lag = NULLOPT
	)
	: MatMniwOutForecastRun<isUpdate>(
			y, num_data, lag,
			num_chains, num_iter, num_burn, thin, fit_record,
			param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
			step, y_test, seed_chain, seed_forecast, display_progress, nthreads,
			row_exogen_prior, row_exogen_init, row_exogen_prior_type,
			col_exogen_prior, col_exogen_init, col_exogen_prior_type,
			exogen, exogen_lag
		) {
		BVHAR_DEBUG_LOG(debug_logger, "MatMniwExpandForecastRun constructor");
		initialize(
			y, fit_record,
			param_coef_sig, coef_sig_init,
			row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
			seed_chain,
			row_exogen_prior, row_exogen_init, row_exogen_prior_type,
			col_exogen_prior, col_exogen_init, col_exogen_prior_type,
			exogen, exogen_lag
		);
	}
	virtual ~MatMniwExpandForecastRun() = default;

protected:
	using MatMniwOutForecastRun<isUpdate>::num_window;
	using MatMniwOutForecastRun<isUpdate>::num_row;
	using MatMniwOutForecastRun<isUpdate>::num_col;
	using MatMniwOutForecastRun<isUpdate>::num_test;
	using MatMniwOutForecastRun<isUpdate>::num_horizon;
	using MatMniwOutForecastRun<isUpdate>::step;
	using MatMniwOutForecastRun<isUpdate>::roll_mat;
	using MatMniwOutForecastRun<isUpdate>::y_test;
	using MatMniwOutForecastRun<isUpdate>::initialize;
	using MatMniwOutForecastRun<isUpdate>::roll_exogen_mat;
	using MatMniwOutForecastRun<isUpdate>::roll_exogen;
	using MatMniwOutForecastRun<isUpdate>::lag_exogen;
	using MatMniwOutForecastRun<isUpdate>::debug_logger;

	void initData(const Eigen::MatrixXd& y, Optional<Eigen::MatrixXd> exogen = NULLOPT) override {
		BVHAR_DEBUG_LOG(debug_logger, "initData(y, ...) called");
		Eigen::MatrixXd tot_mat((num_window + num_test) * num_row, num_col);
		tot_mat << y,
							 y_test;
		for (int i = 0; i < num_horizon; ++i) {
			roll_mat[i] = tot_mat.topRows((num_window + i) * num_row);
			BVHAR_DEBUG_LOG(debug_logger, "roll_mat[{}]: {} x {}", i, roll_mat[i].rows(), roll_mat[i].cols());
		}
		if (lag_exogen) {
			int nrow_exogen = exogen->rows() / (num_window + num_test);
			BVHAR_DEBUG_LOG(debug_logger, "nrow_exogen={}", nrow_exogen);
			for (int i = 0; i < num_horizon; ++i) {
				roll_exogen_mat[i] = (*exogen).topRows((num_window + i) * nrow_exogen);
				roll_exogen[i] = (*exogen).middleRows((num_window - *lag_exogen + i) * nrow_exogen, (*lag_exogen + step) * nrow_exogen);
			}
		}
	}
};

template <template <bool> class BaseOutForecast = MatMniwRollForecastRun>
inline std::unique_ptr<bvhar::McmcOutforecastInterface> initialize_matmniwoutforecaster(
	const Eigen::MatrixXd& y, int num_data, int lag,
	int num_chains, int num_iter, int num_burn, int thin, LIST& fit_record,
	bool run_mcmc,
	LIST& param_coef_sig, LIST_OF_LIST& coef_sig_init,
	LIST& row_prior, LIST_OF_LIST& row_init, const int row_prior_type,
	LIST& col_prior, LIST_OF_LIST& col_init, const int col_prior_type,
	int step, const Eigen::MatrixXd& y_test,
	const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads,
	Optional<LIST> row_exogen_prior = NULLOPT, Optional<LIST_OF_LIST> row_exogen_init = NULLOPT, Optional<int> row_exogen_prior_type = NULLOPT,
	Optional<LIST> col_exogen_prior = NULLOPT, Optional<LIST_OF_LIST> col_exogen_init = NULLOPT, Optional<int> col_exogen_prior_type = NULLOPT,
	Optional<Eigen::MatrixXd> exogen = NULLOPT, Optional<int> exogen_lag = NULLOPT
) {
	if (run_mcmc) {
		return std::make_unique<BaseOutForecast<true>>(
			y, num_data, lag, num_chains, num_iter, num_burn, thin, fit_record,
			param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
			step, y_test, seed_chain, seed_forecast, display_progress, nthreads,
			row_exogen_prior, row_exogen_init, row_exogen_prior_type,
			col_exogen_prior, col_exogen_init, col_exogen_prior_type,
			exogen, exogen_lag
		);
	}
	return std::make_unique<BaseOutForecast<false>>(
		y, num_data, lag, num_chains, num_iter, num_burn, thin, fit_record,
		param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
		step, y_test, seed_chain, seed_forecast, display_progress, nthreads,
		row_exogen_prior, row_exogen_init, row_exogen_prior_type,
		col_exogen_prior, col_exogen_init, col_exogen_prior_type,
		exogen, exogen_lag
	);
}

} // namespace baymar

#endif // BAYMAR_BAYES_MNIW_FORECASTER_H