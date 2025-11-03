#ifndef BAYMAR_BAYES_MDFM_FORECASTER_H
#define BAYMAR_BAYES_MDFM_FORECASTER_H

#include "./mdfm.h"
#include "../../core/forecaster.h"
#include "../mniw/forecaster.h"

namespace baymar {

class MatDfmForecaster;
// class MatDfmVarForecaster;
class MatDfmForecastRun;
template <bool> class MatDfmOutForecastRun;
template <bool> class MatDfmRollForecastRun;
template <bool> class MatDfmExpandForecastRun;

class MatDfmForecaster : public bvhar::BayesForecaster<Eigen::MatrixXd, Eigen::MatrixXd> {
public:
	MatDfmForecaster(
		const MatMniwRecords& mniw_records,
		std::unique_ptr<MatFactorForecaster>& factor_forecaster,
		int step,
		// int nrow_factor, int ncol_factor,
		unsigned int seed
	)
	: bvhar::BayesForecaster<Eigen::MatrixXd, Eigen::MatrixXd>(step, Eigen::MatrixXd(), 1, mniw_records.row_coef_record.rows(), seed),
		mat_record(std::make_unique<MatMniwRecords>(mniw_records)),
		factor_updater(std::move(factor_forecaster)),
		// nrow_factor(nrow_factor), ncol_factor(ncol_factor),
		nrow_factor(factor_updater->get_nrow_factor()), ncol_factor(factor_updater->get_ncol_factor()),
		num_row(mniw_records.row_coef_record.cols() / nrow_factor),
		num_col(mniw_records.col_coef_record.cols() / ncol_factor),
		row_coef(Eigen::MatrixXd::Zero(nrow_factor, num_row)),
		row_sig_lower(Eigen::MatrixXd::Identity(num_row, num_row)),
		col_coef(Eigen::MatrixXd::Zero(ncol_factor, num_col)),
		col_sig_lower(Eigen::MatrixXd::Identity(num_col, num_col)),
		error_mat(Eigen::MatrixXd::Zero(num_row, num_col)) {
		BVHAR_DEBUG_LOG(debug_logger, "MatDfmForecaster Constructor: step={}, nrow_factor={}, ncol_factor={}", step, nrow_factor, ncol_factor);
		initLagged();
	}
	virtual ~MatDfmForecaster() = default;
	
	Eigen::MatrixXd getLastForecast() override {
		return this->doForecast().bottomRows(num_row);
	}

	Eigen::MatrixXd getLastForecast(const Eigen::MatrixXd& valid_vec) override {
		return this->doForecast(valid_vec).bottomRows(num_row);
	}

protected:
	std::unique_ptr<MatMniwRecords> mat_record;
	// std::unique_ptr<MatDfmRecords> mdfm_record;
	std::unique_ptr<MatFactorForecaster> factor_updater;
	int nrow_factor, ncol_factor, num_row, num_col;
	Eigen::MatrixXd row_coef, row_sig_lower, col_coef, col_sig_lower, error_mat;

	void initLagged() override {
		BVHAR_DEBUG_LOG(debug_logger, "initLagged() called");
		// last_pvec = build_dense_design(response, lag);
		point_forecast = Eigen::MatrixXd::Zero(num_row, num_col);
		pred_save = Eigen::MatrixXd::Zero(step * num_row, num_sim * num_col);
		// tmp_vec = last_pvec.block(num_row, num_col, num_row * (lag - 1), num_col * (lag - 1));
	}

	void initRecursion(const Eigen::MatrixXd& obs_vec) override {
		BVHAR_DEBUG_LOG(debug_logger, "initRecursion(obs_vec) called");
		// last_pvec = obs_vec;
		// point_forecast = obs_vec.topLeftCorner(num_row, num_col);
		// tmp_vec = obs_vec.bottomRightCorner(num_row * (lag - 1), num_col * (lag - 1));
	}

	void setRecursion() override {
		BVHAR_DEBUG_LOG(debug_logger, "setRecursion() called");
		// last_pvec.bottomRightCorner(num_row * (lag - 1), num_col * (lag - 1)) = tmp_vec;
		// last_pvec.topLeftCorner(num_row, num_col) = point_forecast;
	}

	void updatePred(const int h, const int i) override {
		BVHAR_DEBUG_LOG(debug_logger, "updatePred(h={}, i={}) called", h, i);
		// computeMean();
		point_forecast.setZero();
		updateVariance();
		factor_updater->appendForecast(point_forecast, h);
		pred_save.block(h * num_row, i * num_col, num_row, num_col) = point_forecast + error_mat;
	}

	void updateRecursion() override {
		BVHAR_DEBUG_LOG(debug_logger, "updateRecursion() called");
		// tmp_vec = last_pvec.topLeftCorner(num_row * (lag - 1), num_col * (lag - 1));
	}

	// void computeMean() {
	// 	BVHAR_DEBUG_LOG(debug_logger, "computeMean() called");
	// 	point_forecast = row_coef.transpose() * last_pvec * col_coef;
	// 	// point_forecast.setZero();
	// 	// for (int i = 0; i < lag; ++i) {
	// 	// 	point_forecast += row_coef.middleRows(i * num_row, num_row).transpose() * last_pvec.block(i * num_row, i * num_col, num_row, num_col) * col_coef.middleRows(i * num_col, num_col);
	// 	// }
	// }

	void updateParams(const int i) override {
		BVHAR_DEBUG_LOG(debug_logger, "updateParams(i={}) called", i);
		row_coef = bvhar::unvectorize(mat_record->row_coef_record.row(i).transpose(), num_row);
		col_coef = bvhar::unvectorize(mat_record->col_coef_record.row(i).transpose(), num_col);
		factor_updater->updateCoefmat(
			mat_record->row_coef_record.row(i).transpose(),
			mat_record->col_coef_record.row(i).transpose(),
			0, 0
		);
		factor_updater->updateVarCoef(i, rng);
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

	void updateLpl(int h, const Eigen::MatrixXd& valid_vec) override {
		BVHAR_DEBUG_LOG(debug_logger, "updateLpl(h={}, valid_vec) called", h);
		lpl[h] -= col_sig_lower.transpose().triangularView<Eigen::Upper>().solve<Eigen::OnTheRight>(
			row_sig_lower.triangularView<Eigen::Lower>().solve(valid_vec - point_forecast)
		).squaredNorm() / 2 + num_row * num_col * log(2 * M_PI) / 2 + num_col * row_sig_lower.diagonal().array().log().sum() + num_row * col_sig_lower.diagonal().array().log().sum();
	}

	Eigen::MatrixXd getDesign() override {
		BVHAR_DEBUG_LOG(debug_logger, "getDesign() called");
		return Eigen::MatrixXd();
	}

	void forecastIn(const int i, const Eigen::MatrixXd& design) override {
		BVHAR_DEBUG_LOG(debug_logger, "forecastIn(i={}, design) called", i);
		for (int h = 0; h < step; ++h) {
			point_forecast.setZero();
			updateVariance();
			factor_updater->appendForecast(point_forecast, 0);
			pred_save.block(h * num_row, i * num_col, num_row, num_col) = point_forecast + error_mat;
		}
	}
};

// class MatDfmVarForecaster : public MatDfmForecaster {
// public:
// 	MatDfmVarForecaster(
// 		const MatMniwRecords& mniw_records,
// 		std::unique_ptr<MatFactorVarForecaster>& factor_forecaster,
// 		int step, unsigned int seed
// 	)
// 	: MatDfmForecaster(mniw_records, step, factor_forecaster->get_nrow_factor(), factor_forecaster->get_ncol_factor(), seed) {
// 		factor_updater = std::move(factor_forecaster);
// 	}
// 	virtual ~MatDfmVarForecaster() = default;
// };

inline std::vector<std::unique_ptr<MatDfmForecaster>> initialize_matdfmforecaster(
	int num_chains, int step, int nrow_factor, int ncol_factor, int factor_lag,
	BVHAR_LIST& fit_record, Eigen::Ref<const Eigen::VectorXi> seed_chain, int nthreads,
	BVHAR_OPTIONAL<bool> factor_insample = BVHAR_NULLOPT
) {
	BVHAR_STRING a_name = "A_record";
	BVHAR_STRING sigr_name = "SigmaR_record";
	BVHAR_STRING b_name = "B_record";
	BVHAR_STRING sigc_name = "SigmaC_record";
	BVHAR_STRING f_name = "F_record";
	BVHAR_STRING rho_name = "Rho_record";
	BVHAR_STRING prec_name = "Lambda_record";
	std::vector<std::unique_ptr<MatDfmForecaster>> forecaster(num_chains);
	for (int i = 0; i < num_chains; ++i) {
		std::unique_ptr<MatMniwRecords> mat_record;
		std::unique_ptr<MatDfmRecords> mdfm_record;
		std::unique_ptr<MatFactorForecaster> factor_updater;
		initialize_matmniw_record(mat_record, i, fit_record, a_name, sigr_name, b_name, sigc_name);
		factor_updater = initialize_matfactorforecaster(
			step, fit_record, i,
			mat_record->row_coef_record.cols() / nrow_factor,
			mat_record->col_coef_record.cols() / ncol_factor,
			nrow_factor, ncol_factor, factor_lag, factor_insample
		);
		// forecaster[i] = std::make_unique<MatDfmVarForecaster>(*mat_record, factor_updater, step, static_cast<unsigned int>(seed_chain[i]));
		forecaster[i] = std::make_unique<MatDfmForecaster>(*mat_record, factor_updater, step, static_cast<unsigned int>(seed_chain[i]));
	}
	return forecaster;
}

class MatDfmForecastRun : public bvhar::McmcForecastRun<Eigen::MatrixXd, Eigen::MatrixXd> {
public:
	MatDfmForecastRun(
		int num_chains, int step, int nrow_factor, int ncol_factor, int factor_lag,
		BVHAR_LIST& fit_record, Eigen::Ref<const Eigen::VectorXi> seed_chain, int nthreads,
		BVHAR_OPTIONAL<bool> factor_insample = BVHAR_NULLOPT
	)
	: bvhar::McmcForecastRun<Eigen::MatrixXd, Eigen::MatrixXd>(num_chains, 1, step, nthreads) {
		BVHAR_DEBUG_LOG(
			debug_logger,
			"MatDfmForecastRun Constructor: num_chains={}, step={}, nrow_factor={}, ncol_factor={} factor_lag={}, nthreads={}",
			num_chains, step, nrow_factor, ncol_factor, factor_lag, nthreads
		);
		auto temp_forecaster = initialize_matdfmforecaster(
			num_chains, step,
			nrow_factor, ncol_factor, factor_lag,
			fit_record, seed_chain, nthreads,
			factor_insample
		);
		for (int i = 0; i < num_chains; ++i) {
			forecaster[i] = std::move(temp_forecaster[i]);
		}
	}
	virtual ~MatDfmForecastRun() = default;
};

template <bool isUpdate = true>
class MatDfmOutForecastRun : public bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate> {
public:
	MatDfmOutForecastRun(
		const Eigen::MatrixXd& y, int num_data,
		int num_chains, int num_iter, int num_burn, int thin, BVHAR_LIST& fit_record,
		BVHAR_LIST& param_coef_sig, BVHAR_LIST_OF_LIST& coef_sig_init,
		BVHAR_LIST& row_prior, BVHAR_LIST_OF_LIST& row_init, const int row_prior_type,
		BVHAR_LIST& col_prior, BVHAR_LIST_OF_LIST& col_init, const int col_prior_type,
		int nrow_factor, int ncol_factor, int factor_lag,
		int step, const Eigen::MatrixXd& y_test, bool get_lpl, bool use_fit,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads
	)
	: bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>(
			num_data, 1, num_chains, num_iter, num_burn, thin, step, y_test, y_test.rows(), get_lpl, use_fit,
			seed_chain, seed_forecast, display_progress, nthreads
		),
		num_row(y.rows() / num_data), num_col(y.cols()),
		nrow_factor(nrow_factor), ncol_factor(ncol_factor), factor_lag(factor_lag) {
		BVHAR_DEBUG_LOG(debug_logger, "MatDfmOutForecastRun Constructor: num_data={}, row_prior_type={}, col_prior_type={}", num_data, row_prior_type, col_prior_type);
		BVHAR_STRING factor_model_nm = BVHAR_CAST<BVHAR_STRING>(param_coef_sig["factor_type"]);
		if (factor_model_nm == "wn") {
			factor_type = 1;
		} else if (factor_model_nm == "var") {
			factor_type = 2;
		} else if (factor_model_nm == "mar") {
			factor_type = 3;
		}
		num_test /= num_row;
		num_horizon = num_test - step + 1;
		roll_mat.resize(num_horizon);
		model.resize(num_horizon);
		out_forecast.resize(num_horizon);
		lpl_record.resize(num_horizon, num_chains);
		lpl_record = Eigen::MatrixXd::Zero(num_horizon, num_chains);
	}
	virtual ~MatDfmOutForecastRun() = default;

protected:
	int num_row, num_col, nrow_factor, ncol_factor, factor_lag, factor_type;
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
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::use_fit;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::display_progress;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::seed_forecast;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::roll_mat;
	// using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::roll_y0;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::y_test;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::model;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::forecaster;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::out_forecast;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::lpl_record;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::debug_logger;

	Eigen::MatrixXd getValid() override {
		BVHAR_DEBUG_LOG(debug_logger, "getValid() called");
		return y_test.bottomRows(num_row);
	}

	void initForecaster(BVHAR_LIST& fit_record) {
		BVHAR_DEBUG_LOG(debug_logger, "initForecaster(fit_record) called");
		using is_mcmc = std::integral_constant<bool, isUpdate>;
		if (is_mcmc::value) {
			auto temp_forecaster = initialize_matdfmforecaster(
				num_chains, step, nrow_factor, ncol_factor, factor_lag,
				fit_record, seed_forecast, nthreads
			);
			for (int i = 0; i < num_chains; ++i) {
				forecaster[0][i] = std::move(temp_forecaster[i]);
			}
		} else {
			for (int window = 0; window < num_horizon; ++window) {
				auto temp_forecaster = initialize_matdfmforecaster(
					num_chains, step, nrow_factor, ncol_factor, factor_lag,
					fit_record, seed_forecast, nthreads
				);
				for (int i = 0; i < num_chains; ++i) {
					forecaster[window][i] = std::move(temp_forecaster[i]);
				}
			}
		}
	}

	void initialize(
		const Eigen::MatrixXd& y, BVHAR_LIST& fit_record,
		BVHAR_LIST& param_coef_sig, BVHAR_LIST_OF_LIST& coef_sig_init,
		BVHAR_LIST& row_prior, BVHAR_LIST_OF_LIST& row_init, const int row_prior_type,
		BVHAR_LIST& col_prior, BVHAR_LIST_OF_LIST& col_init, const int col_prior_type,
		int nrow_factor, int ncol_factor, int factor_lag,
		const Eigen::MatrixXi& seed_chain
	) {
		BVHAR_DEBUG_LOG(debug_logger, "initialize(...) called");
		initData(y);
		if (use_fit) {
			initForecaster(fit_record);
		}
		for (int window = 0; window < num_horizon; ++window) {
			if (use_fit && window == 0) {
				continue;
			}
			std::vector<Eigen::MatrixXd> y_data = marmatrix_to_vector(roll_mat[window], num_row);
			std::vector<Eigen::MatrixXd> response = build_mar_response(y_data, lag);
			auto temp_mcmc = initialize_matdfm(
				num_chains, num_iter - num_burn, response, factor_lag,
				param_coef_sig, coef_sig_init,
				row_prior, row_init, row_prior_type,
				col_prior, col_init, col_prior_type,
				seed_chain.row(window)
			);
			for (int i = 0; i < num_chains; ++i) {
				model[window][i] = std::move(temp_mcmc[i]);
			}
		}
		// using is_mcmc = std::integral_constant<bool, isUpdate>;
		// if (is_mcmc::value) {
		// 	for (int window = 0; window < num_horizon; ++window) {
		// 		std::vector<Eigen::MatrixXd> y_data = marmatrix_to_vector(roll_mat[window], num_row);
		// 		std::vector<Eigen::MatrixXd> response = build_mar_response(y_data, lag);
		// 		auto temp_mcmc = initialize_matdfm(
		// 			num_chains, num_iter - num_burn, response, factor_lag,
		// 			param_coef_sig, coef_sig_init,
		// 			row_prior, row_init, row_prior_type,
		// 			col_prior, col_init, col_prior_type,
		// 			seed_chain.row(window)
		// 		);
		// 		auto temp_forecaster = initialize_matdfmforecaster(
		// 			num_chains, step, nrow_factor, ncol_factor, factor_lag,
		// 			fit_record, seed_forecast, nthreads
		// 		);
		// 		for (int i = 0; i < num_chains; ++i) {
		// 			model[window][i] = std::move(temp_mcmc[i]);
		// 			forecaster[window][i] = std::move(temp_forecaster[i]);
		// 		}
		// 	}
		// } else {
		// 	auto temp_forecaster = initialize_matdfmforecaster(
		// 		num_chains, step, nrow_factor, ncol_factor, factor_lag,
		// 		fit_record, seed_forecast, nthreads
		// 	);
		// 	for (int i = 0; i < num_chains; ++i) {
		// 		forecaster[0][i] = std::move(temp_forecaster[i]);
		// 	}
		// }
	}

	virtual void initData(const Eigen::MatrixXd& y) = 0;

	void updateForecaster(int window, int chain) override {
		BVHAR_DEBUG_LOG(debug_logger, "updateForecaster(window={}, chain={}) called", window, chain);
		auto* mcmc_mdfm = dynamic_cast<McmcMatDfm*>(model[window][chain].get());
		MatMniwRecords mniw_record = mcmc_mdfm->returnMniwRecords(0, thin);
		std::unique_ptr<MatFactorForecaster> factor_updater;
		// if (factor_lag != 0) {
		// 	MatDfmVarRecords mdfm_var_record = mcmc_mdfm->returnStructRecords<MatDfmVarRecords>(0, thin);
		// 	factor_updater = std::make_unique<MatFactorVarForecaster>(mdfm_var_record, step, factor_lag, num_row, num_col, nrow_factor, ncol_factor);
		// } else {
		// 	factor_updater = std::make_unique<MatFactorForecaster>(step, 0, num_row, num_col, nrow_factor, ncol_factor);
		// }
		if (factor_type == 1) {
			factor_updater = std::make_unique<MatFactorForecaster>(step, 0, num_row, num_col, nrow_factor, ncol_factor);
		} else if (factor_type == 2) {
			MatDfmVarRecords mdfm_var_record = mcmc_mdfm->returnStructRecords<MatDfmVarRecords>(0, thin);
			factor_updater = std::make_unique<MatFactorVarForecaster>(mdfm_var_record, step, factor_lag, num_row, num_col, nrow_factor, ncol_factor);
		} else if (factor_type == 3) {
			MatDfmMarRecords mdfm_mar_record = mcmc_mdfm->returnStructRecords<MatDfmMarRecords>(0, thin);
			factor_updater = std::make_unique<MatFactorMarForecaster>(mdfm_mar_record, step, factor_lag, num_row, num_col, nrow_factor, ncol_factor);
		} else {
			BVHAR_STOP("Wrong factor type");
		}
		// forecaster[window][chain] = std::make_unique<MatDfmVarForecaster>(mniw_record, factor_updater, step, static_cast<unsigned int>(seed_forecast[chain]));
		forecaster[window][chain] = std::make_unique<MatDfmForecaster>(mniw_record, factor_updater, step, static_cast<unsigned int>(seed_forecast[chain]));
	}
};

template <bool isUpdate = true>
class MatDfmRollForecastRun : public MatDfmOutForecastRun<isUpdate> {
public:
	MatDfmRollForecastRun(
		const Eigen::MatrixXd& y, int num_data,
		int num_chains, int num_iter, int num_burn, int thin, BVHAR_LIST& fit_record,
		BVHAR_LIST& param_coef_sig, BVHAR_LIST_OF_LIST& coef_sig_init,
		BVHAR_LIST& row_prior, BVHAR_LIST_OF_LIST& row_init, const int row_prior_type,
		BVHAR_LIST& col_prior, BVHAR_LIST_OF_LIST& col_init, const int col_prior_type,
		int nrow_factor, int ncol_factor, int factor_lag,
		int step, const Eigen::MatrixXd& y_test, bool get_lpl, bool use_fit,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads
	)
	: MatDfmOutForecastRun<isUpdate>(
			y, num_data, num_chains, num_iter, num_burn, thin, fit_record,
			param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
			nrow_factor, ncol_factor, factor_lag,
			step, y_test, get_lpl, use_fit,
			seed_chain, seed_forecast, display_progress, nthreads
		) {
		BVHAR_DEBUG_LOG(debug_logger, "MatDfmOutForecastRun constructor");
		initialize(
			y, fit_record,
			param_coef_sig, coef_sig_init,
			row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
			nrow_factor, ncol_factor, factor_lag, seed_chain
		);
	}
	virtual ~MatDfmRollForecastRun() = default;

protected:
	using MatDfmOutForecastRun<isUpdate>::num_window;
	using MatDfmOutForecastRun<isUpdate>::num_row;
	using MatDfmOutForecastRun<isUpdate>::num_col;
	using MatDfmOutForecastRun<isUpdate>::num_test;
	using MatDfmOutForecastRun<isUpdate>::num_horizon;
	using MatDfmOutForecastRun<isUpdate>::step;
	using MatDfmOutForecastRun<isUpdate>::roll_mat;
	using MatDfmOutForecastRun<isUpdate>::y_test;
	using MatDfmOutForecastRun<isUpdate>::initialize;
	using MatDfmOutForecastRun<isUpdate>::roll_exogen_mat;
	using MatDfmOutForecastRun<isUpdate>::roll_exogen;
	using MatDfmOutForecastRun<isUpdate>::lag_exogen;
	using MatDfmOutForecastRun<isUpdate>::debug_logger;

	void initData(const Eigen::MatrixXd& y) override {
		BVHAR_DEBUG_LOG(debug_logger, "initData(y, ...) called");
		Eigen::MatrixXd tot_mat((num_window + num_test) * num_row, num_col);
		tot_mat << y,
							 y_test;
		for (int i = 0; i < num_horizon; ++i) {
			// roll_mat[i] = tot_mat.middleRows(i * num_row, num_window);
			roll_mat[i] = tot_mat.middleRows(i * num_row, num_window * num_row);
			// roll_y0[i] = roll_mat[i].bottomRows(num_window - num_row * lag);
		}
	}
};

template <bool isUpdate = true>
class MatDfmExpandForecastRun : public MatDfmOutForecastRun<isUpdate> {
public:
	MatDfmExpandForecastRun(
		const Eigen::MatrixXd& y, int num_data,
		int num_chains, int num_iter, int num_burn, int thin, BVHAR_LIST& fit_record,
		BVHAR_LIST& param_coef_sig, BVHAR_LIST_OF_LIST& coef_sig_init,
		BVHAR_LIST& row_prior, BVHAR_LIST_OF_LIST& row_init, const int row_prior_type,
		BVHAR_LIST& col_prior, BVHAR_LIST_OF_LIST& col_init, const int col_prior_type,
		int nrow_factor, int ncol_factor, int factor_lag,
		int step, const Eigen::MatrixXd& y_test, bool get_lpl, bool use_fit,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads
	)
	: MatDfmOutForecastRun<isUpdate>(
			y, num_data, num_chains, num_iter, num_burn, thin, fit_record,
			param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
			nrow_factor, ncol_factor, factor_lag,
			step, y_test, get_lpl, use_fit,
			seed_chain, seed_forecast, display_progress, nthreads
		) {
		BVHAR_DEBUG_LOG(debug_logger, "MatDfmOutForecastRun constructor");
		initialize(
			y, fit_record,
			param_coef_sig, coef_sig_init,
			row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
			nrow_factor, ncol_factor, factor_lag, seed_chain
		);
	}
	virtual ~MatDfmExpandForecastRun() = default;

protected:
	using MatDfmOutForecastRun<isUpdate>::num_window;
	using MatDfmOutForecastRun<isUpdate>::num_row;
	using MatDfmOutForecastRun<isUpdate>::num_col;
	using MatDfmOutForecastRun<isUpdate>::num_test;
	using MatDfmOutForecastRun<isUpdate>::num_horizon;
	using MatDfmOutForecastRun<isUpdate>::step;
	using MatDfmOutForecastRun<isUpdate>::roll_mat;
	using MatDfmOutForecastRun<isUpdate>::y_test;
	using MatDfmOutForecastRun<isUpdate>::initialize;
	using MatDfmOutForecastRun<isUpdate>::roll_exogen_mat;
	using MatDfmOutForecastRun<isUpdate>::roll_exogen;
	using MatDfmOutForecastRun<isUpdate>::lag_exogen;
	using MatDfmOutForecastRun<isUpdate>::debug_logger;

	void initData(const Eigen::MatrixXd& y) override {
		BVHAR_DEBUG_LOG(debug_logger, "initData(y, ...) called");
		Eigen::MatrixXd tot_mat((num_window + num_test) * num_row, num_col);
		tot_mat << y,
							 y_test;
		for (int i = 0; i < num_horizon; ++i) {
			roll_mat[i] = tot_mat.topRows((num_window + i) * num_row);
			BVHAR_DEBUG_LOG(debug_logger, "roll_mat[{}]: {} x {}", i, roll_mat[i].rows(), roll_mat[i].cols());
		}
	}
};

template <template <bool> class BaseOutForecast = MatDfmRollForecastRun>
inline std::unique_ptr<bvhar::McmcOutforecastInterface> initialize_matdfmoutforecaster(
	const Eigen::MatrixXd& y, int num_data,
	int num_chains, int num_iter, int num_burn, int thin, BVHAR_LIST& fit_record,
	bool run_mcmc,
	BVHAR_LIST& param_coef_sig, BVHAR_LIST_OF_LIST& coef_sig_init,
	BVHAR_LIST& row_prior, BVHAR_LIST_OF_LIST& row_init, const int row_prior_type,
	BVHAR_LIST& col_prior, BVHAR_LIST_OF_LIST& col_init, const int col_prior_type,
	int nrow_factor, int ncol_factor, int factor_lag,
	int step, const Eigen::MatrixXd& y_test, bool get_lpl, bool use_fit,
	const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads
) {
	if (run_mcmc) {
		return std::make_unique<BaseOutForecast<true>>(
			y, num_data, num_chains, num_iter, num_burn, thin, fit_record,
			param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
			nrow_factor, ncol_factor, factor_lag,
			step, y_test, get_lpl, use_fit,
			seed_chain, seed_forecast, display_progress, nthreads
		);
	}
	return std::make_unique<BaseOutForecast<false>>(
		y, num_data, num_chains, num_iter, num_burn, thin, fit_record,
		param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
		nrow_factor, ncol_factor, factor_lag,
		step, y_test, get_lpl, use_fit,
		seed_chain, seed_forecast, display_progress, nthreads
	);
}

} // namespace baymar

#endif // BAYMAR_BAYES_MDFM_FORECASTER_H