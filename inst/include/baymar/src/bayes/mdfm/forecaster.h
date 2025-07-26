#ifndef BAYMAR_BAYES_MDFM_FORECASTER_H
#define BAYMAR_BAYES_MDFM_FORECASTER_H

#include "./mdfm.h"
#include "../../core/forecaster.h"
#include "../mniw/forecaster.h"

namespace baymar {

class MatDfmForecaster;
class MatDfmVarForecaster;
class MatDfmForecastRun;
template <bool> class MatMDfmOutForecastRun;
template <bool> class MatMDfmRollForecastRun;
template <bool> class MatMDfmExpandForecastRun;

class MatDfmForecaster : public bvhar::BayesForecaster<Eigen::MatrixXd, Eigen::MatrixXd> {
public:
	MatDfmForecaster(
		const MatMniwRecords& mniw_records,
		int step, int nrow_factor, int ncol_factor, unsigned int seed
	)
	: bvhar::BayesForecaster<Eigen::MatrixXd, Eigen::MatrixXd>(step, Eigen::MatrixXd(), 1, mniw_records.row_coef_record.rows(), seed),
		mat_record(std::make_unique<MatMniwRecords>(mniw_records)),
		nrow_factor(nrow_factor), ncol_factor(ncol_factor),
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

	void updateLpl(int h, const Eigen::MatrixXd& valid_vec) override {}
};

class MatDfmVarForecaster : public MatDfmForecaster {
public:
	MatDfmVarForecaster(
		const MatMniwRecords& mniw_records,
		std::unique_ptr<MatFactorVarForecaster>& factor_forecaster,
		int step, unsigned int seed
	)
	: MatDfmForecaster(mniw_records, step, factor_forecaster->get_nrow_factor(), factor_forecaster->get_ncol_factor(), seed) {
		factor_updater = std::move(factor_forecaster);
	}
	virtual ~MatDfmVarForecaster() = default;
};

inline std::vector<std::unique_ptr<MatDfmForecaster>> initialize_matdfmforecaster(
	int num_chains, int step, int nrow_factor, int ncol_factor, int factor_lag,
	BVHAR_LIST& fit_record, Eigen::Ref<const Eigen::VectorXi> seed_chain, int nthreads
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
		std::unique_ptr<MatFactorVarForecaster> factor_updater;
		initialize_matmniw_record(mat_record, i, fit_record, a_name, sigr_name, b_name, sigc_name);
		initialize_matdfm_record(mdfm_record, i, fit_record, f_name, rho_name, prec_name);
		auto* mdfm_var_record = dynamic_cast<MatDfmVarRecords*>(mdfm_record.get());
		// int num_row = mat_record->row_coef_record.cols() / nrow_factor;
		// int num_col = mat_record->col_coef_record.cols() / ncol_factor;
		factor_updater = std::make_unique<MatFactorVarForecaster>(
			*mdfm_var_record, step, factor_lag,
			mat_record->row_coef_record.cols() / nrow_factor,
			mat_record->col_coef_record.cols() / ncol_factor,
			nrow_factor, ncol_factor
		);
		forecaster[i] = std::make_unique<MatDfmVarForecaster>(*mat_record, factor_updater, step, static_cast<unsigned int>(seed_chain[i]));
	}
	return forecaster;
}

class MatDfmForecastRun : public bvhar::McmcForecastRun<Eigen::MatrixXd, Eigen::MatrixXd> {
public:
	MatDfmForecastRun(
		int num_chains, int step, int nrow_factor, int ncol_factor, int factor_lag,
		BVHAR_LIST& fit_record, Eigen::Ref<const Eigen::VectorXi> seed_chain, int nthreads
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
			fit_record, seed_chain, nthreads
		);
		for (int i = 0; i < num_chains; ++i) {
			forecaster[i] = std::move(temp_forecaster[i]);
		}
	}
	virtual ~MatDfmForecastRun() = default;
};

} // namespace baymar

#endif // BAYMAR_BAYES_MDFM_FORECASTER_H