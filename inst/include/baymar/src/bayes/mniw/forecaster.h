#ifndef BAYMAR_BAYES_MNIW_FORECASTER_H
#define BAYMAR_BAYES_MNIW_FORECASTER_H

#include "./mniw.h"
#include "../../math/design.h"
#include "../../core/forecaster.h"
// #include <bvhar/ols>

namespace baecon {
namespace baymar {

class MatMniwExogenForecaster;
class MatFactorForecaster;
class MatFactorRwForecaster;
class MatFactorVarForecaster;
class MatFactorMarForecaster;
class MatMniwForecaster;
class MatMniwForecastRun;
template <bool, bool> class MatMniwOutForecastRun;
template <bool, bool> class MatMniwRollForecastRun;
template <bool, bool> class MatMniwExpandForecastRun;

class MatMniwExogenForecaster : public MatExogenForecaster {
public:
	MatMniwExogenForecaster(int lag, const Eigen::MatrixXd& exogen, int num_exogen, int num_row, int num_col)
	: MatExogenForecaster(lag, exogen, num_exogen, num_row, num_col) {}
	virtual ~MatMniwExogenForecaster() = default;

	void updateCoefmat(const Eigen::VectorXd& row_coef_record, const Eigen::VectorXd& col_coef_record, int nrow_row_coef, int nrow_col_coef) {
		BVHAR_DEBUG_LOG(
			debug_logger,
			"updateCoefmat() called: row_coef_record: {}, nrow_row_coef: {}, num_row: {}, nrow_row_exogen: {}, col_coef_record: {}, nrow_col_coef: {}, num_col: {}, nrow_col_exogen: {}",
			row_coef_record.size(), nrow_row_coef, num_row, nrow_row_exogen,
			col_coef_record.size(), nrow_col_coef, num_col, nrow_col_exogen
		);
		updateCoef(
			bvhar::unvectorize(row_coef_record.segment(nrow_row_coef * num_row, nrow_row_exogen * num_row).transpose(), num_row),
			bvhar::unvectorize(col_coef_record.segment(nrow_col_coef * num_col, nrow_col_exogen * num_col).transpose(), num_col)
		);
	}
};

class MatFactorForecaster : public MatMniwExogenForecaster{
public:
	MatFactorForecaster(int step, int factor_lag, int num_row, int num_col, int nrow_factor, int ncol_factor)
	// : MatMniwExogenForecaster(0, Eigen::MatrixXd::Zero((factor_lag + step) * nrow_factor, ncol_factor), factor_lag + step, num_row, num_col),
	: MatMniwExogenForecaster(0, Eigen::MatrixXd::Zero(step * nrow_factor, ncol_factor), step, num_row, num_col),
		step(step), factor_lag(factor_lag), size_factor(nrow_factor * ncol_factor),
		factor_mean(step, Eigen::MatrixXd::Zero(nrow_factor, ncol_factor)) {}

	MatFactorForecaster(const MatDfmRecords& records, int step, int factor_lag, int num_row, int num_col, int nrow_factor, int ncol_factor)
	// : MatMniwExogenForecaster(0, Eigen::MatrixXd::Zero((factor_lag + step) * nrow_factor, ncol_factor), factor_lag + step, num_row, num_col),
	: MatMniwExogenForecaster(0, Eigen::MatrixXd::Zero(step * nrow_factor, ncol_factor), step, num_row, num_col),
		step(step), factor_lag(factor_lag), size_factor(nrow_factor * ncol_factor),
		factor_mean(step, Eigen::MatrixXd::Zero(nrow_factor, ncol_factor)) {
		mdfm_record = std::make_unique<MatDfmRecords>(records);
		num_design = mdfm_record->factor_record.cols() / size_factor;
	}
	virtual ~MatFactorForecaster() = default;

	int get_nrow_factor() {
		return nrow_exogen;
	}

	int get_ncol_factor() {
		return ncol_exogen;
	}

	void updateDesign(const int id) {
		for (int i = 0; i < step; ++i) {
		// for (int i = 0; i < num_design; ++i) {
			exogen.middleRows(i * nrow_exogen, nrow_exogen) = bvhar::unvectorize(
				mdfm_record->factor_record.row(id).segment(i * size_factor, size_factor),
				ncol_exogen
			);
		}
	}
	
	virtual void updateVarCoef(const int id, BVHAR_BHRNG& rng) {
		if (mdfm_record) {
			updateDesign(id);
			return;
		}
		Eigen::VectorXd vec_normal(size_factor);
		for (int h = 0; h < step; ++h) {
			for (int i = 0; i < size_factor; ++i) {
				vec_normal[i] = bvhar::normal_rand(rng);
			}
			exogen.middleRows(h * nrow_exogen, nrow_exogen) = bvhar::unvectorize(vec_normal, ncol_exogen);
		}
	}

	virtual double getLpl(
		int h, int i, Eigen::Ref<const Eigen::MatrixXd> valid_vec,
		Eigen::Ref<const Eigen::MatrixXd> forecast_mean,
		Eigen::Ref<const Eigen::MatrixXd> mar_row_lower, Eigen::Ref<const Eigen::MatrixXd> mar_col_lower
	) {
		// N(vec(A^T X_{T + h} B), CC^T otimes RR^T + Sigma_c otimes Sigma_r)
		Eigen::MatrixXd var_coef = bvhar::kronecker_eigen(col_coef.transpose(), row_coef.transpose());
		Eigen::MatrixXd noise_lower = bvhar::kronecker_eigen(mar_col_lower, mar_row_lower);
		Eigen::MatrixXd factor_cov = var_coef * var_coef.transpose() + noise_lower * noise_lower.transpose();
		// int var_dim = factor_cov.cols();
		// Eigen::LLT<Eigen::MatrixXd> llt_of_cov;
		// double temp_penalty = 0;
		// do {
		// 	llt_of_cov.compute((
		// 		factor_cov + temp_penalty * Eigen::MatrixXd::Identity(var_coef.rows(), var_coef.cols())
		// 	).selfadjointView<Eigen::Lower>());
		// 	temp_penalty += .01;
		// } while (llt_of_cov.info() == Eigen::NumericalIssue && temp_penalty < .1);
		// Eigen::VectorXd point_error = bvhar::vectorize_eigen(valid_vec - forecast_mean);
		// return -(
		// 	var_dim * log(2 * M_PI) + 2 * llt_of_cov.matrixL().toDenseMatrix().diagonal().array().log().sum() + llt_of_cov.solve(point_error).dot(point_error)
		// ) / 2;
		return computeDensity(valid_vec, forecast_mean, factor_cov);
	}

protected:
	int step, factor_lag;
	int size_factor, num_design;
	// Eigen::VectorXd vec_normal;
	// std::unique_ptr<bvhar::OlsSimulator> factor_generator;
	std::vector<Eigen::MatrixXd> factor_mean;
	std::unique_ptr<MatDfmRecords> mdfm_record;

	double computeDensity(
		Eigen::Ref<const Eigen::MatrixXd> valid_vec,
    Eigen::Ref<const Eigen::MatrixXd> mean_mat,
    Eigen::Ref<const Eigen::MatrixXd> cov) {
		int var_dim = cov.cols();
		Eigen::LLT<Eigen::MatrixXd> llt_of_cov;
		double temp_penalty = 0;
		do {
			llt_of_cov.compute((
				cov + temp_penalty * Eigen::MatrixXd::Identity(var_dim, var_dim)
			).selfadjointView<Eigen::Lower>());
			temp_penalty += .01;
		} while (llt_of_cov.info() == Eigen::NumericalIssue && temp_penalty < .1);
		Eigen::VectorXd point_error = bvhar::vectorize_eigen(valid_vec - mean_mat);
		return -(
			var_dim * log(2 * M_PI) + 2 * llt_of_cov.matrixL().toDenseMatrix().diagonal().array().log().sum() + llt_of_cov.solve(point_error).dot(point_error)
		) / 2;
	}
};

class MatFactorRwForecaster : public MatFactorForecaster {
public:
	MatFactorRwForecaster(const MatDfmRwRecords& records, int step, int num_row, int num_col, int nrow_factor, int ncol_factor)
	: MatFactorForecaster(step, 0, num_row, num_col, nrow_factor, ncol_factor),
		factor_sig(Eigen::VectorXd::Ones(size_factor)) {
		mdfm_record = std::make_unique<MatDfmRwRecords>(records);
		num_design = mdfm_record->factor_record.cols() / size_factor;
	}
	virtual ~MatFactorRwForecaster() = default;
	
	void updateVarCoef(const int id, BVHAR_BHRNG& rng) override {
		BVHAR_DEBUG_LOG(debug_logger, "updateVarCoef(id={}) called", id);
		if (factor_lag == 0) {
			updateDesign(id);
			return;
		}
		mdfm_record->updateParams(id, factor_sig);
		Eigen::VectorXd vec_normal(size_factor);
		Eigen::VectorXd factor_pred = mdfm_record->factor_record.row(id).segment((num_design - 1) * size_factor, size_factor);
		for (int h = 0; h < step; ++h) {
			factor_mean[h] = bvhar::unvectorize(factor_pred, ncol_exogen);
			for (int i = 0; i < size_factor; ++i) {
				vec_normal[i] = bvhar::normal_rand(rng) * sqrt(factor_sig[i]);
			}
			factor_pred.array() += vec_normal.array();
			exogen.middleRows(h * nrow_exogen, nrow_exogen) = bvhar::unvectorize(factor_pred, ncol_exogen);
		}
	}

	double getLpl(
		int h, int i, Eigen::Ref<const Eigen::MatrixXd> valid_vec,
		Eigen::Ref<const Eigen::MatrixXd> forecast_mean,
		Eigen::Ref<const Eigen::MatrixXd> mar_row_lower, Eigen::Ref<const Eigen::MatrixXd> mar_col_lower
	) override {
		// N(vec(A^T X_{T + h} B + R F_{T + h - 1} C^T), (C otimes R) Lambda (C otimes R)^T + Sigma_c otimes Sigma_r)
		Eigen::MatrixXd mean_mat = forecast_mean + row_coef.transpose() * factor_mean[h] * col_coef;
		Eigen::MatrixXd var_coef = bvhar::kronecker_eigen(col_coef.transpose(), row_coef.transpose());
		Eigen::MatrixXd noise_lower = bvhar::kronecker_eigen(mar_col_lower, mar_row_lower);
		Eigen::MatrixXd factor_cov = var_coef * factor_sig.asDiagonal() * var_coef.transpose() + noise_lower * noise_lower.transpose();
		return computeDensity(valid_vec, mean_mat, factor_cov);
	}

private:
	Eigen::VectorXd factor_sig;
};

class MatFactorVarForecaster : public MatFactorForecaster {
public:
	MatFactorVarForecaster(const MatDfmVarRecords& records, int step, int factor_lag, int num_row, int num_col, int nrow_factor, int ncol_factor)
	: MatFactorForecaster(step, factor_lag, num_row, num_col, nrow_factor, ncol_factor),
		factor_coef(Eigen::MatrixXd::Zero(size_factor * factor_lag, size_factor)),
		factor_sig(Eigen::VectorXd::Ones(size_factor)) {
		mdfm_record = std::make_unique<MatDfmVarRecords>(records);
		num_design = mdfm_record->factor_record.cols() / size_factor;
	}
	virtual ~MatFactorVarForecaster() = default;
	
	void updateVarCoef(const int id, BVHAR_BHRNG& rng) override {
		BVHAR_DEBUG_LOG(debug_logger, "updateVarCoef(id={}) called", id);
		if (factor_lag == 0) {
			// for (int i = 0; i < num_design; ++i) {
			// 	exogen.middleRows(i * nrow_exogen, nrow_exogen) = bvhar::unvectorize(
			// 		mdfm_record->factor_record.row(id).segment(i * size_factor, size_factor),
			// 		ncol_exogen
			// 	);
			// }
			updateDesign(id);
			return;
		}
		mdfm_record->updateParams(id, factor_coef, factor_sig, factor_lag);
		// exogen.topRows(factor_lag * nrow_exogen) = F_{T - s + 1}, ..., F_T
		// Eigen::MatrixXd factor_design(factor_lag, size_factor);
		Eigen::VectorXd factor_x(factor_lag * size_factor);
		Eigen::VectorXd vec_normal(size_factor);
		for (int i = 0; i < factor_lag; ++i) {
			// exogen.middleRows(i * nrow_exogen, nrow_exogen) = bvhar::unvectorize(
			// 	mdfm_record->factor_record.row(id).segment((num_design - factor_lag + i) * size_factor, size_factor),
			// 	ncol_exogen
			// );
			// factor_design.row(i) = mdfm_record->factor_record.row(id).segment((num_design - factor_lag + i) * size_factor, size_factor);
			// factor_x.segment(i * size_factor, size_factor) = mdfm_record->factor_record.row(id).segment((num_design - factor_lag + i) * size_factor, size_factor);
			factor_x.segment(i * size_factor, size_factor) = mdfm_record->factor_record.row(id).segment((num_design - 1 - i) * size_factor, size_factor);
		}
		Eigen::VectorXd tmp_x = factor_x.segment(size_factor, (factor_lag - 1) * size_factor);
		Eigen::VectorXd factor_pred = factor_x.head(size_factor);
		for (int h = 0; h < step; ++h) {
			factor_x.segment(size_factor, (factor_lag - 1) * size_factor) = tmp_x;
			factor_x.head(size_factor) = factor_pred;
			for (int i = 0; i < size_factor; ++i) {
				vec_normal[i] = bvhar::normal_rand(rng) * sqrt(factor_sig[i]);
			}
			factor_mean[h] = factor_coef.transpose() * factor_x;
			factor_pred = factor_mean[h] + vec_normal;
			tmp_x = factor_x.head((factor_lag - 1) * size_factor);
			// exogen.middleRows((factor_lag + h) * nrow_exogen, nrow_exogen) = bvhar::unvectorize(factor_pred, ncol_exogen);
			exogen.middleRows(h * nrow_exogen, nrow_exogen) = bvhar::unvectorize(factor_pred, ncol_exogen);
		}
		// factor_generator = std::make_unique<bvhar::OlsSimulator>(
		// 	step, 0, factor_lag,
		// 	// exogen.topRows(factor_lag * nrow_exogen), factor_coef, factor_sig.asDiagonal(), 2, 1
		// 	factor_design, factor_coef, factor_sig.asDiagonal(), 2, 1
		// );
		// exogen.bottomRows(step * nrow_exogen) = factor_generator->returnDgp();
		// factor_generator.reset();
	}

	double getLpl(
		int h, int i, Eigen::Ref<const Eigen::MatrixXd> valid_vec,
		Eigen::Ref<const Eigen::MatrixXd> forecast_mean,
		Eigen::Ref<const Eigen::MatrixXd> mar_row_lower, Eigen::Ref<const Eigen::MatrixXd> mar_col_lower
	) override {
		// N(vec(A^T X_{T + h} B + R E(F_{T + h}) C^T), (C otimes R) Lambda (C otimes R)^T + Sigma_c otimes Sigma_r)
		// E(vec(F_{T + h})) = sum H_i vec(F_{T + h - i})
		Eigen::MatrixXd mean_mat = forecast_mean + row_coef.transpose() * factor_mean[h] * col_coef;
		Eigen::MatrixXd var_coef = bvhar::kronecker_eigen(col_coef.transpose(), row_coef.transpose());
		Eigen::MatrixXd noise_lower = bvhar::kronecker_eigen(mar_col_lower, mar_row_lower);
		Eigen::MatrixXd factor_cov = var_coef * factor_sig.asDiagonal() * var_coef.transpose() + noise_lower * noise_lower.transpose();
		return computeDensity(valid_vec, mean_mat, factor_cov);
	}

private:
	Eigen::MatrixXd factor_coef;
	Eigen::VectorXd factor_sig;
};

class MatFactorMarForecaster : public MatFactorForecaster {
public:
	MatFactorMarForecaster(const MatDfmMarRecords& records, int step, int factor_lag, int num_row, int num_col, int nrow_factor, int ncol_factor)
	: MatFactorForecaster(step, factor_lag, num_row, num_col, nrow_factor, ncol_factor),
		mar_row_coef(Eigen::MatrixXd::Zero(nrow_factor * factor_lag, nrow_factor)),
		row_sig_lower(Eigen::MatrixXd::Ones(nrow_factor, nrow_factor)),
		mar_col_coef(Eigen::MatrixXd::Zero(ncol_factor * factor_lag, ncol_factor)),
		col_sig_lower(Eigen::MatrixXd::Ones(ncol_factor, ncol_factor)) {
		mdfm_record = std::make_unique<MatDfmMarRecords>(records);
		num_design = mdfm_record->factor_record.cols() / size_factor;
	}
	virtual ~MatFactorMarForecaster() = default;
	
	void updateVarCoef(const int id, BVHAR_BHRNG& rng) override {
		BVHAR_DEBUG_LOG(debug_logger, "updateVarCoef(id={}) called", id);
		if (factor_lag == 0) {
			updateDesign(id);
			return;
		}
		mdfm_record->updateParams(id, mar_row_coef, row_sig_lower, mar_col_coef, col_sig_lower);
		Eigen::MatrixXd factor_x = Eigen::MatrixXd::Zero(factor_lag * nrow_exogen, factor_lag * ncol_exogen);
		for (int i = 0; i < factor_lag; ++i) {
			factor_x.block(i * nrow_exogen, i * ncol_exogen, nrow_exogen, ncol_exogen) = bvhar::unvectorize(
				mdfm_record->factor_record.row(id).segment((num_design - 1 - i) * size_factor, size_factor),
				ncol_exogen
			);
		}
		Eigen::MatrixXd tmp_x = factor_x.bottomRightCorner(nrow_exogen * (factor_lag - 1), ncol_exogen * (factor_lag - 1));
		Eigen::MatrixXd factor_pred = factor_x.topLeftCorner(nrow_exogen, ncol_exogen);
		Eigen::MatrixXd error_mat(nrow_exogen, ncol_exogen);
		for (int h = 0; h < step; ++h) {
			factor_x.bottomRightCorner(nrow_exogen * (factor_lag - 1), ncol_exogen * (factor_lag - 1)) = tmp_x;
			factor_x.topLeftCorner(nrow_exogen, ncol_exogen) = factor_pred;
			for (int j = 0; j < ncol_exogen; ++j) {
				for (int i = 0; i < nrow_exogen; ++i) {
					error_mat(i, j) = bvhar::normal_rand(rng);
				}
			}
			error_mat = row_sig_lower * error_mat * col_sig_lower.transpose();
			factor_mean[h].setZero();
			for (int i = 0; i < lag; ++i) {
				factor_mean[h] += mar_row_coef.middleRows(i * nrow_exogen, nrow_exogen).transpose() * factor_x.block(i * num_row, i * num_col, num_row, num_col) * mar_col_coef.middleRows(i * ncol_exogen, ncol_exogen);
			}
			factor_pred = factor_mean[h] + error_mat;
			tmp_x = factor_x.bottomRightCorner(nrow_exogen * (factor_lag - 1), ncol_exogen * (factor_lag - 1));
			exogen.middleRows(h * nrow_exogen, nrow_exogen) = bvhar::unvectorize(factor_pred, ncol_exogen);
		}
	}

	double getLpl(
		int h, int i, Eigen::Ref<const Eigen::MatrixXd> valid_vec,
		Eigen::Ref<const Eigen::MatrixXd> forecast_mean,
		Eigen::Ref<const Eigen::MatrixXd> mar_row_lower, Eigen::Ref<const Eigen::MatrixXd> mar_col_lower
	) override {
		// N(vec(A^T X_{T + h} B + R E(F_{T + h}) C^T), (C Sig_{fc} C^T) otimes (R Sig_{fr} R^T) + Sigma_c otimes Sigma_r)
		// E(F_{T + h}) = G^T diag(F_{T + h - 1}, ..., F_{T + h - p_f}) H
		Eigen::MatrixXd mean_mat = forecast_mean + row_coef.transpose() * factor_mean[h] * col_coef;
		Eigen::MatrixXd var_coef = bvhar::kronecker_eigen(col_coef.transpose(), row_coef.transpose());
		Eigen::MatrixXd factor_mar_lower = bvhar::kronecker_eigen(col_sig_lower, row_sig_lower);
		Eigen::MatrixXd noise_lower = bvhar::kronecker_eigen(mar_col_lower, mar_row_lower);
		Eigen::MatrixXd factor_cov = var_coef * factor_mar_lower * factor_mar_lower.transpose() * var_coef.transpose() + noise_lower * noise_lower.transpose();
		return computeDensity(valid_vec, mean_mat, factor_cov);
	}

private:
	Eigen::MatrixXd mar_row_coef, row_sig_lower, mar_col_coef, col_sig_lower;
};

inline std::unique_ptr<MatFactorForecaster> initialize_matfactorforecaster(
	int step, BVHAR_LIST& fit_record, int chain_id, int num_row, int num_col,
	int nrow_factor, int ncol_factor, int factor_lag,
	BVHAR_OPTIONAL<bool> factor_insample = BVHAR_NULLOPT
) {
	std::unique_ptr<MatFactorForecaster> factor_forecaster;
	std::unique_ptr<MatDfmRecords> mdfm_record;
	BVHAR_STRING f_name = "F_record";
	if (BVHAR_CONTAINS(fit_record, "Rho_record")) {
		BVHAR_STRING rho_name = "Rho_record";
		BVHAR_STRING prec_name = "Lambda_record";
		initialize_matdfm_record(mdfm_record, chain_id, fit_record, f_name, rho_name, prec_name);
		auto* mdfm_var_record = dynamic_cast<MatDfmVarRecords*>(mdfm_record.get());
		factor_forecaster = std::make_unique<MatFactorVarForecaster>(*mdfm_var_record, step, factor_lag, num_row, num_col, nrow_factor, ncol_factor);
	} else if (BVHAR_CONTAINS(fit_record, "Lambda_record")) {
		BVHAR_STRING prec_name = "Lambda_record";
		initialize_matdfm_record(mdfm_record, chain_id, fit_record, f_name, BVHAR_NULLOPT, prec_name);
		auto* mdfm_rw_record = dynamic_cast<MatDfmRwRecords*>(mdfm_record.get());
		factor_forecaster = std::make_unique<MatFactorRwForecaster>(*mdfm_rw_record, step, num_row, num_col, nrow_factor, ncol_factor);
	} else if (BVHAR_CONTAINS(fit_record, "FA_record")) {
		BVHAR_STRING fa_name = "FA_record";
		BVHAR_STRING fb_name = "FB_record";
		BVHAR_STRING omegar_name = "OmegaR_record";
		BVHAR_STRING omegac_name = "OmegaC_record";
		initialize_matdfm_record(
			mdfm_record, chain_id, fit_record,
			f_name, BVHAR_NULLOPT, BVHAR_NULLOPT,
			fa_name, fb_name, omegar_name, omegac_name
		);
		auto* mdfm_mar_record = dynamic_cast<MatDfmMarRecords*>(mdfm_record.get());
		factor_forecaster = std::make_unique<MatFactorMarForecaster>(*mdfm_mar_record, step, factor_lag, num_row, num_col, nrow_factor, ncol_factor);
	} else if (factor_insample && *factor_insample) {
		initialize_matdfm_record(mdfm_record, chain_id, fit_record, f_name);
		factor_forecaster = std::make_unique<MatFactorForecaster>(*mdfm_record, step, 0, num_row, num_col, nrow_factor, ncol_factor);
	} else {
		factor_forecaster = std::make_unique<MatFactorForecaster>(step, 0, num_row, num_col, nrow_factor, ncol_factor);
	}
	return factor_forecaster;
}

class MatMniwForecaster : public bvhar::BayesForecaster<Eigen::MatrixXd, Eigen::MatrixXd> {
public:
	MatMniwForecaster(
		const MatMniwRecords& records, int step, const Eigen::MatrixXd& y, int num_data, int lag, unsigned int seed,
		bool save_mean = false,
		BVHAR_OPTIONAL<std::unique_ptr<MatMniwExogenForecaster>> exogen_forecaster = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<std::unique_ptr<MatFactorForecaster>> famar_forecaster = BVHAR_NULLOPT
	)
	: bvhar::BayesForecaster<Eigen::MatrixXd, Eigen::MatrixXd>(step, y, lag, records.row_coef_record.rows(), seed, save_mean),
		mat_record(std::make_unique<MatMniwRecords>(records)),
		num_row(y.rows() / num_data), num_col(y.cols()), nrow_row_coef(num_row * lag), nrow_col_coef(num_col * lag),
		nrow_row_exogen(0), nrow_col_exogen(0),
		row_coef(Eigen::MatrixXd::Zero(nrow_row_coef, num_row)),
		row_sig_lower(Eigen::MatrixXd::Identity(num_row, num_row)),
		col_coef(Eigen::MatrixXd::Zero(nrow_col_coef, num_col)),
		col_sig_lower(Eigen::MatrixXd::Identity(num_col, num_col)),
		error_mat(Eigen::MatrixXd::Zero(num_row, num_col)) {
		BVHAR_DEBUG_LOG(debug_logger, "MatMniwForecaster Constructor: step={}, num_data={}, lag={}", step, num_data, lag);
		initLagged();
		if (exogen_forecaster) {
			exogen_updater = std::move(*exogen_forecaster);
			nrow_row_exogen = exogen_updater->get_nrow_row_exogen();
			nrow_col_exogen = exogen_updater->get_nrow_col_exogen();
		}
		if (famar_forecaster) {
			famar_updater = std::move(*famar_forecaster);
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
	std::unique_ptr<MatFactorForecaster> famar_updater;
	int num_row, num_col, nrow_row_coef, nrow_col_coef, nrow_row_exogen, nrow_col_exogen;
	Eigen::MatrixXd row_coef, row_sig_lower, col_coef, col_sig_lower, error_mat;

	void initLagged() override {
		BVHAR_DEBUG_LOG(debug_logger, "initLagged() called");
		last_pvec = build_dense_design(response, num_row, lag);
		point_forecast = Eigen::MatrixXd::Zero(num_row, num_col);
		forecast_mean = Eigen::MatrixXd::Zero(num_row, num_col);
		if (save_mean) {
			mean_save = Eigen::MatrixXd::Zero(step * num_row, num_sim * num_col);
		}
		pred_save = Eigen::MatrixXd::Zero(step * num_row, num_sim * num_col);
		tmp_vec = last_pvec.block(num_row, num_col, num_row * (lag - 1), num_col * (lag - 1));
	}

	void initRecursion(const Eigen::MatrixXd& obs_vec) override {
		BVHAR_DEBUG_LOG(debug_logger, "initRecursion(obs_vec) called");
		last_pvec = obs_vec;
		point_forecast = obs_vec.topLeftCorner(num_row, num_col);
		// tmp_vec = obs_vec.bottomRightCorner(num_row * (lag - 1), num_col * (lag - 1));
		tmp_vec = obs_vec.block(num_row, num_col, num_row * (lag - 1), num_col * (lag - 1));
	}

	void setRecursion() override {
		BVHAR_DEBUG_LOG(debug_logger, "setRecursion() called");
		last_pvec.bottomRightCorner(num_row * (lag - 1), num_col * (lag - 1)) = tmp_vec;
		last_pvec.topLeftCorner(num_row, num_col) = point_forecast;
	}

	void updatePred(const int h, const int i) override {
		BVHAR_DEBUG_LOG(debug_logger, "updatePred(h={}, i={}) called", h, i);
		computeMean();
		if (save_mean) {
			mean_save.block(h * num_row, i * num_col, num_row, num_col) = point_forecast;
		}
		forecast_mean = point_forecast;
		updateVariance();
		// point_forecast += error_mat;
		if (exogen_updater) {
			exogen_updater->appendForecast(point_forecast, h);
		}
		if (famar_updater) {
			famar_updater->appendForecast(point_forecast, h);
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
		if (famar_updater) {
			famar_updater->updateCoefmat(
				mat_record->row_coef_record.row(i).transpose(),
				mat_record->col_coef_record.row(i).transpose(),
				nrow_row_coef + nrow_row_exogen,
				nrow_col_coef + nrow_col_exogen
			);
			famar_updater->updateVarCoef(i, rng);
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

	void updateLpl(int h, int i, const Eigen::MatrixXd& valid_vec) override {
		BVHAR_DEBUG_LOG(debug_logger, "updateLpl(h={}, valid_vec) called", h);
		if (famar_updater) {
			lpl(h, i) = famar_updater->getLpl(h, i, valid_vec, forecast_mean, row_sig_lower, col_sig_lower);
		} else {
			lpl(h, i) = -(
				col_sig_lower.transpose().triangularView<Eigen::Upper>().solve<Eigen::OnTheRight>(
					row_sig_lower.triangularView<Eigen::Lower>().solve(valid_vec - point_forecast)
				).squaredNorm() / 2 + num_row * num_col * log(2 * M_PI) / 2 + num_col * row_sig_lower.diagonal().array().log().sum() + num_row * col_sig_lower.diagonal().array().log().sum()
			);
		}
	}

	Eigen::MatrixXd getDesign() override {
		BVHAR_DEBUG_LOG(debug_logger, "getDesign() called");
		return Eigen::MatrixXd();
		// if (exogen_updater) {
		// 	return this->response;
		// }
		// return this->response;
	}

	void forecastIn(const int i, const Eigen::MatrixXd& design) override {
		BVHAR_DEBUG_LOG(debug_logger, "forecastIn(i={}, design) called", i);
		for (int h = 0; h < step; ++h) {
			point_forecast.setZero();
			for (int j = 0; j < lag; ++j) {
				point_forecast += row_coef.middleRows(j * num_row, num_row).transpose() * response.middleRows((lag + h - j - 1) * num_row, num_row) * col_coef.middleRows(j * num_col, num_col);
			}
			if (save_mean) {
				mean_save.block(h * num_row, i * num_col, num_row, num_col) = point_forecast;
			}
			if (exogen_updater) {
				exogen_updater->appendForecast(point_forecast, lag + h - exogen_updater->getLag());
			}
			if (famar_updater) {
				famar_updater->appendForecast(point_forecast, h);
			}
			updateVariance();
			pred_save.block(h * num_row, i * num_col, num_row, num_col) = point_forecast + error_mat;
		}
	}
};

inline std::vector<std::unique_ptr<MatMniwForecaster>> initialize_matmniwforecaster(
	int num_chains, int lag, int step, const Eigen::MatrixXd& y, int num_data,
	BVHAR_LIST& fit_record, Eigen::Ref<const Eigen::VectorXi> seed_chain, int nthreads,
	bool save_mean = false,
	BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT,
	BVHAR_OPTIONAL<int> nrow_factor = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> ncol_factor = BVHAR_NULLOPT,
	BVHAR_OPTIONAL<int> factor_lag = BVHAR_NULLOPT, BVHAR_OPTIONAL<bool> factor_insample = BVHAR_NULLOPT
) {
	// BVHAR_PY_LIST row_coef_record = fit_record["A_record"];
	// BVHAR_PY_LIST row_sigma_record = fit_record["SigmaR_record"];
	// BVHAR_PY_LIST col_coef_record = fit_record["B_record"];
	// BVHAR_PY_LIST col_sigma_record = fit_record["SigmaC_record"];
	BVHAR_STRING a_name = "A_record";
	BVHAR_STRING sigr_name = "SigmaR_record";
	BVHAR_STRING b_name = "B_record";
	BVHAR_STRING sigc_name = "SigmaC_record";
	std::vector<std::unique_ptr<MatMniwForecaster>> forecaster(num_chains);
	for (int i = 0; i < num_chains; ++i) {
		std::unique_ptr<MatMniwRecords> mat_record;
		BVHAR_OPTIONAL<std::unique_ptr<MatMniwExogenForecaster>> exogen_updater = BVHAR_NULLOPT;
		std::unique_ptr<MatDfmRecords> mdfm_record;
		BVHAR_OPTIONAL<std::unique_ptr<MatFactorForecaster>> factor_updater = BVHAR_NULLOPT;
		BVHAR_OPTIONAL<BVHAR_STRING> g_name = BVHAR_NULLOPT;
		BVHAR_OPTIONAL<BVHAR_STRING> h_name = BVHAR_NULLOPT;
		if (nrow_factor) {
			g_name = "G_record";
			h_name = "H_record";
		}
		if (exogen) {
			BVHAR_STRING c_name = "C_record";
			BVHAR_STRING d_name = "D_record";
			exogen_updater = std::make_unique<MatMniwExogenForecaster>(*exogen_lag, *exogen, *exogen_lag + step, y.rows() / num_data, y.cols());
			initialize_matmniw_record(mat_record, i, fit_record, a_name, sigr_name, b_name, sigc_name, c_name, d_name, g_name, h_name);
		} else {
			initialize_matmniw_record(mat_record, i, fit_record, a_name, sigr_name, b_name, sigc_name, g_name, h_name);
		}
		if (nrow_factor) {
			// BVHAR_STRING f_name = "F_record";
			// if (BVHAR_CONTAINS(fit_record, "Rho_record")) {
			// 	BVHAR_STRING rho_name = "Rho_record";
			// 	BVHAR_STRING prec_name = "Lambda_record";
			// 	initialize_matdfm_record(mdfm_record, i, fit_record, f_name, rho_name, prec_name);
			// 	auto* mdfm_var_record = dynamic_cast<MatDfmVarRecords*>(mdfm_record.get());
			// 	factor_updater = std::make_unique<MatFactorVarForecaster>(*mdfm_var_record, step, *factor_lag, y.rows() / num_data, y.cols(), *nrow_factor, *ncol_factor);
			// } else {
			// 	if (factor_insample && *factor_insample) {
			// 		initialize_matdfm_record(mdfm_record, i, fit_record, f_name);
			// 		factor_updater = std::make_unique<MatFactorForecaster>(*mdfm_record, step, 0, y.rows() / num_data, y.cols(), *nrow_factor, *ncol_factor);
			// 	} else {
			// 		factor_updater = std::make_unique<MatFactorForecaster>(step, 0, y.rows() / num_data, y.cols(), *nrow_factor, *ncol_factor);
			// 	}
			// }
			factor_updater = initialize_matfactorforecaster(
				step, fit_record, i, y.rows() / num_data, y.cols(),
				*nrow_factor, *ncol_factor, *factor_lag, factor_insample
			);
		}
		forecaster[i] = std::make_unique<MatMniwForecaster>(
			*mat_record, step, y, num_data, lag, static_cast<unsigned int>(seed_chain[i]), save_mean,
			std::move(exogen_updater), std::move(factor_updater)
		);
	}
	return forecaster;
}

class MatMniwForecastRun : public bvhar::McmcForecastRun<Eigen::MatrixXd, Eigen::MatrixXd> {
public:
	MatMniwForecastRun(
		int num_chains, int lag, int step, const Eigen::MatrixXd& y, int num_data,
		BVHAR_LIST& fit_record, const Eigen::VectorXi& seed_chain, int nthreads,
		bool save_mean = false,
		BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<int> nrow_factor = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> ncol_factor = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<int> factor_lag = BVHAR_NULLOPT, BVHAR_OPTIONAL<bool> factor_insample = BVHAR_NULLOPT
	)
	: bvhar::McmcForecastRun<Eigen::MatrixXd, Eigen::MatrixXd>(num_chains, lag, step, nthreads) {
		BVHAR_DEBUG_LOG(
			debug_logger,
			"MatMniwForecastRun Constructor: num_chains={}, lag={}, step={}, num_data={}, nthreads={}",
			num_chains, lag, step, num_data, nthreads
		);
		auto temp_forecaster = initialize_matmniwforecaster(
			num_chains, lag, step, y, num_data, fit_record, seed_chain, nthreads, save_mean,
			exogen, exogen_lag,
			nrow_factor, ncol_factor, factor_lag, factor_insample
		);
		for (int i = 0; i < num_chains; ++i) {
			forecaster[i] = std::move(temp_forecaster[i]);
		}
	}
	virtual ~MatMniwForecastRun() = default;
};

template <bool isPath = false, bool isUpdate = true>
class MatMniwOutForecastRun : public bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate> {
public:
	MatMniwOutForecastRun(
		const Eigen::MatrixXd& y, int num_data, int lag,
		int num_chains, int num_iter, int num_burn, int thin, BVHAR_LIST& fit_record,
		BVHAR_LIST& param_coef_sig, BVHAR_LIST_OF_LIST& coef_sig_init,
		BVHAR_LIST& row_prior, BVHAR_LIST_OF_LIST& row_init, const int row_prior_type,
		BVHAR_LIST& col_prior, BVHAR_LIST_OF_LIST& col_init, const int col_prior_type,
		int step, const Eigen::MatrixXd& y_test, bool get_lpl, bool use_fit,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads,
		BVHAR_OPTIONAL<BVHAR_LIST> row_exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> row_exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> row_exogen_prior_type = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<BVHAR_LIST> col_exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> col_exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> col_exogen_prior_type = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<BVHAR_LIST> row_factor_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> row_factor_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> row_factor_prior_type = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> nrow_factor = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<BVHAR_LIST> col_factor_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> col_factor_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> col_factor_prior_type = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> ncol_factor = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<int> factor_lag = BVHAR_NULLOPT
	)
	: bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>(
			num_data, lag, num_chains, num_iter, num_burn, thin, step, y_test, y_test.rows(), get_lpl, use_fit,
			seed_chain, seed_forecast, display_progress, nthreads,
			exogen_lag
		),
		num_row(y.rows() / num_data), num_col(y.cols()), nrow_row_coef(num_row * lag), nrow_col_coef(num_col * lag),
		nrow_factor(nrow_factor), ncol_factor(ncol_factor), factor_lag(factor_lag) {
		BVHAR_DEBUG_LOG(debug_logger, "MatMniwOutForecastRun Constructor: num_data={}, row_prior_type={}, col_prior_type={}", num_data, row_prior_type, col_prior_type);
		num_test /= num_row;
		num_horizon = num_test - step + 1;
		roll_mat.resize(num_horizon);
		model.resize(num_horizon);
		out_forecast.resize(num_horizon);
		lpl_record.resize(num_horizon, num_chains);
		lpl_record = Eigen::MatrixXd::Zero(num_horizon, num_chains);
		if (factor_lag) {
			BVHAR_STRING factor_model_nm = BVHAR_CAST<BVHAR_STRING>(param_coef_sig["factor_type"]);
			if (factor_model_nm == "wn") {
				factor_type = 1;
			} else if (factor_model_nm == "rw") {
				factor_type = 4;
			} else if (factor_model_nm == "var") {
				factor_type = 2;
			} else if (factor_model_nm == "mar") {
				factor_type = 3;
			}
		}
		// for (int i = 0; i < num_horizon; ++i) {
		// 	model[i].resize(num_chains);
		// 	forecaster[i].resize(num_chains);
		// 	out_forecast[i].resize(num_chains);
		// }
	}
	virtual ~MatMniwOutForecastRun() = default;

protected:
	int num_row, num_col, nrow_row_coef, nrow_col_coef;
	BVHAR_OPTIONAL<int> nrow_factor, ncol_factor, factor_lag, factor_type;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>::num_window;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>::num_test;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>::num_horizon;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>::step;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>::lag;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>::num_chains;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>::num_iter;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>::num_burn;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>::thin;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>::nthreads;
	// using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::get_lpl;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>::use_fit;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>::display_progress;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>::seed_forecast;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>::roll_mat;
	// using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isUpdate>::roll_y0;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>::y_test;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>::model;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>::forecaster;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>::out_forecast;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>::lpl_record;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>::roll_exogen_mat;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>::roll_exogen;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>::lag_exogen;
	using bvhar::McmcOutForecastRun<Eigen::MatrixXd, Eigen::MatrixXd, isPath, isUpdate>::debug_logger;

	Eigen::MatrixXd getValid() override {
		BVHAR_DEBUG_LOG(debug_logger, "getValid() called");
		return y_test.bottomRows(num_row);
	}

	void initForecaster(BVHAR_LIST& fit_record) {
		BVHAR_DEBUG_LOG(debug_logger, "initForecaster(fit_record) called");
		using is_mcmc = std::integral_constant<bool, isUpdate>;
		if (is_mcmc::value) {
			auto temp_forecaster = initialize_matmniwforecaster(
				num_chains, lag, step, roll_mat[0], num_window, fit_record, seed_forecast, nthreads, false,
				roll_exogen[0], lag_exogen,
				nrow_factor, ncol_factor, factor_lag
			);
			for (int i = 0; i < num_chains; ++i) {
				forecaster[0][i] = std::move(temp_forecaster[i]);
			}
		} else {
			for (int window = 0; window < num_horizon; ++window) {
				auto temp_forecaster = initialize_matmniwforecaster(
					num_chains, lag, step, roll_mat[window], num_window, fit_record, seed_forecast, nthreads, false,
					roll_exogen[window], lag_exogen,
					nrow_factor, ncol_factor, factor_lag
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
		const Eigen::MatrixXi& seed_chain,
		BVHAR_OPTIONAL<BVHAR_LIST> row_exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> row_exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> row_exogen_prior_type = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<BVHAR_LIST> col_exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> col_exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> col_exogen_prior_type = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<BVHAR_LIST> row_factor_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> row_factor_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> row_factor_prior_type = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<BVHAR_LIST> col_factor_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> col_factor_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> col_factor_prior_type = BVHAR_NULLOPT
	) {
		BVHAR_DEBUG_LOG(debug_logger, "initialize(...) called");
		initData(y, exogen);
		if (use_fit) {
			initForecaster(fit_record);
		}
		BVHAR_OPTIONAL<int> exogen_rows = BVHAR_NULLOPT;
 		BVHAR_OPTIONAL<int> exogen_cols = BVHAR_NULLOPT;
		for (int window = 0; window < num_horizon; ++window) {
			if (use_fit && window == 0) {
				continue;
			}
			std::vector<Eigen::MatrixXd> y_data = marmatrix_to_vector(roll_mat[window], num_row);
			std::vector<Eigen::MatrixXd> response = build_mar_response(y_data, lag);
			BVHAR_OPTIONAL<std::vector<Eigen::MatrixXd>> exogen_data = BVHAR_NULLOPT;
			if (lag_exogen) {
				int nrow_exogen = exogen->rows() / (num_window + num_test);
				exogen_data = marmatrix_to_vector(*(roll_exogen_mat[window]), nrow_exogen);
			}
			int rows_factor = nrow_factor ? *nrow_factor : 0;
			int cols_factor = ncol_factor ? *ncol_factor : 0;
			std::vector<Eigen::SparseMatrix<double>> design = lag_exogen ? build_mar_design(y_data, *exogen_data, lag, *lag_exogen, rows_factor, cols_factor) : build_mar_design(y_data, lag, rows_factor, cols_factor);
			// std::vector<Eigen::SparseMatrix<double>> design = lag_exogen ? build_mar_design(y_data, *exogen_data, lag, *lag_exogen) : build_mar_design(y_data, lag);
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
				col_exogen_prior, col_exogen_init, col_exogen_prior_type, exogen_cols,
				lag_exogen,
				row_factor_prior, row_factor_init, row_factor_prior_type, nrow_factor,
				col_factor_prior, col_factor_init, col_factor_prior_type, ncol_factor,
				factor_lag
			);
			for (int i = 0; i < num_chains; ++i) {
				model[window][i] = std::move(temp_mcmc[i]);
			}
		}
		// using is_mcmc = std::integral_constant<bool, isUpdate>;
		// if (is_mcmc::value) {
		// 	// initMcmc(
		// 	// 	param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
		// 	// 	seed_chain
		// 	// );
		// 	BVHAR_OPTIONAL<int> exogen_rows = BVHAR_NULLOPT;
 		// 	BVHAR_OPTIONAL<int> exogen_cols = BVHAR_NULLOPT;
		// 	for (int window = 0; window < num_horizon; ++window) {
		// 		std::vector<Eigen::MatrixXd> y_data = marmatrix_to_vector(roll_mat[window], num_row);
		// 		std::vector<Eigen::MatrixXd> response = build_mar_response(y_data, lag);
		// 		BVHAR_OPTIONAL<std::vector<Eigen::MatrixXd>> exogen_data = BVHAR_NULLOPT;
		// 		if (lag_exogen) {
		// 			int nrow_exogen = exogen->rows() / (num_window + num_test);
		// 			exogen_data = marmatrix_to_vector(*(roll_exogen_mat[window]), nrow_exogen);
		// 		}
		// 		int rows_factor = nrow_factor ? *nrow_factor : 0;
		// 		int cols_factor = ncol_factor ? *ncol_factor : 0;
		// 		std::vector<Eigen::SparseMatrix<double>> design = lag_exogen ? build_mar_design(y_data, *exogen_data, lag, *lag_exogen, rows_factor, cols_factor) : build_mar_design(y_data, lag, rows_factor, cols_factor);
		// 		if (lag_exogen) {
		// 			exogen_rows = (*lag_exogen + 1) * (*exogen_data)[0].rows();
		// 			exogen_cols = (*lag_exogen + 1) * (*exogen_data)[0].cols();
		// 		}
		// 		auto temp_mcmc = initialize_matmcmc(
		// 			num_chains, num_iter - num_burn, design, response,
		// 			param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type,
		// 			col_prior, col_init, col_prior_type,
		// 			seed_chain.row(window),
		// 			row_exogen_prior, row_exogen_init, row_exogen_prior_type, exogen_rows,
		// 			col_exogen_prior, col_exogen_init, col_exogen_prior_type, exogen_cols,
		// 			row_factor_prior, row_factor_init, row_factor_prior_type, nrow_factor,
		// 			col_factor_prior, col_factor_init, col_factor_prior_type, ncol_factor,
		// 			factor_lag
		// 		);
		// 		auto temp_forecaster = initialize_matmniwforecaster(
		// 			num_chains, lag, step, roll_mat[window], num_window, fit_record, seed_forecast, nthreads,
		// 			roll_exogen[window], lag_exogen,
		// 			nrow_factor, ncol_factor, factor_lag
		// 		);
		// 		for (int i = 0; i < num_chains; ++i) {
		// 			model[window][i] = std::move(temp_mcmc[i]);
		// 			forecaster[window][i] = std::move(temp_forecaster[i]);
		// 		}
		// 	}
		// } else {
		// 	auto temp_forecaster = initialize_matmniwforecaster(
		// 		num_chains, lag, step, roll_mat[0], num_window, fit_record, seed_forecast, nthreads,
		// 		roll_exogen[0], lag_exogen,
		// 		nrow_factor, ncol_factor, factor_lag
		// 	);
		// 	for (int i = 0; i < num_chains; ++i) {
		// 		forecaster[0][i] = std::move(temp_forecaster[i]);
		// 	}
		// }
	}

	virtual void initData(const Eigen::MatrixXd& y, BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT) = 0;

	void updateForecaster(int window, int chain) override {
		BVHAR_DEBUG_LOG(debug_logger, "updateForecaster(window={}, chain={}) called", window, chain);
		auto* mcmc_mniw = dynamic_cast<McmcMatMniw*>(model[window][chain].get());
		MatMniwRecords mniw_record = mcmc_mniw->returnStructRecords(0, thin);
		BVHAR_OPTIONAL<std::unique_ptr<MatMniwExogenForecaster>> exogen_updater = BVHAR_NULLOPT;
		BVHAR_OPTIONAL<std::unique_ptr<MatFactorForecaster>> factor_updater = BVHAR_NULLOPT;
		if (lag_exogen) {
			exogen_updater = std::make_unique<MatMniwExogenForecaster>(*lag_exogen, *(roll_exogen[window]), *lag_exogen + step, num_row, num_col);
		}
		if (nrow_factor) {
			// if (factor_lag && *factor_lag != 0) {
			// 	auto mdfm_var_record = mcmc_mniw->returnFactorRecords<MatDfmVarRecords>(0, thin);
			// 	factor_updater = std::make_unique<MatFactorVarForecaster>(mdfm_var_record, step, *factor_lag, num_row, num_col, *nrow_factor, *ncol_factor);
			// } else {
			// 	factor_updater = std::make_unique<MatFactorForecaster>(step, 0, num_row, num_col, *nrow_factor, *ncol_factor);
			// }
			if (*factor_type == 1) {
				factor_updater = std::make_unique<MatFactorForecaster>(step, 0, num_row, num_col, *nrow_factor, *ncol_factor);
			} else if (*factor_type == 2) {
				auto mdfm_var_record = mcmc_mniw->returnFactorRecords<MatDfmVarRecords>(0, thin);
				factor_updater = std::make_unique<MatFactorVarForecaster>(mdfm_var_record, step, *factor_lag, num_row, num_col, *nrow_factor, *ncol_factor);
			} else if (*factor_type == 3) {
				auto mdfm_mar_record = mcmc_mniw->returnFactorRecords<MatDfmMarRecords>(0, thin);
				factor_updater = std::make_unique<MatFactorMarForecaster>(mdfm_mar_record, step, *factor_lag, num_row, num_col, *nrow_factor, *ncol_factor);
			} else if (*factor_type == 4) {
				auto mdfm_rw_record = mcmc_mniw->returnFactorRecords<MatDfmRwRecords>(0, thin);
				factor_updater = std::make_unique<MatFactorRwForecaster>(mdfm_rw_record, step, num_row, num_col, *nrow_factor, *ncol_factor);
			} else {
				BVHAR_STOP("Wrong factor type");
			}
		}
		forecaster[window][chain] = std::make_unique<MatMniwForecaster>(
			mniw_record, step, roll_mat[window], num_window, lag, static_cast<unsigned int>(seed_forecast[chain]), false,
			std::move(exogen_updater), std::move(factor_updater)
		);
	}
};

template <bool isPath = false, bool isUpdate = true>
class MatMniwRollForecastRun : public MatMniwOutForecastRun<isPath, isUpdate> {
public:
	MatMniwRollForecastRun(
		const Eigen::MatrixXd& y, int num_data, int lag,
		int num_chains, int num_iter, int num_burn, int thin, BVHAR_LIST& fit_record,
		BVHAR_LIST& param_coef_sig, BVHAR_LIST_OF_LIST& coef_sig_init,
		BVHAR_LIST& row_prior, BVHAR_LIST_OF_LIST& row_init, const int row_prior_type,
		BVHAR_LIST& col_prior, BVHAR_LIST_OF_LIST& col_init, const int col_prior_type,
		int step, const Eigen::MatrixXd& y_test, bool get_lpl, bool use_fit,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads,
		BVHAR_OPTIONAL<BVHAR_LIST> row_exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> row_exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> row_exogen_prior_type = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<BVHAR_LIST> col_exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> col_exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> col_exogen_prior_type = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<BVHAR_LIST> row_factor_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> row_factor_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> row_factor_prior_type = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> nrow_factor = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<BVHAR_LIST> col_factor_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> col_factor_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> col_factor_prior_type = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> ncol_factor = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<int> factor_lag = BVHAR_NULLOPT
	)
	: MatMniwOutForecastRun<isPath, isUpdate>(
			y, num_data, lag,
			num_chains, num_iter, num_burn, thin, fit_record,
			param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
			step, y_test, get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads,
			row_exogen_prior, row_exogen_init, row_exogen_prior_type,
			col_exogen_prior, col_exogen_init, col_exogen_prior_type,
			exogen, exogen_lag,
			row_factor_prior, row_factor_init, row_factor_prior_type, nrow_factor,
			col_factor_prior, col_factor_init, col_factor_prior_type, ncol_factor,
			factor_lag
		) {
		BVHAR_DEBUG_LOG(debug_logger, "MatMniwRollForecastRun constructor");
		initialize(
			y, fit_record,
			param_coef_sig, coef_sig_init,
			row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
			seed_chain,
			row_exogen_prior, row_exogen_init, row_exogen_prior_type,
			col_exogen_prior, col_exogen_init, col_exogen_prior_type,
			exogen, exogen_lag,
			row_factor_prior, row_factor_init, row_factor_prior_type,
			col_factor_prior, col_factor_init, col_factor_prior_type
		);
	}
	virtual ~MatMniwRollForecastRun() = default;

protected:
	using MatMniwOutForecastRun<isPath, isUpdate>::num_window;
	using MatMniwOutForecastRun<isPath, isUpdate>::num_row;
	using MatMniwOutForecastRun<isPath, isUpdate>::num_col;
	using MatMniwOutForecastRun<isPath, isUpdate>::num_test;
	using MatMniwOutForecastRun<isPath, isUpdate>::num_horizon;
	using MatMniwOutForecastRun<isPath, isUpdate>::step;
	using MatMniwOutForecastRun<isPath, isUpdate>::roll_mat;
	using MatMniwOutForecastRun<isPath, isUpdate>::y_test;
	using MatMniwOutForecastRun<isPath, isUpdate>::initialize;
	using MatMniwOutForecastRun<isPath, isUpdate>::roll_exogen_mat;
	using MatMniwOutForecastRun<isPath, isUpdate>::roll_exogen;
	using MatMniwOutForecastRun<isPath, isUpdate>::lag_exogen;
	using MatMniwOutForecastRun<isPath, isUpdate>::debug_logger;

	void initData(const Eigen::MatrixXd& y, BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT) override {
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

template <bool isPath = false, bool isUpdate = true>
class MatMniwExpandForecastRun : public MatMniwOutForecastRun<isPath, isUpdate> {
public:
	MatMniwExpandForecastRun(
		const Eigen::MatrixXd& y, int num_data, int lag,
		int num_chains, int num_iter, int num_burn, int thin, BVHAR_LIST& fit_record,
		BVHAR_LIST& param_coef_sig, BVHAR_LIST_OF_LIST& coef_sig_init,
		BVHAR_LIST& row_prior, BVHAR_LIST_OF_LIST& row_init, const int row_prior_type,
		BVHAR_LIST& col_prior, BVHAR_LIST_OF_LIST& col_init, const int col_prior_type,
		int step, const Eigen::MatrixXd& y_test, bool get_lpl, bool use_fit,
		const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads,
		BVHAR_OPTIONAL<BVHAR_LIST> row_exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> row_exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> row_exogen_prior_type = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<BVHAR_LIST> col_exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> col_exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> col_exogen_prior_type = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<BVHAR_LIST> row_factor_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> row_factor_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> row_factor_prior_type = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> nrow_factor = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<BVHAR_LIST> col_factor_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> col_factor_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> col_factor_prior_type = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> ncol_factor = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<int> factor_lag = BVHAR_NULLOPT
	)
	: MatMniwOutForecastRun<isPath, isUpdate>(
			y, num_data, lag,
			num_chains, num_iter, num_burn, thin, fit_record,
			param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
			step, y_test, get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads,
			row_exogen_prior, row_exogen_init, row_exogen_prior_type,
			col_exogen_prior, col_exogen_init, col_exogen_prior_type,
			exogen, exogen_lag,
			row_factor_prior, row_factor_init, row_factor_prior_type, nrow_factor,
			col_factor_prior, col_factor_init, col_factor_prior_type, ncol_factor,
			factor_lag
		) {
		BVHAR_DEBUG_LOG(debug_logger, "MatMniwExpandForecastRun constructor");
		initialize(
			y, fit_record,
			param_coef_sig, coef_sig_init,
			row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
			seed_chain,
			row_exogen_prior, row_exogen_init, row_exogen_prior_type,
			col_exogen_prior, col_exogen_init, col_exogen_prior_type,
			exogen, exogen_lag,
			row_factor_prior, row_factor_init, row_factor_prior_type,
			col_factor_prior, col_factor_init, col_factor_prior_type
		);
	}
	virtual ~MatMniwExpandForecastRun() = default;

protected:
	using MatMniwOutForecastRun<isPath, isUpdate>::num_window;
	using MatMniwOutForecastRun<isPath, isUpdate>::num_row;
	using MatMniwOutForecastRun<isPath, isUpdate>::num_col;
	using MatMniwOutForecastRun<isPath, isUpdate>::num_test;
	using MatMniwOutForecastRun<isPath, isUpdate>::num_horizon;
	using MatMniwOutForecastRun<isPath, isUpdate>::step;
	using MatMniwOutForecastRun<isPath, isUpdate>::roll_mat;
	using MatMniwOutForecastRun<isPath, isUpdate>::y_test;
	using MatMniwOutForecastRun<isPath, isUpdate>::initialize;
	using MatMniwOutForecastRun<isPath, isUpdate>::roll_exogen_mat;
	using MatMniwOutForecastRun<isPath, isUpdate>::roll_exogen;
	using MatMniwOutForecastRun<isPath, isUpdate>::lag_exogen;
	using MatMniwOutForecastRun<isPath, isUpdate>::debug_logger;

	void initData(const Eigen::MatrixXd& y, BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT) override {
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

template <template <bool, bool> class BaseOutForecast = MatMniwRollForecastRun>
inline std::unique_ptr<bvhar::McmcOutforecastInterface> initialize_matmniwoutforecaster(
	const Eigen::MatrixXd& y, int num_data, int lag,
	int num_chains, int num_iter, int num_burn, int thin, BVHAR_LIST& fit_record,
	bool run_mcmc,
	BVHAR_LIST& param_coef_sig, BVHAR_LIST_OF_LIST& coef_sig_init,
	BVHAR_LIST& row_prior, BVHAR_LIST_OF_LIST& row_init, const int row_prior_type,
	BVHAR_LIST& col_prior, BVHAR_LIST_OF_LIST& col_init, const int col_prior_type,
	int step, const Eigen::MatrixXd& y_test, bool get_lpl, bool use_fit,
	const Eigen::MatrixXi& seed_chain, const Eigen::VectorXi& seed_forecast, bool display_progress, int nthreads,
	BVHAR_OPTIONAL<BVHAR_LIST> row_exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> row_exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> row_exogen_prior_type = BVHAR_NULLOPT,
	BVHAR_OPTIONAL<BVHAR_LIST> col_exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> col_exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> col_exogen_prior_type = BVHAR_NULLOPT,
	BVHAR_OPTIONAL<Eigen::MatrixXd> exogen = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT,
	BVHAR_OPTIONAL<BVHAR_LIST> row_factor_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> row_factor_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> row_factor_prior_type = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> nrow_factor = BVHAR_NULLOPT,
	BVHAR_OPTIONAL<BVHAR_LIST> col_factor_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> col_factor_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> col_factor_prior_type = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> ncol_factor = BVHAR_NULLOPT,
	BVHAR_OPTIONAL<int> factor_lag = BVHAR_NULLOPT
) {
	if (run_mcmc) {
		return std::make_unique<BaseOutForecast<true, true>>(
			y, num_data, lag, num_chains, num_iter, num_burn, thin, fit_record,
			param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
			step, y_test, get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads,
			row_exogen_prior, row_exogen_init, row_exogen_prior_type,
			col_exogen_prior, col_exogen_init, col_exogen_prior_type,
			exogen, exogen_lag,
			row_factor_prior, row_factor_init, row_factor_prior_type, nrow_factor,
			col_factor_prior, col_factor_init, col_factor_prior_type, ncol_factor,
			factor_lag
		);
	}
	return std::make_unique<BaseOutForecast<true, false>>(
		y, num_data, lag, num_chains, num_iter, num_burn, thin, fit_record,
		param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
		step, y_test, get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads,
		row_exogen_prior, row_exogen_init, row_exogen_prior_type,
		col_exogen_prior, col_exogen_init, col_exogen_prior_type,
		exogen, exogen_lag,
		row_factor_prior, row_factor_init, row_factor_prior_type, nrow_factor,
		col_factor_prior, col_factor_init, col_factor_prior_type, ncol_factor,
		factor_lag
	);
}

} // namespace baymar
} // namespace baecon

#endif // BAYMAR_BAYES_MNIW_FORECASTER_H