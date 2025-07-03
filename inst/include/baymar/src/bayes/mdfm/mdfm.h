#ifndef BAYMAR_BAYES_MDFM_MDFM_H
#define BAYMAR_BAYES_MDFM_MDFM_H

#include "./config.h"
#include "../shrinkage/shrinkage.h"

namespace baymar {

class McmcMatDfm;
class McmcMatDfmVar;

class McmcMatDfm : public bvhar::McmcAlgo {
public:
	McmcMatDfm(
		const MatDfmParams& params, const MatMniwInits& inits,
		std::unique_ptr<MatShrinkageUpdater>& row_updater, std::unique_ptr<MatShrinkageUpdater>& col_updater,
		unsigned int seed
	)
	: bvhar::McmcAlgo(params, seed),
		nrow_factor(params._nrow_factor), ncol_factor(params._ncol_factor), size_factor(params._size_factor),
		lag(params._lag), num_design(params._design),
		num_row(params._row), num_col(params._col),
		nrow_row_coef(params._row_row_coef), nrow_col_coef(params._row_col_coef),
		y(params._y), /*x(num_design),*/ factor_mat(num_design),
		row_updater(std::move(row_updater)), col_updater(std::move(col_updater)),
		row_coef(inits._init_row_coef), row_sig_lower(inits._init_row_lower),
		col_coef(inits._init_col_coef), col_sig_lower(inits._init_col_lower),
		mdfm_record(std::make_unique<MatDfmRecords>(num_iter, num_row, num_col, nrow_row_coef, nrow_col_coef, num_design, size_factor)),
		row_prior_mean(params._row_mean), row_iw_scl(params._row_iw_scl),
		col_prior_mean(params._col_mean), col_iw_scl(params._col_iw_scl),
		row_prior_prec(params._row_prec), col_prior_prec(params._col_prec),
		row_iw_df(params._row_iw_df), col_iw_df(params._col_iw_df) {
		BVHAR_DEBUG_LOG(
			debug_logger,
			"McmcMatDfm Constructor: nrow_factor: {}, ncol_factor: {}, lag: {}, num_design: {}",
			nrow_factor, ncol_factor, lag, num_design
		);
	}
	virtual ~McmcMatDfm() = default;

	void doWarmUp() override {
		BVHAR_DEBUG_LOG(debug_logger, "doWarmUp() called");
		std::lock_guard<std::mutex> lock(mtx);
		updatePrec();
		updateFactor();
		// updateDesign();
		updateCoefCov();
	}

	void doPosteriorDraws() override {
		BVHAR_DEBUG_LOG(debug_logger, "doPosteriorDraws() called");
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updatePrec();
		updateFactor();
		// updateDesign();
		updateCoefCov();
	}

protected:
	int nrow_factor, ncol_factor, size_factor, lag, num_design;
	int num_row, num_col, nrow_row_coef, nrow_col_coef;
	std::vector<Eigen::MatrixXd> y;
	// std::vector<Eigen::SparseMatrix<double>> x;
	std::vector<Eigen::MatrixXd> factor_mat;
	std::unique_ptr<MatShrinkageUpdater> row_updater;
	std::unique_ptr<MatShrinkageUpdater> col_updater;
	Eigen::MatrixXd row_coef, row_sig_lower, col_coef, col_sig_lower;
	std::unique_ptr<MatDfmRecords> mdfm_record;
	Eigen::MatrixXd row_prior_mean, row_iw_scl;
	Eigen::MatrixXd col_prior_mean, col_iw_scl;
	Eigen::VectorXd row_prior_prec, col_prior_prec;
	double row_iw_df, col_iw_df;

	virtual void updateFactor() {}
	
	void updatePrec() {
		BVHAR_DEBUG_LOG(debug_logger, "updatePrec() called");
		row_updater->updatePrec(
			row_prior_prec,
			row_coef,
			row_sig_lower,
			row_prior_mean,
			rng
		);
		col_updater->updatePrec(
			col_prior_prec,
			col_coef,
			col_sig_lower,
			col_prior_mean,
			rng
		);
	}

	void updateCoefCov() {
		BVHAR_DEBUG_LOG(debug_logger, "updateCoefCov() called");
		draw_coef_sig<true, Eigen::MatrixXd>(
			row_coef, row_sig_lower,
			col_coef, col_sig_lower,
			row_prior_mean, row_prior_prec, row_iw_scl, row_iw_df,
			num_design, num_col,
			factor_mat, y, rng
		);
		draw_coef_sig<false, Eigen::MatrixXd>(
			col_coef, col_sig_lower,
			row_coef, row_sig_lower,
			col_prior_mean, col_prior_prec, col_iw_scl, col_iw_df,
			num_design, num_row,
			factor_mat, y, rng
		);
	}

private:
	// void updateDesign() {
	// 	x = build_mar_design(factor_mat, lag);
	// }
};

class McmcMatDfmVar : public McmcMatDfm {
public:
	McmcMatDfmVar(
		const MatDfmVarParams& params, const MatDfmVarInits& inits,
		std::unique_ptr<MatShrinkageUpdater>& row_updater, std::unique_ptr<MatShrinkageUpdater>& col_updater,
		unsigned int seed
	)
	: McmcMatDfm(params, inits, row_updater, col_updater, seed),
		ig_shp(params._sig_shp), ig_scl(params._sig_scl),
		prior_mean(params._mean), prior_prec(params._prec),
		dfm_coef(inits._init_factor_coef), dfm_prec(inits._init_factor_prec) {}
	virtual ~McmcMatDfmVar() = default;

protected:
	void updateFactor() override {
		draw_dfm_factor(
			factor_mat, lag, nrow_factor, ncol_factor, dfm_coef, dfm_prec,
			row_coef, row_sig_lower,
			col_coef, col_sig_lower,
			y, rng
		);
		draw_dfm_prec(dfm_prec, lag, ig_shp, ig_scl, factor_mat, dfm_coef, rng);
		draw_dfm_coef(dfm_coef, dfm_prec, prior_mean, prior_prec, factor_mat, lag, rng);
	}

private:
	Eigen::VectorXd ig_shp, ig_scl;
	Eigen::VectorXd prior_mean, prior_prec;
	Eigen::MatrixXd dfm_coef; // p1*p2 x s
	Eigen::VectorXd dfm_prec; // lambda_{1, 1}, ..., lambda_{p1, p2}
};

} // namespace baymar

#endif // BAYMAR_BAYES_MDFM_MDFM_H