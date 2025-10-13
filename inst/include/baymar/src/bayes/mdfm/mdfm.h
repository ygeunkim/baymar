#ifndef BAYMAR_BAYES_MDFM_MDFM_H
#define BAYMAR_BAYES_MDFM_MDFM_H

#include "../mniw/config.h"
#include "./augment.h"
// #include "../shrinkage/shrinkage.h"

namespace baymar {

class McmcMatDfm;
// class McmcMatDfmVar;
class MatDfmRun;

class McmcMatDfm : public bvhar::McmcAlgo {
public:
	// McmcMatDfm(
	// 	const MatMniwParams& mniw_params, const MatMniwInits& mniw_inits,
	// 	const MatDfmParams& params,
	// 	std::unique_ptr<MatShrinkageUpdater>& row_updater, std::unique_ptr<MatShrinkageUpdater>& col_updater,
	// 	unsigned int seed
	// )
	// : bvhar::McmcAlgo(mniw_params, seed),
	// 	nrow_factor(params._nrow_factor), ncol_factor(params._ncol_factor), size_factor(params._size_factor),
	// 	lag(params._lag), num_design(mniw_params._design),
	// 	num_row(mniw_params._row), num_col(mniw_params._col),
	// 	nrow_row_coef(mniw_params._row_row_coef), nrow_col_coef(mniw_params._row_col_coef),
	// 	y(mniw_params._y), /*x(num_design),*/ factor_mat(num_design),
	// 	row_updater(std::move(row_updater)), col_updater(std::move(col_updater)),
	// 	row_coef(mniw_inits._init_row_coef), row_sig_lower(mniw_inits._init_row_lower),
	// 	col_coef(mniw_inits._init_col_coef), col_sig_lower(mniw_inits._init_col_lower),
	// 	mniw_record(std::make_unique<MatMniwRecords>(num_iter, num_row, num_col, row_coef.rows(), col_coef.rows())),
	// 	// mdfm_record(std::make_unique<MatDfmRecords>(num_iter, num_row, num_col, nrow_row_coef, nrow_col_coef, num_design, size_factor)),
	// 	row_prior_mean(mniw_params._row_mean), row_iw_scl(mniw_params._row_iw_scl),
	// 	col_prior_mean(mniw_params._col_mean), col_iw_scl(mniw_params._col_iw_scl),
	// 	row_prior_prec(mniw_params._row_prec), col_prior_prec(mniw_params._col_prec),
	// 	row_iw_df(mniw_params._row_iw_df), col_iw_df(mniw_params._col_iw_df) {
	// 	BVHAR_DEBUG_LOG(
	// 		debug_logger,
	// 		"McmcMatDfm Constructor: row_coef: {} x {}, row_sig_lower: {} x {}, row_prior_mean: {} x {}, row_prior_prec: {}",
	// 		row_coef.rows(), row_coef.cols(), row_sig_lower.rows(), row_sig_lower.cols(),
	// 		row_prior_mean.rows(), row_prior_mean.cols(), row_prior_prec.size()
	// 	);
	// 	BVHAR_DEBUG_LOG(
	// 		debug_logger,
	// 		"McmcMatDfm Constructor: col_coef: {} x {}, col_sig_lower: {} x {}, col_prior_mean: {} x {}, col_prior_prec: {}",
	// 		col_coef.rows(), col_coef.cols(), col_sig_lower.rows(), col_sig_lower.cols(),
	// 		col_prior_mean.rows(), col_prior_mean.cols(), col_prior_prec.size()
	// 	);
	// 	BVHAR_DEBUG_LOG(
	// 		debug_logger,
	// 		"McmcMatDfm Constructor: nrow_factor: {}, ncol_factor: {}, lag: {}, num_design: {}",
	// 		nrow_factor, ncol_factor, lag, num_design
	// 	);
	// }
	McmcMatDfm(
		const MatMniwParams& mniw_params, const MatMniwInits& mniw_inits,
		std::unique_ptr<MatFactorAugmenter>& factor_updater,
		std::unique_ptr<MatShrinkageUpdater>& row_updater, std::unique_ptr<MatShrinkageUpdater>& col_updater,
		unsigned int seed
	)
	: bvhar::McmcAlgo(mniw_params, seed),
		num_design(mniw_params._design), num_row(mniw_params._row), num_col(mniw_params._col),
		nrow_row_coef(mniw_params._row_row_coef), nrow_col_coef(mniw_params._row_col_coef),
		y(mniw_params._y),
		factor_updater(std::move(factor_updater)),
		row_updater(std::move(row_updater)), col_updater(std::move(col_updater)),
		row_coef(mniw_inits._init_row_coef), row_sig_lower(mniw_inits._init_row_lower),
		col_coef(mniw_inits._init_col_coef), col_sig_lower(mniw_inits._init_col_lower),
		mniw_record(std::make_unique<MatMniwRecords>(num_iter, num_row, num_col, row_coef.rows(), col_coef.rows())),
		row_prior_mean(mniw_params._row_mean), row_iw_scl(mniw_params._row_iw_scl),
		col_prior_mean(mniw_params._col_mean), col_iw_scl(mniw_params._col_iw_scl),
		row_prior_prec(mniw_params._row_prec), col_prior_prec(mniw_params._col_prec),
		row_iw_df(mniw_params._row_iw_df), col_iw_df(mniw_params._col_iw_df) {}
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
		updateRecords();
	}

	BVHAR_LIST returnRecords(int num_burn, int thin) override {
		BVHAR_DEBUG_LOG(debug_logger, "returnRecords(num_burn={}, thin={}) called", num_burn, thin);
		// BVHAR_LIST res = mdfm_record->returnListRecords(nrow_row_coef, num_row, nrow_col_coef, num_col, num_design, size_factor);
		BVHAR_LIST res = mniw_record->returnListRecords(nrow_row_coef, num_row, 0, 0, nrow_col_coef, num_col, 0, 0);
		factor_updater->appendRecords(res);
		// mdfm_record->appendRecords(res);
		row_updater->appendRowRecords(res);
		col_updater->appendColRecords(res);
		for (auto& record : res) {
			if (BVHAR_IS_MATRIX(BVHAR_ACCESS_LIST(record, res))) {
				BVHAR_ACCESS_LIST(record, res) = bvhar::thin_record(BVHAR_CAST<Eigen::MatrixXd>(BVHAR_ACCESS_LIST(record, res)), num_iter, num_burn, thin);
			} else {
				BVHAR_ACCESS_LIST(record, res) = bvhar::thin_record(BVHAR_CAST<Eigen::VectorXd>(BVHAR_ACCESS_LIST(record, res)), num_iter, num_burn, thin);
			}
		}
		return res;
	}

	MatMniwRecords returnMniwRecords(int num_burn, int thin) const {
		return mniw_record->returnRecords<MatMniwRecords>(num_iter, num_burn, thin);
	}

	template <typename RecordType>
	RecordType returnStructRecords(int num_burn, int thin) const {
		return factor_updater->returnStructRecords<RecordType>(num_burn, thin);
		// return mdfm_record->returnRecords<RecordType>(num_iter, num_burn, thin);
	}

protected:
	// int nrow_factor, ncol_factor, size_factor, lag;
	int num_design, num_row, num_col, nrow_row_coef, nrow_col_coef;
	std::vector<Eigen::MatrixXd> y;
	// std::vector<Eigen::SparseMatrix<double>> x;
	std::unique_ptr<MatFactorAugmenter> factor_updater;
	// std::vector<Eigen::MatrixXd> factor_mat;
	std::unique_ptr<MatShrinkageUpdater> row_updater;
	std::unique_ptr<MatShrinkageUpdater> col_updater;
	Eigen::MatrixXd row_coef, row_sig_lower, col_coef, col_sig_lower;
	std::unique_ptr<MatMniwRecords> mniw_record;
	// std::unique_ptr<MatDfmRecords> mdfm_record;
	Eigen::MatrixXd row_prior_mean, row_iw_scl;
	Eigen::MatrixXd col_prior_mean, col_iw_scl;
	Eigen::VectorXd row_prior_prec, col_prior_prec;
	double row_iw_df, col_iw_df;

	virtual void updateFactor() {
		factor_updater->updateFactor(
			row_coef, row_sig_lower,
			col_coef, col_sig_lower,
			y, rng
		);
	}
	
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
		// draw_coef_sig<true, Eigen::MatrixXd>(
		// 	row_coef, row_sig_lower,
		// 	col_coef, col_sig_lower,
		// 	row_prior_mean, row_prior_prec, row_iw_scl, row_iw_df,
		// 	num_design, num_col,
		// 	factor_updater->getFactor(), y, rng
		// );
		// draw_coef_sig<false, Eigen::MatrixXd>(
		// 	col_coef, col_sig_lower,
		// 	row_coef, row_sig_lower,
		// 	col_prior_mean, col_prior_prec, col_iw_scl, col_iw_df,
		// 	num_design, num_row,
		// 	factor_updater->getFactor(), y, rng
		// );
		factor_updater->updateCoefCov<true>(
			row_coef, row_sig_lower,
			col_coef, col_sig_lower,
			row_prior_mean, row_prior_prec, row_iw_scl, row_iw_df,
			num_col, y, rng
		);
		factor_updater->updateCoefCov<false>(
			col_coef, col_sig_lower,
			row_coef, row_sig_lower,
			col_prior_mean, col_prior_prec, col_iw_scl, col_iw_df,
			num_row, y, rng
		);
	}

	void updateRecords() {
		BVHAR_DEBUG_LOG(debug_logger, "updateRecords() called");
		mniw_record->assignRecords(
			mcmc_step, row_coef, row_sig_lower, col_coef, col_sig_lower,
			nrow_row_coef, num_row, 0, 0,
			nrow_col_coef, num_col, 0, 0
		);
		factor_updater->updateRecords(mcmc_step);
		row_updater->updateRecords(mcmc_step);
		col_updater->updateRecords(mcmc_step);
		// updateDfmRecords();
	}

	// virtual void updateDfmRecords() = 0;
};

// class McmcMatDfmVar : public McmcMatDfm {
// public:
// 	McmcMatDfmVar(
// 		const MatMniwParams& mniw_params, const MatMniwInits& mniw_inits,
// 		const MatDfmVarParams& params, const MatDfmVarInits& inits,
// 		std::unique_ptr<MatShrinkageUpdater>& row_updater, std::unique_ptr<MatShrinkageUpdater>& col_updater,
// 		unsigned int seed
// 	)
// 	: McmcMatDfm(mniw_params, mniw_inits, params, row_updater, col_updater, seed),
// 		ig_shp(params._sig_shp), ig_scl(params._sig_scl),
// 		prior_mean(params._mean), prior_prec(params._prec),
// 		factor_coef(inits._init_factor_coef), factor_sig(inits._init_factor_prec) {
// 		BVHAR_DEBUG_LOG(
// 			debug_logger,
// 			"McmcMatDfmVar Constructor: factor_coef: {} x {}, factor_sig: {}, ig_shp: {}, ig_scl: {}, prior_mean: {}, prior_prec: {}",
// 			factor_coef.rows(), factor_coef.cols(), factor_sig.size(),
// 			ig_shp.size(), ig_scl.size(),
// 			prior_mean.size(), prior_prec.size()
// 		);
// 		mdfm_record = std::make_unique<MatDfmVarRecords>(num_iter, num_design, size_factor, lag);
// 	}
// 	virtual ~McmcMatDfmVar() = default;

// protected:
// 	void updateFactor() override {
// 		BVHAR_DEBUG_LOG(debug_logger, "updateFactor() called");
// 		draw_dfm_factor(
// 			factor_mat, lag, nrow_factor, ncol_factor, factor_coef, factor_sig,
// 			row_coef.transpose(), row_sig_lower,
// 			col_coef.transpose(), col_sig_lower,
// 			y, rng
// 		);
// 		draw_dfm_sig(factor_sig, lag, ig_shp, ig_scl, factor_mat, factor_coef, rng);
// 		draw_dfm_coef(factor_coef, factor_sig, prior_mean, prior_prec, factor_mat, lag, rng);
// 	}

// 	void updateDfmRecords() override {
// 		BVHAR_DEBUG_LOG(debug_logger, "updateDfmRecords() called");
// 		mdfm_record->assignRecords(
// 			mcmc_step,
// 			factor_mat, factor_coef, factor_sig,
// 			num_design, size_factor
// 		);
// 	}

// private:
// 	Eigen::VectorXd ig_shp, ig_scl;
// 	Eigen::VectorXd prior_mean, prior_prec;
// 	Eigen::MatrixXd factor_coef; // p1*p2 x s
// 	Eigen::VectorXd factor_sig; // lambda_{1, 1}, ..., lambda_{p1, p2}
// };

// template <typename BaseDfm = McmcMatDfmVar>
inline std::vector<std::unique_ptr<McmcMatDfm>> initialize_matdfm(
	int num_chains, int num_iter,
	std::vector<Eigen::MatrixXd>& y, int factor_lag,
	BVHAR_LIST& param_dfm, BVHAR_LIST_OF_LIST& dfm_init,
	BVHAR_LIST& row_prior, BVHAR_LIST_OF_LIST& row_init, const int row_prior_type,
	BVHAR_LIST& col_prior, BVHAR_LIST_OF_LIST& col_init, const int col_prior_type,
  Eigen::Ref<const Eigen::VectorXi> seed_chain
) {
	// using PARAMS = typename std::conditional<std::is_same<BaseDfm, McmcMatDfmVar>::value, MatDfmVarParams, MatDfmParams>::type;
	// using INITS = typename std::conditional<std::is_same<BaseDfm, McmcMatDfmVar>::value, MatDfmVarInits, MatMniwInits>::type;
	MatMniwParams mniw_params(num_iter, y, param_dfm);
	MatDfmVarParams params(param_dfm);
	std::unique_ptr<MatFactorAugmenter> factor_updater;
	std::vector<std::unique_ptr<McmcMatDfm>> mcmc_ptr(num_chains);
	for (int i = 0; i < num_chains; ++i) {
		BVHAR_LIST row_init_spec = row_init[i];
		BVHAR_LIST col_init_spec = col_init[i];
		auto row_updater = initialize_matshrinkageupdater(num_iter, row_prior, row_init_spec, row_prior_type);
		auto col_updater = initialize_matshrinkageupdater(num_iter, col_prior, col_init_spec, col_prior_type);
		row_updater->initPrec(mniw_params._row_prec.head(mniw_params._row_row_coef));
		col_updater->initPrec(mniw_params._col_prec.head(mniw_params._row_col_coef));
		BVHAR_LIST init_spec = dfm_init[i];
		MatMniwInits mniw_inits(init_spec);
		MatDfmVarInits inits(init_spec);
		factor_updater = std::make_unique<MatFactorVarAugmenter>(num_iter, y.size(), params, inits);
		mcmc_ptr[i] = std::make_unique<McmcMatDfm>(mniw_params, mniw_inits, factor_updater, row_updater, col_updater, static_cast<unsigned int>(seed_chain[i]));
	}
	return mcmc_ptr;
}

class MatDfmRun : public bvhar::McmcRun {
public:
	MatDfmRun(
		int num_chains, int num_iter, int num_burn, int thin,
		std::vector<Eigen::MatrixXd>& y, int factor_lag,
		BVHAR_LIST& param_dfm, BVHAR_LIST_OF_LIST& dfm_init,
		BVHAR_LIST& row_prior, BVHAR_LIST_OF_LIST& row_init, const int row_prior_type,
		BVHAR_LIST& col_prior, BVHAR_LIST_OF_LIST& col_init, const int col_prior_type,
		Eigen::Ref<const Eigen::VectorXi> seed_chain, bool display_progress, int nthreads
	)
	: bvhar::McmcRun(num_chains, num_iter, num_burn, thin, display_progress, nthreads) {
		auto temp_mcmc = initialize_matdfm(
			num_chains, num_iter - num_burn, y, factor_lag, param_dfm, dfm_init,
			row_prior, row_init, row_prior_type,
			col_prior, col_init, col_prior_type,
			seed_chain
		);
		for (int i = 0; i < num_chains; ++i) {
			mcmc_ptr[i] = std::move(temp_mcmc[i]);
		}
	}
	virtual ~MatDfmRun() = default;
};

} // namespace baymar

#endif // BAYMAR_BAYES_MDFM_MDFM_H