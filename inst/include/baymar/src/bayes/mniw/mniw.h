#ifndef BAYMAR_BAYES_MNIW_MNIW_H
#define BAYMAR_BAYES_MNIW_MNIW_H

#include "./config.h"
#include "../mdfm/augment.h"

namespace baymar {

class McmcMatMniw;

class McmcMatMniw : public bvhar::McmcAlgo {
public:
	McmcMatMniw(
		const MatMniwRegParams& params, const MatMniwInits& inits,
		std::unique_ptr<MatShrinkageUpdater>& row_updater, std::unique_ptr<MatShrinkageUpdater>& col_updater,
		unsigned int seed,
		BVHAR_OPTIONAL<std::unique_ptr<MatShrinkageUpdater>> row_exogen = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<std::unique_ptr<MatShrinkageUpdater>> col_exogen = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<std::unique_ptr<MatFactorAugmenter>> famar = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<std::unique_ptr<MatShrinkageUpdater>> row_factor = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<std::unique_ptr<MatShrinkageUpdater>> col_factor = BVHAR_NULLOPT
	)
	: bvhar::McmcAlgo(params, seed),
		x(params._x), y(params._y),
		row_updater(std::move(row_updater)), col_updater(std::move(col_updater)),
		num_row(params._row), num_col(params._col), num_design(params._design),
		nrow_row_coef(params._row_row_coef), nrow_col_coef(params._row_col_coef),
		nrow_row_exogen(params._row_exogen), nrow_col_exogen(params._col_exogen),
		nrow_factor(params._row_factor), ncol_factor(params._col_factor),
		row_record(num_iter + 1, std::vector<Eigen::MatrixXd>(2)), col_record(num_iter + 1, std::vector<Eigen::MatrixXd>(2)),
		row_coef(inits._init_row_coef), row_sig_lower(inits._init_row_lower),
		col_coef(inits._init_col_coef), col_sig_lower(inits._init_col_lower),
		mniw_record(std::make_unique<MatMniwRecords>(num_iter, num_row, num_col, row_coef.rows(), col_coef.rows())),
		row_prior_mean(params._row_mean), row_iw_scl(params._row_iw_scl),
		col_prior_mean(params._col_mean), col_iw_scl(params._col_iw_scl),
		row_prior_prec(params._row_prec), col_prior_prec(params._col_prec),
		row_iw_df(params._row_iw_df), col_iw_df(params._col_iw_df) {
		BVHAR_DEBUG_LOG(
			debug_logger,
			"McmcMatMniw Constructor: row_coef: {} x {}, row_sig_lower: {} x {}, row_prior_mean: {} x {}, row_prior_prec: {}",
			row_coef.rows(), row_coef.cols(), row_sig_lower.rows(), row_sig_lower.cols(),
			row_prior_mean.rows(), row_prior_mean.cols(), row_prior_prec.size()
		);
		BVHAR_DEBUG_LOG(
			debug_logger,
			"McmcMatMniw Constructor: col_coef: {} x {}, col_sig_lower: {} x {}, col_prior_mean: {} x {}, col_prior_prec: {}",
			col_coef.rows(), col_coef.cols(), col_sig_lower.rows(), col_sig_lower.cols(),
			col_prior_mean.rows(), col_prior_mean.cols(), col_prior_prec.size()
		);
		if (row_exogen) {
			exogen_row_updater = std::move(*row_exogen);
		}
		if (col_exogen) {
			exogen_col_updater = std::move(*col_exogen);
		}
		updateRecords();
		if (row_factor) {
			factor_row_updater = std::move(*row_factor);
		}
		if (col_factor) {
			factor_col_updater = std::move(*col_factor);
		}
		if (famar) {
			famar_updater = std::move(*famar);
			// nrow_factor = famar_updater->getRow();
			// ncol_factor = famar_updater->getCol();
			// nrow_row_coef -= nrow_factor;
			// nrow_col_coef -= ncol_factor;
		}
	}
	virtual ~McmcMatMniw() = default;
	
	void doWarmUp() override {
		BVHAR_DEBUG_LOG(debug_logger, "doWarmUp() called");
		std::lock_guard<std::mutex> lock(mtx);
		updatePrec();
		updateCoefCov();
	}

	void doPosteriorDraws() override {
		BVHAR_DEBUG_LOG(debug_logger, "doPosteriorDraws() called");
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updatePrec();
		updateCoefCov();
		updateRecords();
	}

	BVHAR_LIST returnRecords(int num_burn, int thin) override {
		BVHAR_DEBUG_LOG(debug_logger, "returnRecords(num_burn={}, thin={}) called", num_burn, thin);
		BVHAR_LIST res = mniw_record->returnListRecords(nrow_row_coef, num_row, nrow_row_exogen, nrow_factor, nrow_col_coef, num_col, nrow_col_exogen, ncol_factor);
		if (famar_updater) {
			famar_updater->appendRecords(res);
		}
		for (auto& record : res) {
			if (BVHAR_IS_MATRIX(BVHAR_ACCESS_LIST(record, res))) {
				BVHAR_ACCESS_LIST(record, res) = bvhar::thin_record(BVHAR_CAST<Eigen::MatrixXd>(BVHAR_ACCESS_LIST(record, res)), num_iter, num_burn, thin);
			} else {
				BVHAR_ACCESS_LIST(record, res) = bvhar::thin_record(BVHAR_CAST<Eigen::VectorXd>(BVHAR_ACCESS_LIST(record, res)), num_iter, num_burn, thin);
			}
		}
		return res;
	}

	MatMniwRecords returnStructRecords(int num_burn, int thin) const {
		BVHAR_DEBUG_LOG(debug_logger, "returnStructRecords(num_burn={}, thin={}) called", num_burn, thin);
		return mniw_record->returnRecords<MatMniwRecords>(num_iter, num_burn, thin);
	}

	template <typename RecordType>
	RecordType returnFactorRecords(int num_burn, int thin) const {
		BVHAR_DEBUG_LOG(debug_logger, "returnFactorRecords(num_burn={}, thin={}) called", num_burn, thin);
		return famar_updater->returnStructRecords<RecordType>(num_burn, thin);
	}

protected:
	std::vector<Eigen::SparseMatrix<double>> x;
	std::vector<Eigen::MatrixXd> y;
	std::unique_ptr<MatShrinkageUpdater> row_updater;
	std::unique_ptr<MatShrinkageUpdater> col_updater;
	std::unique_ptr<MatShrinkageUpdater> exogen_row_updater;
	std::unique_ptr<MatShrinkageUpdater> exogen_col_updater;
	std::unique_ptr<MatShrinkageUpdater> factor_row_updater;
	std::unique_ptr<MatShrinkageUpdater> factor_col_updater;
	std::unique_ptr<MatFactorAugmenter> famar_updater;
	int num_row;
	int num_col;
	int num_design;
	int nrow_row_coef, nrow_col_coef, nrow_row_exogen, nrow_col_exogen, nrow_factor, ncol_factor;
	std::vector<std::vector<Eigen::MatrixXd>> row_record, col_record;
	Eigen::MatrixXd row_coef, row_sig_lower, col_coef, col_sig_lower;
	std::unique_ptr<MatMniwRecords> mniw_record;
	Eigen::MatrixXd row_prior_mean, row_iw_scl;
	Eigen::MatrixXd col_prior_mean, col_iw_scl;
	Eigen::VectorXd row_prior_prec, col_prior_prec;
	double row_iw_df, col_iw_df;

	void updatePrec() {
		BVHAR_DEBUG_LOG(debug_logger, "updatePrec() called");
		row_updater->updatePrec(
			row_prior_prec.head(nrow_row_coef),
			row_coef.topRows(nrow_row_coef),
			row_sig_lower,
			row_prior_mean.topRows(nrow_row_coef),
			rng
		);
		col_updater->updatePrec(
			col_prior_prec.head(nrow_col_coef),
			col_coef.topRows(nrow_col_coef),
			col_sig_lower,
			col_prior_mean.topRows(nrow_col_coef),
			rng
		);
		if (exogen_row_updater) {
			exogen_row_updater->updatePrec(
				row_prior_prec.segment(nrow_row_coef, nrow_row_exogen),
				row_coef.middleRows(nrow_row_coef, nrow_row_exogen),
				row_sig_lower,
				row_prior_mean.middleRows(nrow_row_coef, nrow_row_exogen),
				rng
			);
		}
		if (exogen_col_updater) {
			exogen_col_updater->updatePrec(
				col_prior_prec.segment(nrow_col_coef, nrow_col_exogen),
				col_coef.middleRows(nrow_col_coef, nrow_col_exogen),
				col_sig_lower,
				col_prior_mean.middleRows(nrow_col_coef, nrow_col_exogen),
				rng
			);
		}
		if (factor_row_updater) {
			factor_row_updater->updatePrec(
				row_prior_prec.tail(nrow_factor),
				row_coef.bottomRows(nrow_factor),
				row_sig_lower,
				row_prior_mean.bottomRows(nrow_factor),
				rng
			);
		}
		if (factor_col_updater) {
			factor_col_updater->updatePrec(
				col_prior_prec.tail(ncol_factor),
				col_coef.bottomRows(ncol_factor),
				col_sig_lower,
				col_prior_mean.bottomRows(ncol_factor),
				rng
			);
		}
	}

	void updateCoefCov() {
		BVHAR_DEBUG_LOG(debug_logger, "updateCoefCov() called");
		if (famar_updater) {
			famar_updater->updateResid(
				x, y,
				row_coef.topRows(nrow_row_coef + nrow_row_exogen), col_coef.topRows(nrow_col_coef + nrow_col_exogen)
			);
			famar_updater->updateFactor(
				row_coef.bottomRows(nrow_factor), row_sig_lower,
				col_coef.bottomRows(ncol_factor), col_sig_lower,
				rng
			);
			famar_updater->appendDesign(x);
		}
		draw_coef_sig<true>(
			row_coef, row_sig_lower,
			col_coef, col_sig_lower,
			row_prior_mean, row_prior_prec, row_iw_scl, row_iw_df,
			num_design, num_col,
			x, y, rng
		);
		draw_coef_sig<false>(
			col_coef, col_sig_lower,
			row_coef, row_sig_lower,
			col_prior_mean, col_prior_prec, col_iw_scl, col_iw_df,
			num_design, num_row,
			x, y, rng
		);
	}

	void updateRecords() {
		BVHAR_DEBUG_LOG(debug_logger, "updateRecords() called");
		// row_record[mcmc_step][0] = row_coef;
		// row_record[mcmc_step][1] = row_sig_lower * row_sig_lower.transpose();
		// col_record[mcmc_step][0] = col_coef;
		// col_record[mcmc_step][1] = col_sig_lower * col_sig_lower.transpose();
		mniw_record->assignRecords(
			mcmc_step, row_coef, row_sig_lower, col_coef, col_sig_lower,
			nrow_row_coef, num_row, nrow_row_exogen, nrow_factor,
			nrow_col_coef, num_col, nrow_col_exogen, ncol_factor
		);
		if (famar_updater) {
			famar_updater->updateRecords(mcmc_step);
		}
	}
};

inline std::vector<std::unique_ptr<McmcMatMniw>> initialize_matmcmc(
	int num_chains, int num_iter, std::vector<Eigen::SparseMatrix<double>>& x, std::vector<Eigen::MatrixXd>& y,
	BVHAR_LIST& param_coef_sig, BVHAR_LIST_OF_LIST& coef_sig_init,
	BVHAR_LIST& row_prior, BVHAR_LIST_OF_LIST& row_init, const int row_prior_type,
	BVHAR_LIST& col_prior, BVHAR_LIST_OF_LIST& col_init, const int col_prior_type,
  Eigen::Ref<const Eigen::VectorXi> seed_chain,
	BVHAR_OPTIONAL<BVHAR_LIST> row_exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> row_exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> row_exogen_prior_type = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_rows = BVHAR_NULLOPT,
	BVHAR_OPTIONAL<BVHAR_LIST> col_exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> col_exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> col_exogen_prior_type = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_cols = BVHAR_NULLOPT,
	BVHAR_OPTIONAL<BVHAR_LIST> row_factor_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> row_factor_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> row_factor_prior_type = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> nrow_factor = BVHAR_NULLOPT,
	BVHAR_OPTIONAL<BVHAR_LIST> col_factor_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> col_factor_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> col_factor_prior_type = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> ncol_factor = BVHAR_NULLOPT,
	BVHAR_OPTIONAL<int> factor_lag = BVHAR_NULLOPT
) {
	std::vector<std::unique_ptr<McmcMatMniw>> mcmc_ptr(num_chains);
	// MatMniwRegParams params(num_iter, x, y, param_coef_sig);
	MatMniwRegParams params = exogen_rows
		? (nrow_factor ? MatMniwRegParams(num_iter, x, y, param_coef_sig, *exogen_rows, *exogen_cols, *nrow_factor, *ncol_factor) : MatMniwRegParams(num_iter, x, y, param_coef_sig, *exogen_rows, *exogen_cols))
		: (nrow_factor ? MatMniwRegParams(num_iter, x, y, param_coef_sig, BVHAR_NULLOPT, BVHAR_NULLOPT, *nrow_factor, *ncol_factor) : MatMniwRegParams(num_iter, x, y, param_coef_sig));
	for (int i = 0; i < num_chains; ++i) {
		BVHAR_LIST row_init_spec = row_init[i];
		BVHAR_LIST col_init_spec = col_init[i];
		auto row_updater = initialize_matshrinkageupdater(num_iter, row_prior, row_init_spec, row_prior_type);
		auto col_updater = initialize_matshrinkageupdater(num_iter, col_prior, col_init_spec, col_prior_type);
		row_updater->initPrec(params._row_prec.head(params._row_row_coef));
		col_updater->initPrec(params._col_prec.head(params._row_col_coef));
		BVHAR_LIST init_spec = coef_sig_init[i];
		MatMniwInits inits(init_spec);
		BVHAR_OPTIONAL<std::unique_ptr<MatShrinkageUpdater>> row_exogen_updater = BVHAR_NULLOPT;
		BVHAR_OPTIONAL<std::unique_ptr<MatShrinkageUpdater>> col_exogen_updater = BVHAR_NULLOPT;
		if (row_exogen_prior_type) {
			BVHAR_LIST row_exogen_init_spec = (*row_exogen_init)[i];
			// auto temp_row_exogen_updater = initialize_matshrinkageupdater(num_iter, *row_exogen_prior, row_exogen_init_spec, *row_exogen_prior_type);
			// row_exogen_updater = std::move(temp_row_exogen_updater);
			row_exogen_updater = initialize_matshrinkageupdater(num_iter, *row_exogen_prior, row_exogen_init_spec, *row_exogen_prior_type);
			(*row_exogen_updater)->initPrec(params._row_prec.segment(params._row_row_coef, params._row_exogen));
		}
		if (col_exogen_prior_type) {
			BVHAR_LIST col_exogen_init_spec = (*col_exogen_init)[i];
			// auto temp_col_exogen_updater = initialize_matshrinkageupdater(num_iter, *col_exogen_prior, col_exogen_init_spec, *col_exogen_prior_type);
			// col_exogen_updater = std::move(temp_col_exogen_updater);
			col_exogen_updater = initialize_matshrinkageupdater(num_iter, *col_exogen_prior, col_exogen_init_spec, *col_exogen_prior_type);
			(*col_exogen_updater)->initPrec(params._col_prec.segment(params._row_col_coef, params._col_exogen));
		}
		BVHAR_OPTIONAL<std::unique_ptr<MatFactorAugmenter>> famar_updater = BVHAR_NULLOPT;
		BVHAR_OPTIONAL<std::unique_ptr<MatShrinkageUpdater>> row_factor_updater = BVHAR_NULLOPT;
		BVHAR_OPTIONAL<std::unique_ptr<MatShrinkageUpdater>> col_factor_updater = BVHAR_NULLOPT;
		if (row_factor_prior_type) {
			BVHAR_LIST row_factor_init_spec = (*row_factor_init)[i];
			row_factor_updater = initialize_matshrinkageupdater(num_iter, *row_factor_prior, row_factor_init_spec, *row_factor_prior_type);
			(*row_factor_updater)->initPrec(params._row_prec.tail(params._row_factor));
		}
		if (col_factor_prior_type) {
			BVHAR_LIST col_factor_init_spec = (*col_factor_init)[i];
			col_factor_updater = initialize_matshrinkageupdater(num_iter, *col_factor_prior, col_factor_init_spec, *col_factor_prior_type);
			(*col_factor_updater)->initPrec(params._col_prec.tail(params._col_factor));
		}
		if (nrow_factor) {
			// famar_updater = std::make_unique<MatFactorVarAugmenter>(num_iter, y.size(), *factor_lag, *nrow_factor, *ncol_factor);
			// MatDfmVarParams dfm_params(*factor_lag, *nrow_factor, *ncol_factor);
			// MatDfmVarInits dfm_inits((*nrow_factor) * (*ncol_factor), *factor_lag);
			// famar_updater = std::make_unique<MatFactorVarAugmenter>(num_iter, y.size(), dfm_params, dfm_inits);
			// BVHAR_LIST dfm_init_spec = coef_sig_init[i];
			famar_updater = initialize_factoraugmenter(num_iter, y.size(), param_coef_sig, init_spec);
		}
		mcmc_ptr[i] = std::make_unique<McmcMatMniw>(
			params, inits, row_updater, col_updater, static_cast<unsigned int>(seed_chain[i]),
			std::move(row_exogen_updater), std::move(col_exogen_updater),
			std::move(famar_updater), std::move(row_factor_updater), std::move(col_factor_updater)
		);
	}
	return mcmc_ptr;
}

class MatMcmcRun : public bvhar::McmcRun {
public:
	MatMcmcRun(
		int num_chains, int num_iter, int num_burn, int thin,
		std::vector<Eigen::SparseMatrix<double>>& x, std::vector<Eigen::MatrixXd>& y,
		BVHAR_LIST& param_coef_sig, BVHAR_LIST_OF_LIST& coef_sig_init,
		BVHAR_LIST& row_prior, BVHAR_LIST_OF_LIST& row_init, const int row_prior_type,
		BVHAR_LIST& col_prior, BVHAR_LIST_OF_LIST& col_init, const int col_prior_type,
		const Eigen::VectorXi& seed_chain, bool display_progress, int nthreads,
		BVHAR_OPTIONAL<BVHAR_LIST> row_exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> row_exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> row_exogen_prior_type = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_rows = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<BVHAR_LIST> col_exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> col_exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> col_exogen_prior_type = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_cols = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<BVHAR_LIST> row_factor_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> row_factor_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> row_factor_prior_type = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> nrow_factor = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<BVHAR_LIST> col_factor_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> col_factor_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> col_factor_prior_type = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> ncol_factor = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<int> factor_lag = BVHAR_NULLOPT
	)
	: bvhar::McmcRun(num_chains, num_iter, num_burn, thin, display_progress, nthreads) {
		auto temp_mcmc = initialize_matmcmc(
			num_chains, num_iter - num_burn, x, y,
			param_coef_sig, coef_sig_init,
			row_prior, row_init, row_prior_type,
			col_prior, col_init, col_prior_type,
			seed_chain,
			row_exogen_prior, row_exogen_init, row_exogen_prior_type, exogen_rows,
			col_exogen_prior, col_exogen_init, col_exogen_prior_type, exogen_cols,
			row_factor_prior, row_factor_init, row_factor_prior_type, nrow_factor,
			col_factor_prior, col_factor_init, col_factor_prior_type, ncol_factor,
			factor_lag
		);
		for (int i = 0; i < num_chains; ++i) {
			mcmc_ptr[i] = std::move(temp_mcmc[i]);
		}
	}
	virtual ~MatMcmcRun() = default;
};

} // namespace baymar

#endif // BAYMAR_BAYES_MNIW_MNIW_H