#ifndef BAYMAR_BAYES_MNIW_MNIW_H
#define BAYMAR_BAYES_MNIW_MNIW_H

#include "./config.h"

namespace baymar {

class McmcMatMniw;

class McmcMatMniw : public bvhar::McmcAlgo {
public:
	McmcMatMniw(
		const MatMniwParams& params, const MatMniwInits& inits,
		std::unique_ptr<MatShrinkageUpdater>& row_updater, std::unique_ptr<MatShrinkageUpdater>& col_updater,
		unsigned int seed,
		Optional<std::unique_ptr<MatShrinkageUpdater>> row_exogen = NULLOPT,
		Optional<std::unique_ptr<MatShrinkageUpdater>> col_exogen = NULLOPT
	)
	: bvhar::McmcAlgo(params, seed),
		x(params._x), y(params._y),
		row_updater(std::move(row_updater)), col_updater(std::move(col_updater)),
		num_row(params._row), num_col(params._col), num_design(params._design),
		nrow_row_coef(params._row_row_coef), nrow_col_coef(params._row_col_coef),
		nrow_row_exogen(params._row_exogen), nrow_col_exogen(params._col_exogen),
		row_record(num_iter + 1, std::vector<Eigen::MatrixXd>(2)), col_record(num_iter + 1, std::vector<Eigen::MatrixXd>(2)),
		row_coef(inits._init_row_coef), row_sig_lower(inits._init_row_lower),
		col_coef(inits._init_col_coef), col_sig_lower(inits._init_col_lower),
		mniw_record(std::make_unique<MatMniwRecords>(num_iter, num_row, num_col, row_coef.rows(), col_coef.rows())),
		row_prior_mean(params._row_mean), row_iw_scl(params._row_iw_scl),
		col_prior_mean(params._col_mean), col_iw_scl(params._col_iw_scl),
		row_prior_prec(params._row_prec), col_prior_prec(params._col_prec),
		row_iw_df(params._row_iw_df), col_iw_df(params._col_iw_df) {
		if (row_exogen) {
			exogen_row_updater = std::move(*row_exogen);
		}
		if (col_exogen) {
			exogen_col_updater = std::move(*col_exogen);
		}
		updateRecords();
	}
	virtual ~McmcMatMniw() = default;
	
	void doWarmUp() override {
		std::lock_guard<std::mutex> lock(mtx);
		updatePrec();
		updateCoefCov();
	}

	void doPosteriorDraws() override {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updatePrec();
		updateCoefCov();
		updateRecords();
	}

	LIST returnRecords(int num_burn, int thin) override {
		// LIST res = CREATE_LIST(
		// 	// NAMED("A_record") = row_coef_record,
		// 	// NAMED("Sigr_record") = row_sig_record,
		// 	// NAMED("B_record") = col_coef_record,
		// 	// NAMED("Sigc_record") = col_sig_record
		// 	NAMED("row_record") = WRAP(row_record),
		// 	NAMED("col_record") = WRAP(col_record)
		// );
		LIST res = mniw_record->returnListRecords(nrow_row_coef, num_row, nrow_row_exogen, nrow_col_coef, num_col, nrow_col_exogen);
		for (auto& record : res) {
			if (IS_MATRIX(ACCESS_LIST(record, res))) {
				ACCESS_LIST(record, res) = bvhar::thin_record(CAST<Eigen::MatrixXd>(ACCESS_LIST(record, res)), num_iter, num_burn, thin);
			} else {
				ACCESS_LIST(record, res) = bvhar::thin_record(CAST<Eigen::VectorXd>(ACCESS_LIST(record, res)), num_iter, num_burn, thin);
			}
		}
		return res;
	}

	MatMniwRecords returnStructRecords(int num_burn, int thin) const {
		return mniw_record->returnRecords(num_iter, num_burn, thin);
	}

protected:
	std::vector<Eigen::SparseMatrix<double>> x;
	std::vector<Eigen::MatrixXd> y;
	std::unique_ptr<MatShrinkageUpdater> row_updater;
	std::unique_ptr<MatShrinkageUpdater> col_updater;
	std::unique_ptr<MatShrinkageUpdater> exogen_row_updater;
	std::unique_ptr<MatShrinkageUpdater> exogen_col_updater;
	int num_row;
	int num_col;
	int num_design;
	int nrow_row_coef, nrow_col_coef, nrow_row_exogen, nrow_col_exogen;
	std::vector<std::vector<Eigen::MatrixXd>> row_record, col_record;
	Eigen::MatrixXd row_coef, row_sig_lower, col_coef, col_sig_lower;
	std::unique_ptr<MatMniwRecords> mniw_record;
	Eigen::MatrixXd row_prior_mean, row_iw_scl;
	Eigen::MatrixXd col_prior_mean, col_iw_scl;
	Eigen::VectorXd row_prior_prec, col_prior_prec;
	double row_iw_df, col_iw_df;

	void updatePrec() {
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
				row_prior_prec.tail(nrow_row_exogen),
				row_coef.bottomRows(nrow_row_exogen),
				row_sig_lower,
				row_prior_mean.bottomRows(nrow_row_exogen),
				rng
			);
		}
		if (exogen_col_updater) {
			exogen_col_updater->updatePrec(
				row_prior_prec.tail(nrow_col_exogen),
				row_coef.bottomRows(nrow_col_exogen),
				row_sig_lower,
				row_prior_mean.bottomRows(nrow_col_exogen),
				rng
			);
		}
	}

	void updateCoefCov() {
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
		// row_record[mcmc_step][0] = row_coef;
		// row_record[mcmc_step][1] = row_sig_lower * row_sig_lower.transpose();
		// col_record[mcmc_step][0] = col_coef;
		// col_record[mcmc_step][1] = col_sig_lower * col_sig_lower.transpose();
		mniw_record->assignRecords(
			mcmc_step, row_coef, row_sig_lower, col_coef, col_sig_lower,
			nrow_row_coef, num_row, nrow_row_exogen,
			nrow_col_coef, num_col, nrow_col_exogen
		);
	}
};

inline std::vector<std::unique_ptr<McmcMatMniw>> initialize_matmcmc(
	int num_chains, int num_iter, std::vector<Eigen::SparseMatrix<double>>& x, std::vector<Eigen::MatrixXd>& y,
	LIST& param_coef_sig, LIST_OF_LIST& coef_sig_init,
	LIST& row_prior, LIST_OF_LIST& row_init, const int row_prior_type,
	LIST& col_prior, LIST_OF_LIST& col_init, const int col_prior_type,
  Eigen::Ref<const Eigen::VectorXi> seed_chain,
	Optional<LIST> row_exogen_prior = NULLOPT, Optional<LIST_OF_LIST> row_exogen_init = NULLOPT, Optional<int> row_exogen_prior_type = NULLOPT, Optional<int> exogen_rows = NULLOPT,
	Optional<LIST> col_exogen_prior = NULLOPT, Optional<LIST_OF_LIST> col_exogen_init = NULLOPT, Optional<int> col_exogen_prior_type = NULLOPT, Optional<int> exogen_cols = NULLOPT
) {
	std::vector<std::unique_ptr<McmcMatMniw>> mcmc_ptr(num_chains);
	// MatMniwParams params(num_iter, x, y, param_coef_sig);
	MatMniwParams params = exogen_rows ? MatMniwParams(num_iter, x, y, param_coef_sig, *exogen_rows, *exogen_cols) : MatMniwParams(num_iter, x, y, param_coef_sig);
	for (int i = 0; i < num_chains; ++i) {
		LIST row_init_spec = row_init[i];
		LIST col_init_spec = col_init[i];
		auto row_updater = initialize_matshrinkageupdater(num_iter, row_prior, row_init_spec, row_prior_type);
		auto col_updater = initialize_matshrinkageupdater(num_iter, col_prior, col_init_spec, col_prior_type);
		row_updater->initPrec(params._row_prec.head(params._row_row_coef));
		col_updater->initPrec(params._col_prec.head(params._row_col_coef));
		LIST init_spec = coef_sig_init[i];
		MatMniwInits inits(init_spec);
		Optional<std::unique_ptr<MatShrinkageUpdater>> row_exogen_updater = NULLOPT;
		Optional<std::unique_ptr<MatShrinkageUpdater>> col_exogen_updater = NULLOPT;
		if (row_exogen_prior_type) {
			LIST row_exogen_init_spec = (*row_exogen_init)[i];
			// auto temp_row_exogen_updater = initialize_matshrinkageupdater(num_iter, *row_exogen_prior, row_exogen_init_spec, *row_exogen_prior_type);
			// row_exogen_updater = std::move(temp_row_exogen_updater);
			row_exogen_updater = initialize_matshrinkageupdater(num_iter, *row_exogen_prior, row_exogen_init_spec, *row_exogen_prior_type);
			(*row_exogen_updater)->initPrec(params._row_prec.tail(params._row_exogen));
		}
		if (col_exogen_prior_type) {
			LIST col_exogen_init_spec = (*col_exogen_init)[i];
			col_exogen_updater = initialize_matshrinkageupdater(num_iter, *col_exogen_prior, col_exogen_init_spec, *col_exogen_prior_type);
			(*col_exogen_updater)->initPrec(params._col_prec.tail(params._col_exogen));
		}
		// if (row_exogen_prior_type && col_exogen_prior_type) {
		// 	LIST row_exogen_init_spec = (*row_exogen_init)[i];
		// 	auto row_exogen_updater = initialize_matshrinkageupdater(num_iter, *row_exogen_prior, row_exogen_init_spec, *row_exogen_prior_type);
		// 	row_exogen_updater->initPrec(params._row_prec.head(params._row_exogen));
		// 	LIST col_exogen_init_spec = (*col_exogen_init)[i];
		// 	auto col_exogen_updater = initialize_matshrinkageupdater(num_iter, *col_exogen_prior, col_exogen_init_spec, *col_exogen_prior_type);
		// 	col_exogen_updater->initPrec(params._col_prec.head(params._col_exogen));
		// 	mcmc_ptr[i] = std::make_unique<McmcMatMniw>(
		// 		params, inits, row_updater, col_updater, static_cast<unsigned int>(seed_chain[i]),
		// 		std::move(row_exogen_updater), std::move(col_exogen_updater)
		// 	);
		// } else {
		// 	mcmc_ptr[i] = std::make_unique<McmcMatMniw>(
		// 		params, inits, row_updater, col_updater, static_cast<unsigned int>(seed_chain[i])
		// 	);
		// }
		mcmc_ptr[i] = std::make_unique<McmcMatMniw>(
			params, inits, row_updater, col_updater, static_cast<unsigned int>(seed_chain[i]),
			std::move(row_exogen_updater), std::move(col_exogen_updater)
		);
	}
	return mcmc_ptr;
}

class MatMcmcRun : public bvhar::McmcRun {
public:
	MatMcmcRun(
		int num_chains, int num_iter, int num_burn, int thin,
		std::vector<Eigen::SparseMatrix<double>>& x, std::vector<Eigen::MatrixXd>& y,
		LIST& param_coef_sig, LIST_OF_LIST& coef_sig_init,
		LIST& row_prior, LIST_OF_LIST& row_init, const int row_prior_type,
		LIST& col_prior, LIST_OF_LIST& col_init, const int col_prior_type,
		const Eigen::VectorXi& seed_chain, bool display_progress, int nthreads,
		Optional<LIST> row_exogen_prior = NULLOPT, Optional<LIST_OF_LIST> row_exogen_init = NULLOPT, Optional<int> row_exogen_prior_type = NULLOPT, Optional<int> exogen_rows = NULLOPT,
		Optional<LIST> col_exogen_prior = NULLOPT, Optional<LIST_OF_LIST> col_exogen_init = NULLOPT, Optional<int> col_exogen_prior_type = NULLOPT, Optional<int> exogen_cols = NULLOPT
	)
	: bvhar::McmcRun(num_chains, num_iter, num_burn, thin, display_progress, nthreads) {
		auto temp_mcmc = initialize_matmcmc(
			num_chains, num_iter - num_burn, x, y,
			param_coef_sig, coef_sig_init,
			row_prior, row_init, row_prior_type,
			col_prior, col_init, col_prior_type,
			seed_chain,
			row_exogen_prior, row_exogen_init, row_exogen_prior_type, exogen_rows,
			col_exogen_prior, col_exogen_init, col_exogen_prior_type, exogen_cols
		);
		for (int i = 0; i < num_chains; ++i) {
			mcmc_ptr[i] = std::move(temp_mcmc[i]);
		}
	}
	virtual ~MatMcmcRun() = default;
};

} // namespace baymar

#endif // BAYMAR_BAYES_MNIW_MNIW_H