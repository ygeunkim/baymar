#ifndef BAYMAR_BAYES_MNIW_MNIW_H
#define BAYMAR_BAYES_MNIW_MNIW_H

#include "./config.h"

namespace baymar {

class McmcMatMniw;

class McmcMatMniw {
public:
	McmcMatMniw(
		const MatMniwParams& params, const MatMniwInits& inits,
		std::unique_ptr<MatShrinkageUpdater>& row_updater, std::unique_ptr<MatShrinkageUpdater>& col_updater,
		unsigned int seed
	)
	: x(params._x), y(params._y),
		row_updater(std::move(row_updater)), col_updater(std::move(col_updater)),
		num_iter(params._iter), num_row(params._row), num_col(params._col), num_design(params._design),
		row_record(num_iter + 1, std::vector<Eigen::MatrixXd>(2)), col_record(num_iter + 1, std::vector<Eigen::MatrixXd>(2)),
		mcmc_step(0), rng(seed),
		row_coef(inits._init_row_coef), row_sig_lower(inits._init_row_lower),
		col_coef(inits._init_col_coef), col_sig_lower(inits._init_col_lower),
		// row_kappa(.1), col_kappa(.1),
		row_prior_mean(params._row_mean), row_iw_scl(params._row_iw_scl),
		col_prior_mean(params._col_mean), col_iw_scl(params._col_iw_scl),
		row_prior_prec(params._row_prec.diagonal()), col_prior_prec(params._col_prec.diagonal()),
		row_iw_df(params._row_iw_df), col_iw_df(params._col_iw_df) {
		updateRecords();
	}
	virtual ~McmcMatMniw() = default;
	
	void doWarmUp() {
		std::lock_guard<std::mutex> lock(mtx);
		updatePrec();
		updateCoefCov();
	}

	void doPosteriorDraws() {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updatePrec();
		updateCoefCov();
		updateRecords();
	}

	LIST returnRecords() {
		LIST res = CREATE_LIST(
			// NAMED("A_record") = row_coef_record,
			// NAMED("Sigr_record") = row_sig_record,
			// NAMED("B_record") = col_coef_record,
			// NAMED("Sigc_record") = col_sig_record
			NAMED("row_record") = WRAP(row_record),
			NAMED("col_record") = WRAP(col_record)
		);
		return res;
	}

protected:
	std::mutex mtx;
	std::vector<Eigen::SparseMatrix<double>> x;
	std::vector<Eigen::MatrixXd> y;
	std::unique_ptr<MatShrinkageUpdater> row_updater;
	std::unique_ptr<MatShrinkageUpdater> col_updater;
	int num_iter;
	int num_row;
	int num_col;
	int num_design;
	std::vector<std::vector<Eigen::MatrixXd>> row_record, col_record;
	std::atomic<int> mcmc_step; // MCMC step
	BHRNG rng; // RNG instance for multi-chain
	// std::vector<Eigen::MatrixXd> row_params, col_params;
	Eigen::MatrixXd row_coef, row_sig_lower, col_coef, col_sig_lower;
	// double row_kappa, col_kappa;
	Eigen::MatrixXd row_prior_mean, row_iw_scl;
	Eigen::MatrixXd col_prior_mean, col_iw_scl;
	Eigen::VectorXd row_prior_prec, col_prior_prec;
	double row_iw_df, col_iw_df;

	/**
	 * @brief Increment the MCMC step
	 * 
	 */
	void addStep() { ++mcmc_step; }

	void updatePrec() {
		// minnesota_kappa(row_kappa, row_prior_mean, row_prior_prec, row_coef, row_sig_lower, 3.0, 2.0, rng);
		// minnesota_kappa(col_kappa, col_prior_mean, col_prior_prec, col_coef, col_sig_lower, 3.0, 2.0, rng);
		row_updater->updatePrec(row_prior_prec, row_coef, row_sig_lower, row_prior_mean, rng);
		col_updater->updatePrec(col_prior_prec, col_coef, col_sig_lower, col_prior_mean, rng);
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
		row_record[mcmc_step][0] = row_coef;
		row_record[mcmc_step][1] = row_sig_lower * row_sig_lower.transpose();
		col_record[mcmc_step][0] = col_coef;
		col_record[mcmc_step][1] = col_sig_lower * col_sig_lower.transpose();
	}
};

inline std::vector<std::unique_ptr<McmcMatMniw>> initialize_matmcmc(
	int num_chains, int num_iter, std::vector<Eigen::SparseMatrix<double>>& x, std::vector<Eigen::MatrixXd>& y,
	LIST& param_coef_sig, LIST_OF_LIST& coef_sig_init,
	LIST& row_prior, LIST_OF_LIST& row_init, const int row_prior_type,
	LIST& col_prior, LIST_OF_LIST& col_init, const int col_prior_type,
  Eigen::Ref<const Eigen::VectorXi> seed_chain
) {
	std::vector<std::unique_ptr<McmcMatMniw>> mcmc_ptr(num_chains);
	MatMniwParams params(num_iter, x, y, param_coef_sig);
	for (int i = 0; i < num_chains; ++i) {
		LIST row_init_spec = row_init[i];
		LIST col_init_spec = col_init[i];
		auto row_updater = initialize_matshrinkageupdater(num_iter, row_prior, row_init_spec, row_prior_type);
		auto col_updater = initialize_matshrinkageupdater(num_iter, col_prior, col_init_spec, col_prior_type);
		LIST init_spec = coef_sig_init[i];
		MatMniwInits inits(init_spec);
		mcmc_ptr[i] = std::make_unique<McmcMatMniw>(
			params, inits, row_updater, col_updater, static_cast<unsigned int>(seed_chain[i])
		);
	}
	return mcmc_ptr;
}

class MatMcmcRun {
public:
	MatMcmcRun(
		int num_chains, int num_iter, int num_burn, int thin,
		std::vector<Eigen::SparseMatrix<double>>& x, std::vector<Eigen::MatrixXd>& y,
		LIST& param_coef_sig, LIST_OF_LIST& coef_sig_init,
		LIST& row_prior, LIST_OF_LIST& row_init, const int row_prior_type,
		LIST& col_prior, LIST_OF_LIST& col_init, const int col_prior_type,
		const Eigen::VectorXi& seed_chain, bool display_progress, int nthreads
	)
	: num_chains(num_chains), num_iter(num_iter), num_burn(num_burn), thin(thin), nthreads(nthreads),
		display_progress(display_progress), mcmc_ptr(num_chains), res(num_chains) {
		auto temp_mcmc = initialize_matmcmc(
			num_chains, num_iter - num_burn, x, y,
			param_coef_sig, coef_sig_init,
			row_prior, row_init, row_prior_type,
			col_prior, col_init, col_prior_type,
			seed_chain
		);
		for (int i = 0; i < num_chains; ++i) {
			mcmc_ptr[i] = std::move(temp_mcmc[i]);
		}
	}
	virtual ~MatMcmcRun() = default;

	/**
	 * @brief Conduct multi-chain MCMC
	 *
	 */
	void fit() {
		if (num_chains == 1) {
			runGibbs(0);
		} else {
		#ifdef _OPENMP
			#pragma omp parallel for num_threads(nthreads)
		#endif
			for (int chain = 0; chain < num_chains; ++chain) {
				runGibbs(chain);
			}
		}
	}

	/**
	 * @brief Conduct multi-chain MCMC and return MCMC records of every chain
	 *
	 * @return LIST_OF_LIST `LIST_OF_LIST`
	 */
	LIST_OF_LIST returnRecords() {
		fit();
		return WRAP(res);
	}

private:
	int num_chains;
	int num_iter;
	int num_burn;
	int thin;
	int nthreads;
	bool display_progress;
	std::vector<std::unique_ptr<McmcMatMniw>> mcmc_ptr;
	std::vector<LIST> res;

	/**
	 * @brief Single chain MCMC
	 *
	 * @param chain Chain id
	 */
	void runGibbs(int chain) {
		std::string log_name = fmt::format("Chain {}", chain + 1);
		auto logger = spdlog::get(log_name);
		if (logger == nullptr) {
			logger = SPDLOG_SINK_MT(log_name);
		}
		logger->set_pattern("[%n] [Thread " + std::to_string(omp_get_thread_num()) + "] %v");
		int logging_freq = num_iter / 20; // 5 percent
		if (logging_freq == 0) {
			logging_freq = 1;
		}
		for (int i = 0; i < num_burn; ++i) {
			mcmc_ptr[chain]->doWarmUp();
			if (display_progress && (i + 1) % logging_freq == 0) {
				logger->info("{} / {} (Warmup)", i + 1, num_iter);
			}
		}
		logger->flush();
		for (int i = num_burn; i < num_iter; ++i) {
			mcmc_ptr[chain]->doPosteriorDraws();
			if (display_progress && (i + 1) % logging_freq == 0) {
				logger->info("{} / {} (Sampling)", i + 1, num_iter);
			}
		}
	#ifdef _OPENMP
		#pragma omp critical
	#endif
		{
			res[chain] = mcmc_ptr[chain]->returnRecords();
		}
		logger->flush();
		spdlog::drop(log_name);
	}
};

} // namespace baymar

#endif // BAYMAR_BAYES_MNIW_MNIW_H