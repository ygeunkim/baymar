#include <baymar/mniw>

//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_bmar_mniw(int num_chains, int num_iter, int num_burn, int thin,
															std::vector<Eigen::SparseMatrix<double>>& x, std::vector<Eigen::MatrixXd>& y,
															Rcpp::List param_coef_sig, Rcpp::List coef_sig_init,
															Rcpp::List row_prior, Rcpp::List row_init, int row_prior_type,
															Rcpp::List col_prior, Rcpp::List col_init, int col_prior_type,
															Eigen::VectorXi seed_chain, bool display_progress, int nthreads) {
	auto mcmc_run = std::make_unique<baymar::MatMcmcRun>(
		num_chains, num_iter, num_burn, thin,
		x, y,
		param_coef_sig, coef_sig_init,
		row_prior, row_init, row_prior_type,
		col_prior, col_init, col_prior_type,
		seed_chain, display_progress, nthreads
	);
	return mcmc_run->returnRecords();
// 	std::vector<std::unique_ptr<baymar::McmcMatMniw>> mcmc_ptr(num_chains);
// 	for (int i = 0; i < num_chains; ++i) {
// 		mcmc_ptr[i] = std::make_unique<baymar::McmcMatMniw>(
// 			num_iter - num_burn,
// 			x, y,
// 			row_prior_mean, row_prior_prec, row_iw_scl, row_iw_df,
// 			col_prior_mean, col_prior_prec, col_iw_scl, col_iw_df,
// 			init_row[i], init_col[i], static_cast<unsigned int>(seed_chain[i])
// 		);
// 	}
// 	std::vector<Rcpp::List> res(num_chains);
// #ifdef _OPENMP
// 	#pragma omp parallel for num_threads(nthreads)
// #endif
// 	for (int chain = 0; chain < num_chains; ++chain) {
// 		std::string log_name = fmt::format("Chain {}", chain + 1);
// 		auto logger = spdlog::get(log_name);
// 		if (logger == nullptr) {
// 			logger = SPDLOG_SINK_MT(log_name);
// 		}
// 		logger->set_pattern("[%n] [Thread " + std::to_string(omp_get_thread_num()) + "] %v");
// 		int logging_freq = num_iter / 20; // 5 percent
// 		if (logging_freq == 0) {
// 			logging_freq = 1;
// 		}
// 		for (int i = 0; i < num_burn; ++i) {
// 			mcmc_ptr[chain]->doWarmUp();
// 			if ((i + 1) % logging_freq == 0) {
// 				logger->info("{} / {} (Warmup)", i + 1, num_iter);
// 			}
// 		}
// 		for (int i = num_burn; i < num_iter; ++i) {
// 			mcmc_ptr[chain]->doPosteriorDraws();
// 			if ((i + 1) % logging_freq == 0) {
// 				logger->info("{} / {} (Sampling)", i + 1, num_iter);
// 			}
// 		}
// 	#ifdef _OPENMP
// 		#pragma omp critical
// 	#endif
// 		{
// 			res[chain] = mcmc_ptr[chain]->returnRecords();
// 		}
// 		logger->flush();
// 		spdlog::drop(log_name);
// 	}
// 	// return Rcpp::List::create(
// 	// 	Rcpp::Named("_record") = 1
// 	// );
// 	return Rcpp::wrap(res);
}
