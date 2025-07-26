#include <baymar/mdfm>

//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_bmdfm(int num_chains, int num_iter, int num_burn, int thin,
													std::vector<Eigen::MatrixXd>& y,
													int factor_lag, Rcpp::List param_dfm, Rcpp::List dfm_init,
													Rcpp::List row_prior, Rcpp::List row_init, int row_prior_type,
													Rcpp::List col_prior, Rcpp::List col_init, int col_prior_type,
													Eigen::VectorXi seed_chain, bool display_progress, int nthreads) {
	// auto mcmc_run = [&]() -> std::unique_ptr<bvhar::McmcRun> {
	// 	return std::make_unique<baymar::MatDfmRun<baymar::McmcMatDfmVar>>(
	// 		num_chains, num_iter, num_burn, thin,
	// 		y, factor_lag, param_dfm, dfm_init,
	// 		row_prior, row_init, row_prior_type,
	// 		col_prior, col_init, col_prior_type,
	// 		seed_chain, display_progress, nthreads
	// 	);
	// }();
	auto mcmc_run = std::make_unique<baymar::MatDfmRun<baymar::McmcMatDfmVar>>(
		num_chains, num_iter, num_burn, thin,
		y, factor_lag, param_dfm, dfm_init,
		row_prior, row_init, row_prior_type,
		col_prior, col_init, col_prior_type,
		seed_chain, display_progress, nthreads
	);
	return mcmc_run->returnRecords();
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_bdfm_mniw(int num_chains, int step,
															int nrow_factor, int ncol_factor, int factor_lag,
													 	 	Rcpp::List fit_record, Eigen::VectorXi seed_chain, int nthreads) {
	auto forecaster = std::make_unique<baymar::MatDfmForecastRun>(
		num_chains, step, nrow_factor, ncol_factor, factor_lag,
		fit_record, seed_chain, nthreads
	);
	return Rcpp::wrap(forecaster->returnForecast());
}
