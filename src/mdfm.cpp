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
	// 	return std::make_unique<baecon::baymar::MatDfmRun<baecon::baymar::McmcMatDfmVar>>(
	// 		num_chains, num_iter, num_burn, thin,
	// 		y, factor_lag, param_dfm, dfm_init,
	// 		row_prior, row_init, row_prior_type,
	// 		col_prior, col_init, col_prior_type,
	// 		seed_chain, display_progress, nthreads
	// 	);
	// }();
	auto mcmc_run = std::make_unique<baecon::baymar::MatDfmRun>(
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
													 	 	Rcpp::List fit_record, Eigen::VectorXi seed_chain, int nthreads,
															bool insample) {
	auto forecaster = std::make_unique<baecon::baymar::MatDfmForecastRun>(
		num_chains, step, nrow_factor, ncol_factor, factor_lag,
		fit_record, seed_chain, nthreads, insample
	);
	if (insample) {
		return Rcpp::wrap(forecaster->returnPredict());
	}
	return Rcpp::wrap(forecaster->returnForecast());
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List roll_bdfm_mniw(Eigen::MatrixXd y, int num_data, int num_chains, int num_iter, int num_burn, int thin,
													Rcpp::List fit_record, bool run_mcmc,
													Rcpp::List param_coef_sig, Rcpp::List coef_sig_init,
													Rcpp::List row_prior, Rcpp::List row_init, int row_prior_type,
													Rcpp::List col_prior, Rcpp::List col_init, int col_prior_type,
													int factor_rows, int factor_cols, int factor_lag,
													int step, Eigen::MatrixXd y_test, bool get_lpl, bool use_fit,
													Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast,
													bool display_progress, int nthreads) {
	auto forecaster = baecon::baymar::initialize_matdfmoutforecaster<baecon::baymar::MatDfmRollForecastRun>(
		y, num_data, num_chains, num_iter, num_burn, thin, fit_record, run_mcmc,
		param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
		factor_rows, factor_cols, factor_lag,
		step, y_test, get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads
	);
	return forecaster->returnForecast();
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List expand_bdfm_mniw(Eigen::MatrixXd y, int num_data, int num_chains, int num_iter, int num_burn, int thin,
													  Rcpp::List fit_record, bool run_mcmc,
													  Rcpp::List param_coef_sig, Rcpp::List coef_sig_init,
													  Rcpp::List row_prior, Rcpp::List row_init, int row_prior_type,
													  Rcpp::List col_prior, Rcpp::List col_init, int col_prior_type,
													  int factor_rows, int factor_cols, int factor_lag,
													  int step, Eigen::MatrixXd y_test, bool get_lpl, bool use_fit,
													  Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast,
													  bool display_progress, int nthreads) {
	auto forecaster = baecon::baymar::initialize_matdfmoutforecaster<baecon::baymar::MatDfmExpandForecastRun>(
		y, num_data, num_chains, num_iter, num_burn, thin, fit_record, run_mcmc,
		param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
		factor_rows, factor_cols, factor_lag,
		step, y_test, get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads
	);
	return forecaster->returnForecast();
}
