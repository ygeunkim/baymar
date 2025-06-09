#include <baymar/mniw>

//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_bmar_mniw(int num_chains, int num_iter, int num_burn, int thin,
															std::vector<Eigen::SparseMatrix<double>>& x, std::vector<Eigen::MatrixXd>& y,
															Rcpp::List param_coef_sig, Rcpp::List coef_sig_init,
															Rcpp::List row_prior, Rcpp::List row_init, int row_prior_type,
															Rcpp::List col_prior, Rcpp::List col_init, int col_prior_type,
															Rcpp::List exogen_row_prior, Rcpp::List exogen_row_init, int exogen_row_prior_type, int exogen_rows,
															Rcpp::List exogen_col_prior, Rcpp::List exogen_col_init, int exogen_col_prior_type, int exogen_cols,
															Eigen::VectorXi seed_chain, bool display_progress, int nthreads) {
	auto mcmc_run = [&]() -> std::unique_ptr<baymar::MatMcmcRun> {
		if (exogen_row_prior_type != 0 && exogen_col_prior_type != 0) {
			return std::make_unique<baymar::MatMcmcRun>(
				num_chains, num_iter, num_burn, thin,
				x, y,
				param_coef_sig, coef_sig_init,
				row_prior, row_init, row_prior_type,
				col_prior, col_init, col_prior_type,
				seed_chain, display_progress, nthreads,
				exogen_row_prior, exogen_row_init, exogen_row_prior_type, exogen_rows,
				exogen_col_prior, exogen_col_init, exogen_col_prior_type, exogen_cols
			);
		} if (exogen_row_prior_type != 0 && exogen_col_prior_type == 0) {
			return std::make_unique<baymar::MatMcmcRun>(
				num_chains, num_iter, num_burn, thin,
				x, y,
				param_coef_sig, coef_sig_init,
				row_prior, row_init, row_prior_type,
				col_prior, col_init, col_prior_type,
				seed_chain, display_progress, nthreads,
				exogen_row_prior, exogen_row_init, exogen_row_prior_type, exogen_rows
			); 
		} else if (exogen_row_prior_type == 0 && exogen_col_prior_type != 0) {
			return std::make_unique<baymar::MatMcmcRun>(
				num_chains, num_iter, num_burn, thin,
				x, y,
				param_coef_sig, coef_sig_init,
				row_prior, row_init, row_prior_type,
				col_prior, col_init, col_prior_type,
				seed_chain, display_progress, nthreads,
				NULLOPT, NULLOPT, NULLOPT, NULLOPT,
				exogen_col_prior, exogen_col_init, exogen_col_prior_type, exogen_cols
			);
		}
		return std::make_unique<baymar::MatMcmcRun>(
			num_chains, num_iter, num_burn, thin,
			x, y,
			param_coef_sig, coef_sig_init,
			row_prior, row_init, row_prior_type,
			col_prior, col_init, col_prior_type,
			seed_chain, display_progress, nthreads
		);
	}();
	return mcmc_run->returnRecords();
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_bmar_mniw(int num_chains, int lag, int step, Eigen::MatrixXd response_mat, int num_data,
													 	 	Rcpp::List fit_record, Eigen::VectorXi seed_chain, int nthreads) {
	auto forecaster = std::make_unique<baymar::MatMniwForecastRun>(num_chains, lag, step, response_mat, num_data, fit_record, seed_chain, nthreads);
	return Rcpp::wrap(forecaster->returnForecast());
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List roll_bmar_mniw(Eigen::MatrixXd y, int lag, int num_data, int num_chains, int num_iter, int num_burn, int thin,
													Rcpp::List fit_record, bool run_mcmc,
													Rcpp::List param_coef_sig, Rcpp::List coef_sig_init,
													Rcpp::List row_prior, Rcpp::List row_init, int row_prior_type,
													Rcpp::List col_prior, Rcpp::List col_init, int col_prior_type,
													int step, Eigen::MatrixXd y_test,
													Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast,
													bool display_progress, int nthreads) {
	auto forecaster = baymar::initialize_matmniwoutforecaster<baymar::MatMniwRollForecastRun>(
		y, num_data, lag, num_chains, num_iter, num_burn, thin, fit_record, run_mcmc,
		param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
		step, y_test, seed_chain, seed_forecast, display_progress, nthreads
	);
	return forecaster->returnForecast();
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List expand_bmar_mniw(Eigen::MatrixXd y, int lag, int num_data, int num_chains, int num_iter, int num_burn, int thin,
													  Rcpp::List fit_record, bool run_mcmc,
													  Rcpp::List param_coef_sig, Rcpp::List coef_sig_init,
													  Rcpp::List row_prior, Rcpp::List row_init, int row_prior_type,
													  Rcpp::List col_prior, Rcpp::List col_init, int col_prior_type,
													  int step, Eigen::MatrixXd y_test,
													  Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast,
													  bool display_progress, int nthreads) {
	auto forecaster = baymar::initialize_matmniwoutforecaster<baymar::MatMniwExpandForecastRun>(
		y, num_data, lag, num_chains, num_iter, num_burn, thin, fit_record, run_mcmc,
		param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
		step, y_test, seed_chain, seed_forecast, display_progress, nthreads
	);
	return forecaster->returnForecast();
}
