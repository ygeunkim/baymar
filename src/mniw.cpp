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
															Rcpp::List factor_row_prior, Rcpp::List factor_row_init, int factor_row_prior_type, int factor_rows,
															Rcpp::List factor_col_prior, Rcpp::List factor_col_init, int factor_col_prior_type, int factor_cols,
															int factor_lag,
															Eigen::VectorXi seed_chain, bool display_progress, int nthreads) {
	auto mcmc_run = [&]() -> std::unique_ptr<baymar::MatMcmcRun> {
		if (factor_row_prior_type != 0 && factor_col_prior_type != 0) {
			if (exogen_row_prior_type != 0 && exogen_col_prior_type != 0) {
				return std::make_unique<baymar::MatMcmcRun>(
					num_chains, num_iter, num_burn, thin,
					x, y,
					param_coef_sig, coef_sig_init,
					row_prior, row_init, row_prior_type,
					col_prior, col_init, col_prior_type,
					seed_chain, display_progress, nthreads,
					exogen_row_prior, exogen_row_init, exogen_row_prior_type, exogen_rows,
					exogen_col_prior, exogen_col_init, exogen_col_prior_type, exogen_cols,
					factor_row_prior, factor_row_init, factor_row_prior_type, factor_rows,
					factor_col_prior, factor_col_init, factor_col_prior_type, factor_cols,
					factor_lag
				);
			} if (exogen_row_prior_type != 0 && exogen_col_prior_type == 0) {
				return std::make_unique<baymar::MatMcmcRun>(
					num_chains, num_iter, num_burn, thin,
					x, y,
					param_coef_sig, coef_sig_init,
					row_prior, row_init, row_prior_type,
					col_prior, col_init, col_prior_type,
					seed_chain, display_progress, nthreads,
					exogen_row_prior, exogen_row_init, exogen_row_prior_type, exogen_rows,
					BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT,
					factor_row_prior, factor_row_init, factor_row_prior_type, factor_rows,
					factor_col_prior, factor_col_init, factor_col_prior_type, factor_cols,
					factor_lag
				); 
			} else if (exogen_row_prior_type == 0 && exogen_col_prior_type != 0) {
				return std::make_unique<baymar::MatMcmcRun>(
					num_chains, num_iter, num_burn, thin,
					x, y,
					param_coef_sig, coef_sig_init,
					row_prior, row_init, row_prior_type,
					col_prior, col_init, col_prior_type,
					seed_chain, display_progress, nthreads,
					BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT,
					exogen_col_prior, exogen_col_init, exogen_col_prior_type, exogen_cols,
					factor_row_prior, factor_row_init, factor_row_prior_type, factor_rows,
					factor_col_prior, factor_col_init, factor_col_prior_type, factor_cols,
					factor_lag
				);
			}
			return std::make_unique<baymar::MatMcmcRun>(
				num_chains, num_iter, num_burn, thin,
				x, y,
				param_coef_sig, coef_sig_init,
				row_prior, row_init, row_prior_type,
				col_prior, col_init, col_prior_type,
				seed_chain, display_progress, nthreads,
				BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT,
				BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT,
				factor_row_prior, factor_row_init, factor_row_prior_type, factor_rows,
				factor_col_prior, factor_col_init, factor_col_prior_type, factor_cols,
				factor_lag
			);
		}
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
				BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT,
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
Rcpp::List forecast_bmarx_mniw(int num_chains, int lag, int step, Eigen::MatrixXd response_mat, int num_data,
													 	 	 Rcpp::List fit_record, Eigen::VectorXi seed_chain,
															 Eigen::MatrixXd exogen, int exogen_lag, int nthreads) {
	auto forecaster = std::make_unique<baymar::MatMniwForecastRun>(
		num_chains, lag, step, response_mat, num_data, fit_record, seed_chain, nthreads,
		exogen, exogen_lag
	);
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
Rcpp::List roll_bmarx_mniw(Eigen::MatrixXd y, int lag, int num_data, int num_chains, int num_iter, int num_burn, int thin,
													 Rcpp::List fit_record, bool run_mcmc,
													 Rcpp::List param_coef_sig, Rcpp::List coef_sig_init,
													 Rcpp::List row_prior, Rcpp::List row_init, int row_prior_type,
													 Rcpp::List col_prior, Rcpp::List col_init, int col_prior_type,
													 int step, Eigen::MatrixXd y_test,
													 Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast,
													 bool display_progress, int nthreads,
													 Eigen::MatrixXd exogen, int exogen_lag,
													 Rcpp::List exogen_row_prior, Rcpp::List exogen_row_init, int exogen_row_prior_type,
													 Rcpp::List exogen_col_prior, Rcpp::List exogen_col_init, int exogen_col_prior_type) {
	auto forecaster = baymar::initialize_matmniwoutforecaster<baymar::MatMniwRollForecastRun>(
		y, num_data, lag, num_chains, num_iter, num_burn, thin, fit_record, run_mcmc,
		param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
		step, y_test, seed_chain, seed_forecast, display_progress, nthreads,
		exogen_row_prior, exogen_row_init, exogen_row_prior_type,
		exogen_col_prior, exogen_col_init, exogen_col_prior_type,
		exogen, exogen_lag
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

//' @noRd
// [[Rcpp::export]]
Rcpp::List expand_bmarx_mniw(Eigen::MatrixXd y, int lag, int num_data, int num_chains, int num_iter, int num_burn, int thin,
													   Rcpp::List fit_record, bool run_mcmc,
													   Rcpp::List param_coef_sig, Rcpp::List coef_sig_init,
													   Rcpp::List row_prior, Rcpp::List row_init, int row_prior_type,
													   Rcpp::List col_prior, Rcpp::List col_init, int col_prior_type,
													   int step, Eigen::MatrixXd y_test,
													   Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast,
													   bool display_progress, int nthreads,
														 Eigen::MatrixXd exogen, int exogen_lag,
													   Rcpp::List exogen_row_prior, Rcpp::List exogen_row_init, int exogen_row_prior_type,
													   Rcpp::List exogen_col_prior, Rcpp::List exogen_col_init, int exogen_col_prior_type) {
	auto forecaster = baymar::initialize_matmniwoutforecaster<baymar::MatMniwExpandForecastRun>(
		y, num_data, lag, num_chains, num_iter, num_burn, thin, fit_record, run_mcmc,
		param_coef_sig, coef_sig_init, row_prior, row_init, row_prior_type, col_prior, col_init, col_prior_type,
		step, y_test, seed_chain, seed_forecast, display_progress, nthreads,
		exogen_row_prior, exogen_row_init, exogen_row_prior_type,
		exogen_col_prior, exogen_col_init, exogen_col_prior_type,
		exogen, exogen_lag
	);
	return forecaster->returnForecast();
}
