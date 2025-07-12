#include <baymar/ols>

//' @noRd
// [[Rcpp::export]]
std::vector<Eigen::MatrixXd> sim_mar_process(int num_sim, int num_burn, int lag,
																		 				 Eigen::MatrixXd init,
																		 				 Eigen::MatrixXd row_coef, Eigen::MatrixXd col_coef,
																		 				 Eigen::MatrixXd row_sig, Eigen::MatrixXd col_sig,
																						 unsigned int seed) {
	auto dgp_run = std::make_unique<baymar::MarSimulator>(num_sim, num_burn, lag, init, row_coef, col_coef, row_sig, col_sig, seed);
	return dgp_run->returnDgp();
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List sim_mdfm_process(int num_sim, int num_burn, int lag,
													  Eigen::MatrixXd row_coef, Eigen::MatrixXd col_coef,
													  Eigen::MatrixXd row_sig, Eigen::MatrixXd col_sig,
														Eigen::MatrixXd factor_init,
													  Eigen::MatrixXd factor_row_coef, Eigen::MatrixXd factor_col_coef,
													  Eigen::MatrixXd factor_row_sig, Eigen::MatrixXd factor_col_sig,
													  unsigned int seed) {
	auto dgp_run = std::make_unique<baymar::MdfmMarSimulator>(
		num_sim, num_burn, lag,
		row_coef, col_coef, row_sig, col_sig,
		factor_init, factor_row_coef, factor_col_coef, factor_row_sig, factor_col_sig,
		seed
	);
	return dgp_run->returnDgp();
}
