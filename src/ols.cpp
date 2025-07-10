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
