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
}
