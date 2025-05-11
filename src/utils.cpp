#include <baymar/utils>

//' @noRd
// [[Rcpp::export]]
std::vector<Eigen::MatrixXd> sim_mar(int num_sim,
																		 int num_burn,
																		 Eigen::SparseMatrix<double> init,
																		 Eigen::MatrixXd row_coef, Eigen::MatrixXd col_coef,
																		 Eigen::MatrixXd row_sig, Eigen::MatrixXd col_sig) {
	std::vector<Eigen::MatrixXd> y(num_sim);
	int nrow = row_coef.cols();
	int ncol = col_coef.cols();
	Eigen::MatrixXd error(nrow, ncol);
	for (int i = 0; i < num_burn; ++i) {
		Eigen::MatrixXd temp_y(nrow, ncol);
		baymar::update_mar(temp_y, init, row_coef, col_coef);
		error = bvhar::sim_mn(
			Eigen::MatrixXd::Zero(nrow, ncol),
			row_sig, col_sig, false
		);
		temp_y += error;
		baymar::update_x(init, temp_y, nrow, ncol);
	}
	for (int i = 0; i < num_sim; ++i) {
		y[i].setZero(nrow, ncol);
		baymar::update_mar(y[i], init, row_coef, col_coef);
		error = bvhar::sim_mn(
			Eigen::MatrixXd::Zero(nrow, ncol),
			row_sig, col_sig, false
		);
		y[i] += error;
		baymar::update_x(init, y[i], nrow, ncol);
	}
	return y;
}

// Eigen::MatrixXd sim_var_eigen(int num_sim, 
//                               int num_burn, 
//                               Eigen::MatrixXd var_coef, 
//                               int var_lag, 
//                               Eigen::MatrixXd sig_error, 
//                               Eigen::MatrixXd init,
//                               int process,
//                               double mvt_df) {
//   int dim = sig_error.cols(); // m: dimension of time series
//   int dim_design = var_coef.rows(); // k = mp + 1 (const) or mp (none)
//   int num_rand = num_sim + num_burn; // sim + burnin
//   Eigen::MatrixXd obs_p(1, dim_design); // row vector of X0: yp^T, ..., y1^T, (1)
//   obs_p(0, dim_design - 1) = 1.0; // for constant term if exists
//   for (int i = 0; i < var_lag; i++) {
//     obs_p.block(0, i * dim, 1, dim) = init.row(var_lag - i - 1);
//   }
//   Eigen::MatrixXd res(num_rand, dim); // Output: from y(p + 1)^T to y(n + p)^T
//   // epsilon ~ N(0, sig_error)
//   Eigen::VectorXd sig_mean = Eigen::VectorXd::Zero(dim); // zero mean
//   // Eigen::MatrixXd error_term = sim_mgaussian(num_rand, sig_mean, sig_error); // simulated error term: num_rand x m
//   Eigen::MatrixXd error_term(num_rand, dim);
//   switch (process) {
//   case 1:
//     error_term = sim_mgaussian(num_rand, sig_mean, sig_error);
//     break;
//   case 2:
//     error_term = sim_mstudent(num_rand, mvt_df, sig_mean, sig_error * (mvt_df - 2) / mvt_df, 1);
//     break;
//   default:
//     Rcpp::stop("Invalid 'process' option.");
//   }
//   res.row(0) = obs_p * var_coef + error_term.row(0); // y(p + 1) = [yp^T, ..., y1^T, 1] A + eps(T)
//   for (int i = 1; i < num_rand; i++) {
//     for (int t = 1; t < var_lag; t++) {
//       obs_p.block(0, t * dim, 1, dim) = obs_p.block(0, (t - 1) * dim, 1, dim);
//     }
//     obs_p.block(0, 0, 1, dim) = res.row(i - 1);
//     res.row(i) = obs_p * var_coef + error_term.row(i); // yi = [y(i-1), ..., y(i-p), 1] A + eps(i)
//   }
//   return res.bottomRows(num_rand - num_burn);
// }

