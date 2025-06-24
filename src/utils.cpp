#include <baymar/utils>

//' @noRd
// [[Rcpp::export]]
std::vector<Eigen::MatrixXd> sim_mar_export(int num_sim,
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
