#include <baymar/utils>

// Get standard error of AR(p) Ridge regression
//' @noRd
// [[Rcpp::export]]
double ar_ols_sd(Eigen::MatrixXd y, int p, bool include_mean, double penalty) {
	Eigen::VectorXd response = baecon::bvhar::build_y0(y, p, p + 1).col(0);
	Eigen::MatrixXd design = baecon::bvhar::build_x0(y, p, include_mean);
	Eigen::LLT<Eigen::MatrixXd> llt_of_xtx(
		(design.transpose() * design + penalty * Eigen::MatrixXd::Identity(design.cols(), design.cols())).selfadjointView<Eigen::Lower>()
	);
	Eigen::VectorXd resid = response - design * llt_of_xtx.solve(design.transpose() * response);
	return resid.squaredNorm() / (response.size() - static_cast<double>(include_mean));
}

// std::vector<Eigen::MatrixXd> sim_mar_export(int num_sim,
// 																		 				int num_burn,
// 																		 				Eigen::SparseMatrix<double> init,
// 																		 				Eigen::MatrixXd row_coef, Eigen::MatrixXd col_coef,
// 																		 				Eigen::MatrixXd row_sig, Eigen::MatrixXd col_sig) {
// 	std::vector<Eigen::MatrixXd> y(num_sim);
// 	int nrow = row_coef.cols();
// 	int ncol = col_coef.cols();
// 	Eigen::MatrixXd error(nrow, ncol);
// 	for (int i = 0; i < num_burn; ++i) {
// 		Eigen::MatrixXd temp_y(nrow, ncol);
// 		baecon::baymar::update_mar(temp_y, init, row_coef, col_coef);
// 		error = baecon::bvhar::sim_mn(
// 			Eigen::MatrixXd::Zero(nrow, ncol),
// 			row_sig, col_sig, false
// 		);
// 		temp_y += error;
// 		baecon::baymar::update_x(init, temp_y, nrow, ncol);
// 	}
// 	for (int i = 0; i < num_sim; ++i) {
// 		y[i].setZero(nrow, ncol);
// 		baecon::baymar::update_mar(y[i], init, row_coef, col_coef);
// 		error = baecon::bvhar::sim_mn(
// 			Eigen::MatrixXd::Zero(nrow, ncol),
// 			row_sig, col_sig, false
// 		);
// 		y[i] += error;
// 		baecon::baymar::update_x(init, y[i], nrow, ncol);
// 	}
// 	return y;
// }
