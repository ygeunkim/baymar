#ifndef BAYMAR_BAYES_MISC_COEF_HELPER_H
#define BAYMAR_BAYES_MISC_COEF_HELPER_H

#include <bvhar/utils>

namespace baymar {

inline void draw_coefsig_row(std::vector<Eigen::MatrixXd>& row_params, std::vector<Eigen::MatrixXd> col_params,
												 		 Eigen::Ref<Eigen::MatrixXd> prior_mean, Eigen::Ref<Eigen::MatrixXd> prior_prec,
												 		 Eigen::Ref<Eigen::MatrixXd> iw_scl,
												 		 double iw_df, int num_mat, int other_dim,
												 		 std::vector<Eigen::SparseMatrix<double>>& x, std::vector<Eigen::MatrixXd>& y,
												 		 BHRNG& rng) {
	Eigen::MatrixXd post_cov = prior_prec;
	for (const auto& x_t : x) {
		post_cov += x_t * col_params[0] * col_params[1].inverse() * col_params[0].transpose() * x_t.transpose();
	}
	Eigen::MatrixXd post_solve = prior_prec * prior_mean;
	for (int i = 0; i < num_mat; ++i) {
		post_solve += x[i] * col_params[0] * col_params[1].inverse() * y[i].transpose();
	}
	Eigen::LLT<Eigen::MatrixXd> llt_of_prec(post_cov.selfadjointView<Eigen::Lower>());
	Eigen::MatrixXd post_mean = llt_of_prec.solve(post_solve);
	double post_df = iw_df + num_mat * other_dim;
	Eigen::MatrixXd post_iw_scl = iw_scl + prior_mean.transpose() * prior_prec * prior_mean - post_mean.transpose() * post_cov * post_mean;
	for (const auto& y_t : y) {
		post_iw_scl += y_t * col_params[1].inverse() * y_t.transpose();
	}
	Eigen::MatrixXd iw_lower = bvhar::sim_iw_tri(post_iw_scl, post_df, rng).triangularView<Eigen::Lower>();
	row_params[1] = iw_lower * iw_lower.transpose();
	for (int i = 0; i < prior_mean.cols(); ++i) {
		for (int j = 0; j < prior_mean.rows(); ++j) {
			row_params[0].col(i)[j] = bvhar::normal_rand(rng);
		}
	}
	row_params[0] = llt_of_prec.matrixU().solve(row_params[0] * iw_lower.transpose());
	row_params[0] += post_mean;
}

inline void draw_coefsig_col(std::vector<Eigen::MatrixXd>& col_params, std::vector<Eigen::MatrixXd> row_params,
												 		 Eigen::Ref<Eigen::MatrixXd> prior_mean, Eigen::Ref<Eigen::MatrixXd> prior_prec,
												 		 Eigen::Ref<Eigen::MatrixXd> iw_scl,
												 		 double iw_df, int num_mat, int other_dim,
												 		 std::vector<Eigen::SparseMatrix<double>>& x, std::vector<Eigen::MatrixXd>& y,
												 		 BHRNG& rng) {
	Eigen::MatrixXd post_cov = prior_prec;
	for (const auto& x_t : x) {
		post_cov += x_t.transpose() * row_params[0] * row_params[1].inverse() * row_params[0].transpose() * x_t;
	}
	Eigen::MatrixXd post_solve = prior_prec * prior_mean;
	for (int i = 0; i < num_mat; ++i) {
		post_solve += x[i].transpose() * row_params[0] * row_params[1].inverse() * y[i];
	}
	Eigen::LLT<Eigen::MatrixXd> llt_of_prec(post_cov.selfadjointView<Eigen::Lower>());
	Eigen::MatrixXd post_mean = llt_of_prec.solve(post_solve);
	double post_df = iw_df + num_mat * other_dim;
	Eigen::MatrixXd post_iw_scl = iw_scl + prior_mean.transpose() * prior_prec * prior_mean - post_mean.transpose() * post_cov * post_mean;
	for (const auto& y_t : y) {
		post_iw_scl += y_t.transpose() * row_params[1].inverse() * y_t;
	}
	Eigen::MatrixXd iw_lower = bvhar::sim_iw_tri(post_iw_scl, post_df, rng).triangularView<Eigen::Lower>();
	col_params[1] = iw_lower * iw_lower.transpose();
	for (int i = 0; i < prior_mean.cols(); ++i) {
		for (int j = 0; j < prior_mean.rows(); ++j) {
			col_params[0].col(i)[j] = bvhar::normal_rand(rng);
		}
	}
	col_params[0] = llt_of_prec.matrixU().solve(col_params[0] * iw_lower.transpose()) + post_mean;
}

} // namespace baymar

#endif // BAYMAR_BAYES_MISC_COEF_HELPER_H