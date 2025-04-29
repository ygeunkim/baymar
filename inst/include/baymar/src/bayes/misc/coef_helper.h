#ifndef BAYMAR_BAYES_MISC_COEF_HELPER_H
#define BAYMAR_BAYES_MISC_COEF_HELPER_H

#include <bvhar/utils>

namespace baymar {

inline void draw_coefsig(std::vector<Eigen::MatrixXd>& coef_sig, std::vector<Eigen::MatrixXd> other_params,
												 Eigen::Ref<Eigen::MatrixXd> prior_mean, Eigen::Ref<Eigen::MatrixXd> prior_prec,
												 Eigen::Ref<Eigen::MatrixXd> iw_scl,
												 double iw_df, int num_mat, int other_dim,
												 std::vector<Eigen::SparseMatrix<double>>& x, std::vector<Eigen::MatrixXd>& y,
												 BHRNG& rng) {
	Eigen::MatrixXd post_cov = prior_prec;
	for (const auto& x_t : x) {
		post_cov += x_t * other_params[0] * other_params[1].inverse() * other_params[0].transpose() * x_t.transpose();
	}
	Eigen::MatrixXd post_solve = prior_prec * prior_mean;
	for (int i = 0; i < num_mat; ++i) {
		post_solve += x[i] * other_params[0] * other_params[1].inverse() * y[i].transpose();
	}
	// Eigen::MatrixXd post_mean = post_cov.llt().solve(post_solve);
	Eigen::LLT<Eigen::MatrixXd> llt_of_prec(post_cov.selfadjointView<Eigen::Lower>());
	Eigen::MatrixXd post_mean = llt_of_prec.solve(post_solve);
	double post_df = iw_df + num_mat * other_dim;
	Eigen::MatrixXd post_iw_scl = iw_scl + prior_mean.transpose() * prior_prec * prior_mean - post_mean.transpose() * post_cov * post_mean;
	for (const auto& y_t : y) {
		post_iw_scl += y_t * other_params[1].inverse() * y_t.transpose();
	}
	Eigen::MatrixXd iw_lower = bvhar::sim_iw_tri(post_iw_scl, post_df, rng).triangularView<Eigen::Lower>();
	coef_sig[1] = iw_lower * iw_lower.transpose();
	for (int i = 0; i < prior_mean.cols(); ++i) {
		for (int j = 0; j < prior_mean.rows(); ++j) {
			coef_sig[0].col(i)[j] = bvhar::normal_rand(rng);
		}
	}
	coef_sig[0] = llt_of_prec.matrixU().solve(coef_sig[0] * iw_lower);
	// coef_sig[0] = iw_lower.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(coef_sig[0]) + post_mean;
	// coef_sig = bvhar::sim_mn_iw(post_mean, post_cov, post_iw_scl, post_df, true, rng);
}

inline void draw_coefsig_col(std::vector<Eigen::MatrixXd>& coef_sig, std::vector<Eigen::MatrixXd> other_params,
												 		 Eigen::Ref<Eigen::MatrixXd> prior_mean, Eigen::Ref<Eigen::MatrixXd> prior_prec,
												 		 Eigen::Ref<Eigen::MatrixXd> iw_scl,
												 		 double iw_df, int num_mat, int other_dim,
												 		 std::vector<Eigen::SparseMatrix<double>>& x, std::vector<Eigen::MatrixXd>& y,
												 		 BHRNG& rng) {
	Eigen::MatrixXd post_cov = prior_prec;
	for (const auto& x_t : x) {
		post_cov += x_t.transpose() * other_params[0] * other_params[1].inverse() * other_params[0].transpose() * x_t;
	}
	Eigen::MatrixXd post_solve = prior_prec * prior_mean;
	for (int i = 0; i < num_mat; ++i) {
		post_solve += x[i].transpose() * other_params[0] * other_params[1].inverse() * y[i];
	}
	// Eigen::MatrixXd post_mean = post_cov.llt().solve(post_solve);
	Eigen::LLT<Eigen::MatrixXd> llt_of_prec(post_cov.selfadjointView<Eigen::Lower>());
	Eigen::MatrixXd post_mean = llt_of_prec.solve(post_solve);
	double post_df = iw_df + num_mat * other_dim;
	Eigen::MatrixXd post_iw_scl = iw_scl + prior_mean.transpose() * prior_prec * prior_mean - post_mean.transpose() * post_cov * post_mean;
	for (const auto& y_t : y) {
		post_iw_scl += y_t.transpose() * other_params[1].inverse() * y_t;
	}
	// coef_sig = bvhar::sim_mn_iw(post_mean, post_cov, post_iw_scl, post_df, true, rng);
	Eigen::MatrixXd iw_lower = bvhar::sim_iw_tri(post_iw_scl, post_df, rng).triangularView<Eigen::Lower>();
	coef_sig[1] = iw_lower * iw_lower.transpose();
	for (int i = 0; i < prior_mean.cols(); ++i) {
		for (int j = 0; j < prior_mean.rows(); ++j) {
			coef_sig[0].col(i)[j] = bvhar::normal_rand(rng);
		}
	}
	coef_sig[0] = llt_of_prec.matrixU().solve(coef_sig[0] * iw_lower);
	// coef_sig[0] = llt_of_prec.matrixU().solve(coef_sig[0]);
	// coef_sig[0] = iw_lower.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(coef_sig[0]) + post_mean;
}

} // namespace baymar

#endif // BAYMAR_BAYES_MISC_COEF_HELPER_H