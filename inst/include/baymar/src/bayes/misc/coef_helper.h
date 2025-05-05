#ifndef BAYMAR_BAYES_MISC_COEF_HELPER_H
#define BAYMAR_BAYES_MISC_COEF_HELPER_H

#include <bvhar/utils>
#include <type_traits>

namespace baymar {

/**
 * @brief Generate MNIW coefficient and LLT decomposition of Sigma
 * 
 * @tparam isRow Generate row or column
 * @param params (Coefficient, Sigma)
 * @param other_params Other (Coefficient, Sigma)
 * @param prior_mean Prior MN mean of coefficient
 * @param prior_prec Prior MN precision of coefficient
 * @param iw_scl Prior IW scale of Sigma
 * @param iw_df Prior IW shape of Sigma
 * @param num_mat Number of matrix time series
 * @param other_dim When row, column size. When column, row size.
 * @param x X_t
 * @param y Y_t
 * @param rng boost rng
 */
template <bool isRow = true>
inline void draw_coef_sig(
	std::vector<Eigen::MatrixXd>& params, std::vector<Eigen::MatrixXd> other_params,
	Eigen::Ref<Eigen::MatrixXd> prior_mean, Eigen::Ref<Eigen::MatrixXd> prior_prec,
	Eigen::Ref<Eigen::MatrixXd> iw_scl,
	double iw_df, int num_mat, int other_dim,
	std::vector<Eigen::SparseMatrix<double>>& x, std::vector<Eigen::MatrixXd>& y,
	BHRNG& rng
) {
	using is_row = std::integral_constant<bool, isRow>;
	Eigen::MatrixXd post_cov = prior_prec;
	Eigen::MatrixXd post_solve = prior_prec * prior_mean;
	Eigen::MatrixXd post_iw_scl = iw_scl + prior_mean.transpose() * prior_prec * prior_mean;
	Eigen::MatrixXd inv_sig_coef_x, inv_sig_y;
	for (int i = 0; i < num_mat; ++i) {
		if (is_row::value) {
			inv_sig_coef_x = other_params[1].triangularView<Eigen::Lower>().solve(other_params[0].transpose() * x[i].transpose());
			inv_sig_y = other_params[1].triangularView<Eigen::Lower>().solve(y[i].transpose());
		} else {
			inv_sig_coef_x = other_params[1].triangularView<Eigen::Lower>().solve(other_params[0].transpose() * x[i]);
			inv_sig_y = other_params[1].triangularView<Eigen::Lower>().solve(y[i]);
		}
		post_cov += inv_sig_coef_x.transpose() * inv_sig_coef_x;
		post_solve += inv_sig_coef_x.transpose() * inv_sig_y;
		post_iw_scl += inv_sig_y.transpose() * inv_sig_y;
	}
	Eigen::LLT<Eigen::MatrixXd> llt_of_prec(post_cov.selfadjointView<Eigen::Lower>());
	Eigen::MatrixXd post_mean = llt_of_prec.solve(post_solve);
	double post_df = iw_df + num_mat * other_dim;
	params[1] = bvhar::sim_iw_tri(post_iw_scl, post_df, rng);
	for (int i = 0; i < prior_mean.cols(); ++i) {
		for (int j = 0; j < prior_mean.rows(); ++j) {
			params[0].col(i)[j] = bvhar::normal_rand(rng);
		}
	}
	params[0] = llt_of_prec.matrixU().solve(params[0] * params[1].transpose());
	params[0] += post_mean;
}

} // namespace baymar

#endif // BAYMAR_BAYES_MISC_COEF_HELPER_H