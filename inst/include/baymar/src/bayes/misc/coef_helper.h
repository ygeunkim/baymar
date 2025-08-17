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
template <bool isRow = true, typename xType = Eigen::SparseMatrix<double>>
inline void draw_coef_sig(
	Eigen::Ref<Eigen::MatrixXd> coef, Eigen::Ref<Eigen::MatrixXd> sig_lower,
	Eigen::Ref<Eigen::MatrixXd> other_coef, Eigen::Ref<Eigen::MatrixXd> other_sig_lower,
	Eigen::Ref<Eigen::MatrixXd> prior_mean, Eigen::Ref<Eigen::VectorXd> prior_prec,
	Eigen::Ref<Eigen::MatrixXd> iw_scl,
	double iw_df, int num_mat, int other_dim,
	const std::vector<xType>& x, const std::vector<Eigen::MatrixXd>& y,
	BVHAR_BHRNG& rng
) {
	using is_row = std::integral_constant<bool, isRow>;
	Eigen::MatrixXd post_cov = prior_prec.asDiagonal();
	Eigen::MatrixXd post_solve = prior_prec.asDiagonal() * prior_mean;
	Eigen::MatrixXd post_iw_scl = iw_scl + prior_mean.transpose() * prior_prec.asDiagonal() * prior_mean;
	Eigen::MatrixXd inv_sig_coef_x, inv_sig_y;
	for (int i = 0; i < num_mat; ++i) {
		if (is_row::value) {
			inv_sig_coef_x = other_sig_lower.triangularView<Eigen::Lower>().solve(other_coef.transpose() * x[i].transpose());
			inv_sig_y = other_sig_lower.triangularView<Eigen::Lower>().solve(y[i].transpose());
		} else {
			inv_sig_coef_x = other_sig_lower.triangularView<Eigen::Lower>().solve(other_coef.transpose() * x[i]);
			inv_sig_y = other_sig_lower.triangularView<Eigen::Lower>().solve(y[i]);
		}
		post_cov += inv_sig_coef_x.transpose() * inv_sig_coef_x;
		post_solve += inv_sig_coef_x.transpose() * inv_sig_y;
		post_iw_scl += inv_sig_y.transpose() * inv_sig_y;
	}
	Eigen::LLT<Eigen::MatrixXd> llt_of_prec(post_cov.selfadjointView<Eigen::Lower>());
	double temp_penalty = .0001;
	do {
		llt_of_prec.compute((
			post_cov + temp_penalty * Eigen::MatrixXd::Identity(post_cov.rows(), post_cov.cols())
		).selfadjointView<Eigen::Lower>());
		// post_cov.diagonal().array() += temp_penalty;
		// llt_of_prec.compute(post_cov.selfadjointView<Eigen::Lower>());
		temp_penalty *= 2;
	} while (llt_of_prec.info() != Eigen::Success && temp_penalty < .1);
	if (llt_of_prec.info() != Eigen::Success) {
		eigen_assert("LLT failed in precision sampler.");
	}
	Eigen::MatrixXd post_mean = llt_of_prec.solve(post_solve);
	post_iw_scl -= post_mean.transpose() * post_cov * post_mean;
	double post_df = iw_df + num_mat * other_dim;
	sig_lower = bvhar::sim_iw_tri(post_iw_scl, post_df, rng);
	for (int i = 0; i < prior_mean.rows(); ++i) {
		for (int j = 0; j < prior_mean.cols(); ++j) {
			coef(i, j) = bvhar::normal_rand(rng); // MN(0, I_n, I_k)
		}
	}
	coef = llt_of_prec.matrixU().solve(coef * sig_lower.transpose()) + post_mean;
}

} // namespace baymar

#endif // BAYMAR_BAYES_MISC_COEF_HELPER_H