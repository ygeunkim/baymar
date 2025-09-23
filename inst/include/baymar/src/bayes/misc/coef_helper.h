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
	Eigen::Ref<const Eigen::MatrixXd> other_coef, Eigen::Ref<const Eigen::MatrixXd> other_sig_lower,
	Eigen::Ref<const Eigen::MatrixXd> prior_mean, Eigen::Ref<const Eigen::VectorXd> prior_prec,
	Eigen::Ref<const Eigen::MatrixXd> iw_scl,
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
	Eigen::LLT<Eigen::MatrixXd> llt_of_prec;
	double temp_penalty = 0;
	do {
		llt_of_prec.compute((
			post_cov + temp_penalty * Eigen::MatrixXd::Identity(post_cov.rows(), post_cov.cols())
		).selfadjointView<Eigen::Lower>());
		// post_cov.diagonal().array() += temp_penalty;
		// llt_of_prec.compute(post_cov.selfadjointView<Eigen::Lower>());
		temp_penalty += .01;
	} while (llt_of_prec.info() == Eigen::NumericalIssue && temp_penalty < .1);
	if (llt_of_prec.info() == Eigen::NumericalIssue) {
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
	// Indentifiability restriction
	// M = e_k^T otimes (I_p otimes e_k^T) = M_1 otimes M_2
	// B = B_u + V_1 (B_0 - M_2 B_u M_1^T) V_2^T
	// V_1 = K_B^{-1} M_2^T (M_2 K_B^{-1} M_2^T)^{-1}
	// V_2 = Sigma_c M_1^T (M_1 Sigma_c M_1^T)^{-1}
	// if (!is_row::value) {
	if (!is_row::value && std::is_same<xType, Eigen::SparseMatrix<double>>::value) {
		int num_col = prior_mean.cols();
		int lag = prior_mean.rows() / num_col; // when B = (B_1, ..., B_p)^T
		Eigen::MatrixXd left_map = Eigen::VectorXd::Unit(num_col, 0).transpose();
		Eigen::MatrixXd right_map = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(lag, lag), left_map).eval();
		// Eigen::SparseMatrix<double> left_map = Eigen::VectorXd::Unit(num_col, 0).transpose().sparseView();;
		// Eigen::SparseMatrix<double> right_map = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(lag, lag), left_map).eval().sparseView();
		Eigen::MatrixXd v2_inv = left_map * sig_lower;
		Eigen::LLT<Eigen::MatrixXd> v2_llt(v2_inv * v2_inv.transpose());
		// Eigen::MatrixXd v1_inv = llt_of_prec.matrixL().solve(right_map.transpose().toDense());
		Eigen::MatrixXd v1_inv = llt_of_prec.matrixL().solve(right_map.transpose());
		Eigen::LLT<Eigen::MatrixXd> v1_llt(v1_inv.transpose() * v1_inv);
		Eigen::MatrixXd V1_trans = llt_of_prec.matrixL().solve<Eigen::OnTheRight>(v1_llt.solve(v1_inv.transpose()));
		Eigen::MatrixXd V2_trans = v2_llt.solve(v2_inv * sig_lower.transpose());
		coef += V1_trans.transpose() * (Eigen::VectorXd::Ones(lag) - right_map * coef * left_map.transpose()) * V2_trans;
		// The following is slow when matrix is large
		// coef += llt_of_prec.matrixU().solve(v1_inv * v1_llt.solve(
		// 	(Eigen::VectorXd::Ones(lag) - right_map * coef * left_map.transpose()) * v2_llt.solve(v2_inv * sig_lower.transpose())
		// ));
	}
}

} // namespace baymar

#endif // BAYMAR_BAYES_MISC_COEF_HELPER_H