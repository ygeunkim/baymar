#ifndef BAYMAR_BAYES_MISC_DFM_HELPER_H
#define BAYMAR_BAYES_MISC_DFM_HELPER_H

#include <bvhar/utils>

namespace baymar {

// Generate F_t: p1 x p2, t = p + 1, ..., T
// vec(F_t) = f_t = H_1 f_{t - 1} + ... + H_s f_{t - s} + u_t
// u_t ~ N(0, Lambda)
// H_i = diag(rho_{1, i}, ..., rho_{p1 * p2, i})
// fac_coef_diag: p1*p2 x s
// fac_lambda: p1*p2 x 1
inline void draw_dfm_factor(std::vector<Eigen::MatrixXd>& factor_mat, int factor_lag,
													  Eigen::Ref<Eigen::MatrixXd> fac_coef_diag, Eigen::Ref<Eigen::VectorXd> fac_lambda,
													  Eigen::Ref<Eigen::MatrixXd> row_coef, Eigen::Ref<Eigen::MatrixXd> row_sig_lower,
													  Eigen::Ref<Eigen::MatrixXd> col_coef, Eigen::Ref<Eigen::MatrixXd> col_sig_lower,
													  std::vector<Eigen::MatrixXd>& y, BHRNG& rng) {
	Eigen::MatrixXd col_inv_sig_coef = col_sig_lower.triangularView<Eigen::Lower>().solve(col_coef);
	Eigen::MatrixXd row_inv_sig_coef = row_sig_lower.triangularView<Eigen::Lower>().solve(row_coef);
	Eigen::MatrixXd post_solve = bvhar::kronecker_eigen(
		col_sig_lower.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(col_inv_sig_coef.transpose()),
		row_sig_lower.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(row_inv_sig_coef.transpose())
	);
	Eigen::MatrixXd post_cov = post_solve * bvhar::kronecker_eigen(col_coef, row_coef);
	Eigen::LLT<Eigen::MatrixXd> llt_of_prec;
	int cols_factor = factor_mat[0].cols();
	int len_factor = factor_mat[0].rows() * cols_factor;
	Eigen::VectorXd vec_normal(len_factor);
	Eigen::VectorXd post_mean(len_factor);
	Eigen::VectorXd prec_t(len_factor);
	// 1) t = p + 1, ..., p + s with Lambda = diag(lambda^2 / (1 - sum_i^s rho_i^2))
	prec_t = (1 - fac_coef_diag.rowwise().squaredNorm().array()) / fac_lambda.array().square();
	for (int i = 0; i < factor_lag; ++i) {
		llt_of_prec.compute((post_cov + prec_t.asDiagonal().toDenseMatrix()).selfadjointView<Eigen::Lower>());
		post_mean = llt_of_prec.solve(post_solve * y[i].reshaped());
		for (int j = 0; j < len_factor; ++j) {
			vec_normal[j] = bvhar::normal_rand(rng);
		}
		factor_mat[i] = bvhar::unvectorize(post_mean + llt_of_prec.matrixU().solve(vec_normal), cols_factor);
	}
	// 2) t = p + s + 1, ..., T with Lambda = diag(lambda^2)
	Eigen::VectorXd fac_ar(len_factor); // sum_{i = 1}^s H_i f_{t - i}
	prec_t = 1 / fac_lambda.array().square();
	llt_of_prec.compute((post_cov + prec_t.asDiagonal().toDenseMatrix()).selfadjointView<Eigen::Lower>());
	for (int i = factor_lag; i < y.size(); ++i) {
		fac_ar.setZero();
		for (int j = 0; j < factor_lag; ++j) {
			fac_ar.array() += fac_coef_diag.col(j).array() * factor_mat[i - j - 1].reshaped().array();
		}
		post_mean = llt_of_prec.solve(post_solve * y[i].reshaped() + fac_ar.cwiseProduct(prec_t));
		for (int j = 0; j < len_factor; ++j) {
			vec_normal[j] = bvhar::normal_rand(rng);
		}
		factor_mat[i] = bvhar::unvectorize(post_mean + llt_of_prec.matrixU().solve(vec_normal), cols_factor);
	}
}

} // namespace baymar

#endif // BAYMAR_BAYES_MISC_DFM_HELPER_H