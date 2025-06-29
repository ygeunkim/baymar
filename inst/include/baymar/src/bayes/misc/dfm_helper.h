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
	prec_t = (1 - fac_coef_diag.rowwise().squaredNorm().array()) / fac_lambda.array();
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
	prec_t = 1 / fac_lambda.array();
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

// Generate lambda^2
inline void draw_dfm_prec(Eigen::Ref<Eigen::VectorXd> fac_lambda, int factor_lag,
													Eigen::Ref<Eigen::VectorXd> ig_shp, Eigen::Ref<Eigen::VectorXd> ig_scl,
													std::vector<Eigen::MatrixXd>& factor_mat, Eigen::Ref<Eigen::MatrixXd> fac_coef_diag,
													BHRNG& rng) {
	int num_design = factor_mat.size();
	int rows_factor = factor_mat[0].rows();
	int cols_factor = factor_mat[0].cols();
	// Eigen::VectorXd ssr_item(num_design);
	double post_scl, coef_square, resid;
	for (int i = 0; i < rows_factor * cols_factor; ++i) {
		int row_id = i % rows_factor;
		int col_id = i / rows_factor;
		post_scl = ig_scl[i];
		// 1) t = p + 1, ..., p + s => sum_t f_{jk, t}^2 (1 - sum_i rho_{jk,i}^2)
		coef_square = 1 - fac_coef_diag.row(i).squaredNorm();
		for (int j = 0; j < factor_lag; ++j) {
			// ssr_item[j] = factor_mat[j](row_id, col_id) * factor_mat[j](row_id, col_id) * coef_square;
			post_scl += factor_mat[j](row_id, col_id) * factor_mat[j](row_id, col_id) * coef_square;
		}
		// 2) t = p + s + 1, ..., T => sum_t (f_{jk, t} - rho_{jk, 1} f_{jk, t - 1} - ... - rho_{jk, s} f_{jk, t - s})^2
		for (int j = factor_lag; j < num_design; ++j) {
			// ssr_item[j] = factor_mat[j](row_id, col_id);
			resid = factor_mat[j](row_id, col_id);
			for (int l = 0; l < factor_lag; ++l) {
				resid -= fac_coef_diag(i, l) * factor_mat[j - l - 1](row_id, col_id);
				// ssr_item[j] -= (fac_coef_diag(i, l) * factor_mat[l + factor_lag - l - 1](row_id, col_id)) * (fac_coef_diag(i, l) * factor_mat[j - l - 1](row_id, col_id));
			}
			// ssr_item[j] = resid * resid;
			post_scl += resid * resid;
		}
		fac_lambda[i] = 1 / bvhar::gamma_rand(
			ig_shp[i] + num_design / 2,
			// (ig_scl[i] + ssr_item.sum()) / 2,
			post_scl / 2,
			rng
		);
	}
	// int id = 0;
	// Eigen::VectorXd fac_vector(factor_lag);
	// Eigen::VectorXd fac_resid(num_design - factor_lag);
	// for (int j = 0; j < cols_factor; ++j) {
	// 	for (int i = 0; i < rows_factor; ++i) {
	// 		id = i + j * rows_factor;
	// 		// 1) t = p + 1, ..., p + s => sum_t f_{jk, t}^2 (1 - sum_i rho_{jk,i}^2)
	// 		for (int t = 0; t < factor_lag; ++t) {
	// 			fac_vector[t] = factor_mat[t](i, j);
	// 		}
	// 		// 2) t = p + s + 1, ..., T => sum_t (f_{jk, t} - rho_{jk, 1} f_{jk, t - 1} - ... - rho_{jk, s} f_{jk, t - s})^2
	// 		for (int t = 0; t < num_design - factor_lag; ++t) {
	// 			fac_resid[t] = factor_mat[t](i, j);
	// 			for (int l = 0; l < factor_lag; ++l) {
	// 				fac_resid[t] -= fac_coef_diag(id, l) * factor_mat[t + factor_lag - l - 1](i, j);
	// 			}
	// 		}
	// 		fac_lambda[id] = 1 / bvhar::gamma_rand(
	// 			ig_shp[id] + num_design / 2,
	// 			(ig_scl[id] + (fac_vector.array().square() * (1 - fac_coef_diag.rowwise().squaredNorm().array())).sum() + fac_resid.squaredNorm()) / 2,
	// 			rng
	// 		);
	// 	}
	// }
}

} // namespace baymar

#endif // BAYMAR_BAYES_MISC_DFM_HELPER_H