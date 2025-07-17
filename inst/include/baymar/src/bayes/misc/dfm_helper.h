#ifndef BAYMAR_BAYES_MISC_DFM_HELPER_H
#define BAYMAR_BAYES_MISC_DFM_HELPER_H

#include <bvhar/utils>

namespace baymar {

/**
 * @brief Generate factor matrix of vec(F_t) modeling
 * 
 * vec(F_t) = f_t = H_1 f_{t - 1} + ... + H_s f_{t - s} + u_t
 * u_t ~ N(0, Lambda)
 * H_i = diag(rho_{1, 1, i}, rho_{2, 1, i}, ..., rho_{p1, p2, i})
 * 
 * @param factor_mat Factor matrices
 * @param factor_lag Lag of the factor model (s)
 * @param rows_factor Factor matrix number of rows
 * @param cols_factor Factor matrix number of columns
 * @param fac_coef_diag Each column is the j-th lag coefficient diagonal: p1*p2 x s dimension matrix from H_i = diag(rho_{1, i}, ..., rho_{p1 * p2, i})
 * @param fac_lambda Diagonal of factor model variance: p1*p2 size vector
 * @param row_coef Transpose of MAR row coefficient
 * @param row_sig_lower MAR row covariance L decomposition
 * @param col_coef Transpose of MAR column coefficient
 * @param col_sig_lower MAR column covariance L decomposition
 * @param y Matrix time series
 * @param rng boost rng
 */
inline void draw_dfm_factor(std::vector<Eigen::MatrixXd>& factor_mat, int factor_lag, int rows_factor, int cols_factor,
													  Eigen::Ref<Eigen::MatrixXd> fac_coef_diag, Eigen::Ref<Eigen::VectorXd> fac_lambda,
													  Eigen::Ref<const Eigen::MatrixXd> row_coef, Eigen::Ref<const Eigen::MatrixXd> row_sig_lower,
													  Eigen::Ref<const Eigen::MatrixXd> col_coef, Eigen::Ref<const Eigen::MatrixXd> col_sig_lower,
													  std::vector<Eigen::MatrixXd>& y, BHRNG& rng) {
	Eigen::MatrixXd col_inv_sig_coef = col_sig_lower.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(
		col_sig_lower.triangularView<Eigen::Lower>().solve(col_coef).transpose()
	);
	Eigen::MatrixXd row_inv_sig_coef = row_sig_lower.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(
		row_sig_lower.triangularView<Eigen::Lower>().solve(row_coef).transpose()
	);
	Eigen::MatrixXd post_solve = bvhar::kronecker_eigen(col_inv_sig_coef, row_inv_sig_coef);
	Eigen::MatrixXd post_cov = post_solve * bvhar::kronecker_eigen(col_coef, row_coef);
	Eigen::LLT<Eigen::MatrixXd> llt_of_prec;
	int len_factor = rows_factor * cols_factor;
	Eigen::VectorXd vec_normal(len_factor);
	Eigen::VectorXd post_mean(len_factor);
	Eigen::VectorXd prec_t(len_factor);
	// 1) t = p + 1, ..., p + s with Lambda = diag(lambda^2 / (1 - sum_i^s rho_i^2))
	prec_t = (1 - fac_coef_diag.rowwise().squaredNorm().array()) / fac_lambda.array();
	llt_of_prec.compute((post_cov + prec_t.asDiagonal().toDenseMatrix()).selfadjointView<Eigen::Lower>());
	for (int i = 0; i < factor_lag; ++i) {
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

/**
 * @brief Generate the diagonal of vec(F_t) model variance
 * 
 * @param fac_lambda Diagonal of factor model variance: p1*p2 size vector
 * @param factor_lag Lag of the factor model (s)
 * @param ig_shp Inverse-gamma shape
 * @param ig_scl Inverse-gamma scale
 * @param factor_mat Factor matrix
 * @param fac_coef_diag Each column is the j-th lag coefficient diagonal: p1*p2 x s dimension matrix from H_i = diag(rho_{1, i}, ..., rho_{p1 * p2, i})
 * @param rng boost rng
 */
inline void draw_dfm_prec(Eigen::Ref<Eigen::VectorXd> fac_lambda, int factor_lag,
													Eigen::Ref<Eigen::VectorXd> ig_shp, Eigen::Ref<Eigen::VectorXd> ig_scl,
													std::vector<Eigen::MatrixXd>& factor_mat, Eigen::Ref<Eigen::MatrixXd> fac_coef_diag,
													BHRNG& rng) {
	int num_design = factor_mat.size();
	int rows_factor = factor_mat[0].rows();
	int cols_factor = factor_mat[0].cols();
	double post_scl, coef_square, resid;
	for (int i = 0; i < rows_factor * cols_factor; ++i) {
		int row_id = i % rows_factor;
		int col_id = i / rows_factor;
		post_scl = ig_scl[i];
		// 1) t = p + 1, ..., p + s => sum_t f_{jk, t}^2 (1 - sum_i rho_{jk,i}^2)
		coef_square = 1 - fac_coef_diag.row(i).squaredNorm();
		for (int j = 0; j < factor_lag; ++j) {
			post_scl += factor_mat[j](row_id, col_id) * factor_mat[j](row_id, col_id) * coef_square / 2;
		}
		// 2) t = p + s + 1, ..., T => sum_t (f_{jk, t} - rho_{jk, 1} f_{jk, t - 1} - ... - rho_{jk, s} f_{jk, t - s})^2
		for (int j = factor_lag; j < num_design; ++j) {
			resid = factor_mat[j](row_id, col_id);
			for (int l = 0; l < factor_lag; ++l) {
				resid -= fac_coef_diag(i, l) * factor_mat[j - l - 1](row_id, col_id);
			}
			post_scl += resid * resid / 2;
		}
		fac_lambda[i] = 1 / bvhar::gamma_rand(ig_shp[i] + num_design / 2, 1 / post_scl, rng);
	}
}

/**
 * @brief Log of proposal density for factor model coefficient
 * 
 * @param cand_coef Candidate rho_{jk, 1}, ... rho_{jk, s}
 * @param lambda_i lambda^2
 * @param fac_init Initial factor
 * @return double 
 */
inline double compute_dfmcoef_logdens(Eigen::Ref<const Eigen::VectorXd> cand_coef,
																			double lambda_i, Eigen::Ref<const Eigen::VectorXd> fac_init) {
	double res = 0;
	double variance = lambda_i / (1 - cand_coef.squaredNorm());
	for (int i = 0; i < fac_init.size(); ++i) {
		res += -log(variance) / 2 - fac_init[i] * fac_init[i] / (2 * variance);
	}
	return res;
}

/**
 * @brief Build response and design for factor linear equation
 * 
 * f_{j,k} = F_{j,k} rho_{j,k} + u_{j,k}
 * 
 * @param factor_response Vector f_{j,k} = (f_{j,k, s + 1}, ..., f_{j,k, T})^T
 * @param factor_design Matrix F_{j,k} = rbind((f_{j,k,1}, ... f_{j,k,s}), (f_{j,k,2}, ... f_{j,k,s + 1}), ..., (f_{j,k,T-s}, ... f_{j,k,T}))
 * @param row_id j
 * @param col_id k
 * @param factor_mat F_t
 * @param factor_lag s
 */
inline void build_factor_lin(Eigen::Ref<Eigen::VectorXd> factor_response, Eigen::Ref<Eigen::MatrixXd> factor_design,
														 int row_id, int col_id,
														 std::vector<Eigen::MatrixXd>& factor_mat, int factor_lag, int num_design) {
	// for (int i = 0; i < num_design; ++i) {
	// 	factor_response[i] = factor_mat[i + factor_lag](row_id, col_id);
	// 	for (int j = 0; j < factor_lag; ++j) {
	// 		factor_design(i, j) = factor_mat[i + j](row_id, col_id);
	// 	}
	// }
	int len_factor_series = factor_mat.size();
	Eigen::VectorXd full_fjk(len_factor_series); // (f_{j,k, 1}, ..., f_{j,k, T})^T
	for (int i = 0; i < len_factor_series; ++i) {
		full_fjk[i] = factor_mat[i](row_id, col_id);
	}
	factor_response = full_fjk.tail(num_design);
	for (int i = 0; i < num_design; ++i) {
		factor_design.row(i) = full_fjk.segment(i, factor_lag);
	}
}

/**
 * @brief Draw factor model coefficient
 * 
 * @param fac_coef_diag Each column is the j-th lag coefficient diagonal: p1*p2 x s dimension matrix from H_i = diag(rho_{1, i}, ..., rho_{p1 * p2, i})
 * @param fac_lambda Diagonal of factor model variance: p1*p2 size vector
 * @param prior_mean Prior mean of the factor coefficient
 * @param prior_prec Prior precision of the factor coefficient
 * @param factor_mat Factor matrix
 * @param factor_lag Lag of the factor model (s)
 * @param rng boost rng
 */
inline void draw_dfm_coef(Eigen::Ref<Eigen::MatrixXd> fac_coef_diag, Eigen::Ref<Eigen::VectorXd> fac_lambda,
													Eigen::Ref<Eigen::VectorXd> prior_mean, Eigen::Ref<Eigen::VectorXd> prior_prec,
													std::vector<Eigen::MatrixXd>& factor_mat, int factor_lag,
													BHRNG& rng) {
	int num_design = factor_mat.size() - factor_lag;
	int num_coef = fac_coef_diag.rows(); // p1 * p2
	int rows_factor = factor_mat[0].rows(); // p1
	double numerator, denom;
	Eigen::MatrixXd factor_design(num_design, factor_lag);
	Eigen::VectorXd factor_response(num_design);
	Eigen::VectorXd cand_rho(num_coef);
	Eigen::VectorXd normal_vector(factor_lag);
	Eigen::VectorXd post_mean(num_coef);
	Eigen::LLT<Eigen::MatrixXd> llt_of_prec;
	for (int i = 0; i < num_coef; ++i) {
		int row_id = i % rows_factor;
		int col_id = i / rows_factor;
		build_factor_lin(factor_response, factor_design, row_id, col_id, factor_mat, factor_lag, num_design);
		llt_of_prec.compute((prior_prec.asDiagonal().toDenseMatrix() + factor_design.transpose() * factor_design / fac_lambda[i]).selfadjointView<Eigen::Lower>());
		post_mean = llt_of_prec.solve(prior_prec.cwiseProduct(prior_mean) + factor_design.transpose() * factor_response / fac_lambda[i]);
		for (int j = 0; j < factor_lag; ++j) {
			normal_vector[j] = bvhar::normal_rand(rng);
		}
		cand_rho = post_mean + llt_of_prec.matrixU().solve(normal_vector);
		numerator = compute_dfmcoef_logdens(cand_rho, fac_lambda[i], factor_design.row(0));
		denom = compute_dfmcoef_logdens(fac_coef_diag.row(i), fac_lambda[i], factor_design.row(0));
		if (log(bvhar::unif_rand(rng)) < std::min(numerator - denom, 0.0)) {
			fac_coef_diag.row(i) = cand_rho.transpose();
		}
	}
}

} // namespace baymar

#endif // BAYMAR_BAYES_MISC_DFM_HELPER_H