#ifndef BAYMAR_BAYES_MISC_COEF_HELPER_H
#define BAYMAR_BAYES_MISC_COEF_HELPER_H

// #include <bvhar/utils>
#include "../../math/random.h"
#include <type_traits>

namespace baecon {
namespace baymar {

// MAR Indentifiability restriction
// M = e_k^T otimes (I_p otimes e_k^T) = M_1 otimes M_2
// B = B_u + V_1 (B_0 - M_2 B_u M_1^T) V_2^T
// V_1 = K_B^{-1} M_2^T (M_2 K_B^{-1} M_2^T)^{-1}
// V_2 = Sigma_c M_1^T (M_1 Sigma_c M_1^T)^{-1}
// inline void restrict_mar_col(int num_col, int lag,
// 														 Eigen::Ref<Eigen::MatrixXd> coef,
// 														 Eigen::Ref<const Eigen::MatrixXd> sig_lower,
// 														 const Eigen::LLT<Eigen::MatrixXd> llt_of_prec) {
// 	Eigen::MatrixXd left_map = Eigen::VectorXd::Unit(num_col, 0).transpose();
// 	// Eigen::MatrixXd right_map = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(lag, lag), left_map).eval();
// 	Eigen::MatrixXd right_map = bvhar::kronecker_eigen(Eigen::MatrixXd::Identity(lag, lag), left_map);
// 	// Eigen::SparseMatrix<double> left_map = Eigen::VectorXd::Unit(num_col, 0).transpose().sparseView();
// 	// Eigen::SparseMatrix<double> right_map = bvhar::kronecker_eigen(Eigen::MatrixXd::Identity(lag, lag), left_map).sparseView();
// 	Eigen::MatrixXd v2_inv = left_map * sig_lower;
// 	Eigen::LLT<Eigen::MatrixXd> v2_llt(v2_inv * v2_inv.transpose());
// 	Eigen::MatrixXd v1_inv = llt_of_prec.matrixL().solve(right_map.transpose());
// 	Eigen::LLT<Eigen::MatrixXd> v1_llt(v1_inv.transpose() * v1_inv);
// 	Eigen::MatrixXd V1_trans = llt_of_prec.matrixL().solve<Eigen::OnTheRight>(v1_llt.solve(v1_inv.transpose()));
// 	Eigen::MatrixXd V2_trans = v2_llt.solve(v2_inv * sig_lower.transpose());
// 	coef += V1_trans.transpose() * (Eigen::VectorXd::Ones(lag) - right_map * coef * left_map.transpose()) * V2_trans;
// }
inline void restrict_mar_col(int num_col, int lag,
														 Eigen::Ref<Eigen::MatrixXd> coef,
														 Eigen::Ref<const Eigen::MatrixXd> sig_lower,
														 Eigen::Ref<const Eigen::MatrixXd> post_cov) {
	Eigen::MatrixXd left_map = Eigen::VectorXd::Unit(num_col, 0).transpose();
	// Eigen::MatrixXd right_map = Eigen::KroneckerProduct(Eigen::MatrixXd::Identity(lag, lag), left_map).eval();
	// Eigen::MatrixXd right_map = bvhar::kronecker_eigen(Eigen::MatrixXd::Identity(lag, lag), left_map);
	Eigen::MatrixXd right_map = bvhar::kronecker_eigen(
		Eigen::MatrixXd::Identity(lag, lag),
		Eigen::VectorXd::Unit(coef.rows() / lag, 0).transpose()
	);
	// Eigen::SparseMatrix<double> left_map = Eigen::VectorXd::Unit(num_col, 0).transpose().sparseView();
	// Eigen::SparseMatrix<double> right_map = bvhar::kronecker_eigen(Eigen::MatrixXd::Identity(lag, lag), left_map).sparseView();
	Eigen::LLT<Eigen::MatrixXd> llt_of_prec;
	double temp_penalty = 0;
	do {
		llt_of_prec.compute((
			post_cov + temp_penalty * Eigen::MatrixXd::Identity(post_cov.rows(), post_cov.cols())
		).selfadjointView<Eigen::Lower>());
		temp_penalty += .01;
	} while (llt_of_prec.info() == Eigen::NumericalIssue && temp_penalty < .1);
	if (llt_of_prec.info() == Eigen::NumericalIssue) {
		eigen_assert("LLT failed in precision sampler.");
	}
	Eigen::MatrixXd v2_inv = left_map * sig_lower;
	Eigen::LLT<Eigen::MatrixXd> v2_llt(v2_inv * v2_inv.transpose());
	Eigen::MatrixXd v1_inv = llt_of_prec.matrixL().solve(right_map.transpose());
	Eigen::LLT<Eigen::MatrixXd> v1_llt(v1_inv.transpose() * v1_inv);
	Eigen::MatrixXd V1_trans = llt_of_prec.matrixL().solve<Eigen::OnTheRight>(v1_llt.solve(v1_inv.transpose()));
	Eigen::MatrixXd V2_trans = v2_llt.solve(v2_inv * sig_lower.transpose());
	coef += V1_trans.transpose() * (Eigen::VectorXd::Ones(lag) - right_map * coef * left_map.transpose()) * V2_trans;
}

// Factor loading identifiability restriction
// R^T Q_1 = I, C^T Q_2 = I
inline void restrict_mat_loading(int num_col, int dim_factor,
																 Eigen::Ref<Eigen::MatrixXd> coef,
																 Eigen::Ref<const Eigen::MatrixXd> sig_lower) {
	// int num_col = prior_mean.cols();
	Eigen::MatrixXd project_restr = Eigen::MatrixXd::Zero(num_col, dim_factor);
	project_restr.topRows(dim_factor).setIdentity();
	// Eigen::MatrixXd v_inv = llt_of_prec.matrixL().solve(project_restr.transpose());
	// Eigen::LLT<Eigen::MatrixXd> v_llt(v_inv.transpose() * v_inv);
	// Eigen::MatrixXd V_trans = llt_of_prec.matrixL().solve<Eigen::OnTheRight>(v_llt.solve(v_inv.transpose()));
	// coef += V_trans.transpose() * (Eigen::MatrixXd::Identity(num_col, num_col) - project_restr * coef);
	Eigen::MatrixXd v_inv = project_restr.transpose() * sig_lower;
	Eigen::LLT<Eigen::MatrixXd> v_llt(v_inv * v_inv.transpose());
	Eigen::MatrixXd V_trans = v_llt.solve(v_inv * sig_lower.transpose());
	// coef.bottomRows(dim_factor) += (Eigen::MatrixXd::Identity(num_col, num_col) - coef.bottomRows(dim_factor) * project_restr) * V_trans;
	coef += (Eigen::MatrixXd::Identity(dim_factor, dim_factor) - coef * project_restr) * V_trans;
	// The following is slow when matrix is large
	// coef += llt_of_prec.matrixU().solve(v1_inv * v1_llt.solve(
	// 	(Eigen::VectorXd::Ones(lag) - right_map * coef * left_map.transpose()) * v2_llt.solve(v2_inv * sig_lower.transpose())
	// ));
}

inline void restrict_mat_loading2(int num_col, int dim_factor,
																  Eigen::Ref<Eigen::MatrixXd> coef,
																  Eigen::Ref<const Eigen::MatrixXd> sig_lower) {
	// int sign_11 = coef(0, 0) > 0 ? 1 : -1;
	// coef /= (sign_11 * coef.squaredNorm());
	int num_restr = dim_factor * (dim_factor + 1) / 2;
	Eigen::SparseMatrix<double> project_restr(num_restr, dim_factor * num_col);
	// Eigen::MatrixXd project_restr = Eigen::MatrixXd::Zero(num_restr, dim_factor * num_col);
	Eigen::VectorXd restr_vec = Eigen::VectorXd::Zero(num_restr);
	int row_id = 0;
	int id = 0;
	std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(num_restr);
	for (int i = 0; i < dim_factor; ++i) {
		for (int j = i; j < dim_factor; ++j) {
			id = i * dim_factor + j;
			triplets.push_back(Eigen::Triplet<double>(row_id, id, 1.0));
			if (i == j) {
				restr_vec[row_id] = 1.0;
			}
			++row_id;
		}
	}
	project_restr.setFromTriplets(triplets.begin(), triplets.end());
	Eigen::MatrixXd u_mat = project_restr.transpose();
	sig_lower.triangularView<Eigen::Lower>().solveInPlace(u_mat);
	sig_lower.triangularView<Eigen::Upper>().solveInPlace(u_mat);
	Eigen::MatrixXd v_mat = (project_restr * u_mat).llt().solve(u_mat.transpose());
	Eigen::VectorXd coef_vec = coef.reshaped();
	coef_vec += v_mat.transpose() * (restr_vec - project_restr * coef_vec);
	coef = bvhar::unvectorize(coef_vec, num_col);
}

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
	Eigen::Ref<Eigen::MatrixXd> other_coef, Eigen::Ref<const Eigen::MatrixXd> other_sig_lower,
	Eigen::Ref<const Eigen::MatrixXd> prior_mean, Eigen::Ref<const Eigen::VectorXd> prior_prec,
	Eigen::Ref<const Eigen::MatrixXd> iw_scl,
	double iw_df, int num_mat, int other_dim,
	int nrow_exogen_coef, int exogen_lag,
	int dim_factor, bool factor_restrict,
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
	// if (is_row::value) {
	// 	sig_lower = bvhar::sim_iw_tri(post_iw_scl, post_df, rng);
	// } else {
	// 	// Indentifiability restriction
	// 	// Sigma_c(1, 1) = 1
	// 	sig_lower = sim_iw_tri_restr(post_iw_scl, post_df, rng);
	// }
	for (int i = 0; i < prior_mean.rows(); ++i) {
		for (int j = 0; j < prior_mean.cols(); ++j) {
			coef(i, j) = bvhar::normal_rand(rng); // MN(0, I_n, I_k)
		}
	}
	coef = llt_of_prec.matrixU().solve(coef * sig_lower.transpose()) + post_mean;
	// if (std::is_same<xType, Eigen::SparseMatrix<double>>::value) {
	// 	int ncol_coef = prior_mean.cols();
	// 	int nrow_coef = prior_mean.rows() - nrow_exogen_coef - dim_factor;
	// 	int lag = nrow_coef / ncol_coef;
	// 	if (!is_row::value) {
	// 		// int num_col = prior_mean.cols();
	// 		// int nrow_col_coef = prior_mean.rows() - nrow_exogen_coef - dim_factor;
	// 		// int lag = nrow_col_coef / num_col; // when B = (B_1, ..., B_p)^T
	// 		// restrict_mar_col(num_col, lag, coef.topRows(nrow_col_coef), sig_lower, llt_of_prec);
	// 		restrict_mar_col(ncol_coef, lag, coef.topRows(nrow_coef), sig_lower, post_cov.topLeftCorner(nrow_coef, nrow_coef));
	// 		if (nrow_exogen_coef > 0) {
	// 			restrict_mar_col(ncol_coef, exogen_lag + 1, coef.middleRows(nrow_coef, nrow_exogen_coef), sig_lower, post_cov.block(nrow_coef, nrow_coef, nrow_exogen_coef, nrow_exogen_coef));
	// 		}
	// 	} else {
	// 		for (int i = 0; i < lag; ++i) {
	// 			if (coef.middleRows(i * ncol_coef, ncol_coef).trace() <= 0) { // tr(A_i) > 0
	// 				coef.middleRows(i * ncol_coef, ncol_coef) *= -1.0;
	// 				other_coef.middleRows(i * other_dim, other_dim) *= -1.0;
	// 			}
	// 			// if (coef(i * ncol_coef, 0) <= 0) { // A_i(1,1) > 0
	// 			// 	coef.middleRows(i * ncol_coef, ncol_coef) *= -1.0;
	// 			// 	other_coef.middleRows(i * other_dim, other_dim) *= -1.0;
	// 			// }
	// 		}
	// 		// Add trace > 0 for exogen part later
	// 		if (nrow_exogen_coef > 0) {
	// 			int sign_x = coef(nrow_coef, 0) > 0 ? 1 : -1;
	// 			coef.middleRows(nrow_coef, nrow_exogen_coef) *= sign_x;
	// 		}
	// 		// if (coef(0, 0) <= 0) {
	// 		// 	coef = -coef;
	// 		// 	other_coef = -other_coef;
	// 		// }
	// 	}
	// }
	if (dim_factor > 0 && factor_restrict) {
		restrict_mat_loading(prior_mean.cols(), dim_factor, coef.bottomRows(dim_factor), sig_lower);
		// int sign_11 = coef.bottomRows(dim_factor)(0, 0) > 0 ? 1 : -1;
		// coef.bottomRows(dim_factor) /= (sign_11 * coef.bottomRows(dim_factor).squaredNorm());
		// Eigen::MatrixXd lower_sig_post = bvhar::kronecker_eigen(sig_lower.inverse().eval(), llt_of_prec.matrixL().toDenseMatrix());
		// restrict_mat_loading2(prior_mean.cols(), dim_factor, coef.bottomRows(dim_factor), lower_sig_post);
	}
}

template <bool isRow = true, typename xType = Eigen::SparseMatrix<double>>
inline void draw_coef_only(
	Eigen::Ref<Eigen::MatrixXd> coef, Eigen::Ref<const Eigen::MatrixXd> sig_lower,
	Eigen::Ref<const Eigen::MatrixXd> other_coef, Eigen::Ref<const Eigen::MatrixXd> other_sig_lower,
	Eigen::Ref<const Eigen::MatrixXd> prior_mean, Eigen::Ref<const Eigen::VectorXd> prior_prec,
	Eigen::Ref<const Eigen::MatrixXd> iw_scl,
	double iw_df, int num_mat, int other_dim, int dim_factor,
	bool factor_restrict,
	const std::vector<xType>& x, const std::vector<Eigen::MatrixXd>& y,
	BVHAR_BHRNG& rng
) {
	using is_row = std::integral_constant<bool, isRow>;
	Eigen::MatrixXd post_cov = prior_prec.asDiagonal();
	Eigen::MatrixXd post_solve = prior_prec.asDiagonal() * prior_mean;
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
	}
	Eigen::LLT<Eigen::MatrixXd> llt_of_prec;
	double temp_penalty = 0;
	do {
		llt_of_prec.compute((
			post_cov + temp_penalty * Eigen::MatrixXd::Identity(post_cov.rows(), post_cov.cols())
		).selfadjointView<Eigen::Lower>());
		temp_penalty += .01;
	} while (llt_of_prec.info() == Eigen::NumericalIssue && temp_penalty < .1);
	if (llt_of_prec.info() == Eigen::NumericalIssue) {
		eigen_assert("LLT failed in precision sampler.");
	}
	Eigen::MatrixXd post_mean = llt_of_prec.solve(post_solve);
	for (int i = 0; i < prior_mean.rows(); ++i) {
		for (int j = 0; j < prior_mean.cols(); ++j) {
			coef(i, j) = bvhar::normal_rand(rng); // MN(0, I_n, I_k)
		}
	}
	coef = llt_of_prec.matrixU().solve(coef * sig_lower.transpose()) + post_mean;
	// R^T Q_1 = I, C^T Q_2 = I
	if (factor_restrict) {
		restrict_mat_loading(prior_mean.cols(), dim_factor, coef, sig_lower);
	}
}

} // namespace baymar
} // namespace baecon

#endif // BAYMAR_BAYES_MISC_COEF_HELPER_H