#ifndef BAYMAR_BAYES_MISC_SSVS_HELPER_H
#define BAYMAR_BAYES_MISC_SSVS_HELPER_H

#include <bvhar/utils>

namespace baecon {
namespace baymar {

// inline void ssvs_global_slab(
// 	double& slab_param, Eigen::Ref<Eigen::VectorXd> dummy_param,
// 	Eigen::Ref<Eigen::MatrixXd> prior_mean,
// 	Eigen::Ref<Eigen::MatrixXd> coef, Eigen::Ref<Eigen::MatrixXd> sig_lower,
// 	double& shp, double& scl, double& spike_scl,
// 	BVHAR_BHRNG& rng
// ) {
// 	int col_coef = coef.cols();
// 	Eigen::MatrixXd inv_sig_coef = sig_lower.triangularView<Eigen::Lower>().solve((coef - prior_mean).transpose());
// 	Eigen::VectorXd prod = (inv_sig_coef.transpose() * inv_sig_coef).diagonal();
// 	slab_param = 1 / bvhar::gamma_rand(
// 		shp + col_coef / 2,
// 		1 / (scl + (prod.array() / (dummy_param.array() + (1 - dummy_param.array()) * spike_scl).array()).sum() / 2),
// 		rng
// 	);
// }

// inline void ssvs_local_slab(
// 	Eigen::Ref<Eigen::VectorXd> local_slab, Eigen::Ref<Eigen::VectorXd> dummy_param,
// 	Eigen::Ref<Eigen::MatrixXd> prior_mean,
// 	Eigen::Ref<Eigen::MatrixXd> coef, Eigen::Ref<Eigen::MatrixXd> sig_lower,
// 	double& shp, double& scl, double& spike_scl,
// 	BVHAR_BHRNG& rng
// ) {
// 	int col_coef = coef.cols();
// 	Eigen::MatrixXd inv_sig_coef = sig_lower.triangularView<Eigen::Lower>().solve((coef - prior_mean).transpose());
// 	Eigen::VectorXd prod = (inv_sig_coef.transpose() * inv_sig_coef).diagonal();
// 	for (int i = 0; i < col_coef; ++i) {
// 		local_slab[i] = 1 / bvhar::gamma_rand(
// 			shp + .5,
// 			1 / (scl + prod[i] / ((dummy_param[i] + (1 - dummy_param[i]) * spike_scl) * 2)),
// 			rng
// 		);
// 	}
// }

inline void ssvs_sparsity(
	Eigen::Ref<Eigen::VectorXd> local_slab, Eigen::Ref<Eigen::VectorXd> dummy_param,
	Eigen::Ref<Eigen::VectorXd> slab_weight,
	Eigen::Ref<Eigen::MatrixXd> prior_mean,
	Eigen::Ref<Eigen::MatrixXd> coef, Eigen::Ref<Eigen::MatrixXd> sig_lower,
	const double& shp, const double& scl,
	const double& prior_s1, const double& prior_s2,
	double& spike_scl, const int grid_size,
	BVHAR_BHRNG& rng
) {
	int col_coef = coef.cols();
	Eigen::MatrixXd inv_sig_coef = sig_lower.triangularView<Eigen::Lower>().solve((coef - prior_mean).transpose());
	Eigen::VectorXd prod = (inv_sig_coef.transpose() * inv_sig_coef).diagonal();
	for (int i = 0; i < col_coef; ++i) {
		local_slab[i] = 1 / bvhar::gamma_rand(
			shp + .5,
			1 / (scl + prod[i] / ((dummy_param[i] + (1 - dummy_param[i]) * spike_scl) * 2)),
			rng
		);
	}
	// Griddy gibbs for spike_scl
	Eigen::VectorXd grid = Eigen::VectorXd::LinSpaced(grid_size + 2, 0.0, 1.0).segment(1, grid_size);
	Eigen::VectorXd log_wt(grid_size);
	for (int i = 0; i < grid_size; ++i) {
		log_wt[i] = -(prod.array() / local_slab.array()).sum() / (2 * spike_scl) - col_coef * log(grid[i]);
	}
	Eigen::VectorXd weight = (log_wt.array() - log_wt.maxCoeff()).exp();
	weight /= weight.sum();
	spike_scl = grid[bvhar::cat_rand(weight, rng)];
	Eigen::VectorXd exp_u1 = prod.array() / (2 * local_slab.array());
	Eigen::VectorXd exp_u2 = exp_u1 / spike_scl;
	Eigen::VectorXd max_exp = exp_u1.cwiseMax(exp_u2);
	exp_u1 = slab_weight.array() * (exp_u1 - max_exp).array().exp() / local_slab.array();
	exp_u2 = (1 - slab_weight.array()) * (exp_u2 - max_exp).array().exp() / (spike_scl * local_slab.array());
	for (int i = 0; i < col_coef; ++i) {
		dummy_param[i] = bvhar::ber_rand(exp_u1[i] / (exp_u1[i] + exp_u2[i]), rng);
	}
	double post_s1 = prior_s1 + dummy_param.sum(); // s1 + number of ones
  double post_s2 = prior_s2 + col_coef - dummy_param.sum(); // s2 + number of zeros
  for (int i = 0; i < col_coef; i++) {
		slab_weight[i] = bvhar::beta_rand(post_s1, post_s2, rng);
  }
}

} // namespace baymar
} // namespace baecon

#endif // BAYMAR_BAYES_MISC_SSVS_HELPER_H