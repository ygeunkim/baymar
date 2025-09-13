#ifndef BAYMAR_BAYES_MISC_HS_HELPER_H
#define BAYMAR_BAYES_MISC_HS_HELPER_H

#include <bvhar/utils>

namespace baymar {

// inline void horseshoe_latent(Eigen::VectorXd& latent, Eigen::VectorXd& hyperparam, BVHAR_BHRNG& rng) {
//   int dim = hyperparam.size();
//   for (int i = 0; i < dim; i++) {
// 		latent[i] = 1 / gamma_rand(1.0, 1 / (1 + 1 / (hyperparam[i] * hyperparam[i])), rng);
//   }
// }
// // overloading
// inline void horseshoe_latent(double& latent, double& hyperparam, BVHAR_BHRNG& rng) {
//   latent = 1 / gamma_rand(1.0, 1 / (1 + 1 / (hyperparam * hyperparam)), rng);
// }

inline void horseshoe_sparsity(
	Eigen::Ref<Eigen::VectorXd> local_sparsity, double& global_sparsity,
	Eigen::Ref<Eigen::MatrixXd> coef, Eigen::Ref<Eigen::MatrixXd> sig_lower,
	Eigen::Ref<Eigen::VectorXd> local_latent, double& global_latent,
	BVHAR_BHRNG& rng
) {
	global_latent = bvhar::gamma_rand(1.0, 1 / (1 + global_sparsity), rng);
	int col_coef = coef.cols();
	Eigen::MatrixXd inv_sig_coef = sig_lower.triangularView<Eigen::Lower>().solve(coef.transpose());
	Eigen::VectorXd prod = (inv_sig_coef.transpose() * inv_sig_coef).diagonal();
	global_sparsity = bvhar::gamma_rand(
		(col_coef + 1) / 2,
		1 / (global_latent + (prod.array() * local_sparsity.array()).sum() / 2),
		rng
	);
	for (int i = 0; i < local_sparsity.size(); ++i) {
		local_latent[i] = bvhar::gamma_rand(1.0, 1 / (1 + local_sparsity[i]), rng);
		local_sparsity[i] = bvhar::gamma_rand(
			1.0,
			1 / (local_latent[i] + 2 * prod[i] * global_sparsity),
			rng
		);
	}
	// group
}

} // namespace baymar

#endif // BAYMAR_BAYES_MISC_HS_HELPER_H