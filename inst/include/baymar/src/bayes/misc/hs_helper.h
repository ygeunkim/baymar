#ifndef BAYMAR_BAYES_MISC_HS_HELPER_H
#define BAYMAR_BAYES_MISC_HS_HELPER_H

#include <bvhar/utils>

namespace baymar {

inline void horseshoe_sparsity(
	Eigen::Ref<Eigen::VectorXd> local_sparsity, double& global_sparsity, Eigen::Ref<Eigen::VectorXd> prec,
	Eigen::Ref<Eigen::MatrixXd> coef, Eigen::Ref<Eigen::MatrixXd> sig_lower,
	Eigen::Ref<Eigen::VectorXd> local_latent, double& global_latent,
	BHRNG& rng
) {
	int col_coef = coef.cols();
	Eigen::MatrixXd inv_sig_coef = sig_lower.triangularView<Eigen::Lower>().solve(coef.transpose());
	Eigen::VectorXd prod = (inv_sig_coef.transpose() * inv_sig_coef).diagonal();
	prec.array() *= global_sparsity;
	global_sparsity = 1 / bvhar::gamma_rand(
		(col_coef + 1) / 2,
		1 / (1 / global_latent + (prod.diagonal().array() * prec.array()).sum()),
		rng
	);
	prec.array() /= global_sparsity;
	prec.array() *= local_sparsity.array();
	for (int i = 0; i < local_sparsity.size(); ++i) {
		local_sparsity[i] = 1 / bvhar::gamma_rand(
			(col_coef + 1) / 2,
			1 / (1 / local_latent[i] + prod[i] / (2 * global_sparsity)),
			rng
		);
	}
	prec.array() /= local_sparsity.array();
	// group
}

} // namespace baymar

#endif // BAYMAR_BAYES_MISC_HS_HELPER_H