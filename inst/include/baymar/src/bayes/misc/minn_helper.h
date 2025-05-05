#ifndef BAYMAR_BAYES_MISC_MINN_HELPER_H
#define BAYMAR_BAYES_MISC_MINN_HELPER_H

#include <bvhar/utils>
#include <type_traits>

namespace baymar {

inline void minnesota_kappa(
	double& kappa, Eigen::Ref<Eigen::MatrixXd> prior_mean, Eigen::Ref<Eigen::VectorXd> prec, std::vector<Eigen::MatrixXd>& params,
	const double gamma_shp, const double gamma_rate, BHRNG& rng
) {
	prec.array() *= kappa;
	Eigen::MatrixXd inv_sig_coef = params[1].triangularView<Eigen::Lower>().solve((params[0] - prior_mean).transpose());
	kappa = bvhar::sim_gig(
		gamma_shp - prior_mean.rows() * prior_mean.cols() / 2,
		2 * gamma_rate,
		((inv_sig_coef.transpose() * inv_sig_coef).diagonal().array() * prec.array()).sum(),
		rng
	);
	prec /= kappa;
}

} // namespace baymar

#endif // BAYMAR_BAYES_MISC_MINN_HELPER_H