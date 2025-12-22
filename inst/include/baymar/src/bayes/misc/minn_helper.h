#ifndef BAYMAR_BAYES_MISC_MINN_HELPER_H
#define BAYMAR_BAYES_MISC_MINN_HELPER_H

#include <bvhar/utils>
#include <type_traits>

namespace baecon {
namespace baymar {

inline void minnesota_kappa(
	double& kappa, Eigen::Ref<const Eigen::MatrixXd> prior_mean, Eigen::Ref<Eigen::VectorXd> prec,
	Eigen::Ref<const Eigen::MatrixXd> coef, Eigen::Ref<const Eigen::MatrixXd> sig_lower,
	const double gamma_shp, const double gamma_rate, BVHAR_BHRNG& rng
) {
	prec.array() *= kappa;
	Eigen::MatrixXd inv_sig_coef = sig_lower.triangularView<Eigen::Lower>().solve((coef - prior_mean).transpose());
	kappa = bvhar::sim_gig(
		gamma_shp - prior_mean.rows() * prior_mean.cols() / 2,
		2 * gamma_rate,
		((inv_sig_coef.transpose() * inv_sig_coef).diagonal().array() * prec.array()).sum(),
		rng
	);
	prec /= kappa;
}

} // namespace baymar
} // namespace baecon

#endif // BAYMAR_BAYES_MISC_MINN_HELPER_H