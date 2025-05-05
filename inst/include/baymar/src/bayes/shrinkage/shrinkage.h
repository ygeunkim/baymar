#ifndef BAYMAR_BAYES_SHRINKAGE_SHRINKAGE_H
#define BAYMAR_BAYES_SHRINKAGE_SHRINKAGE_H

#include "./config.h"

namespace baymar {

class MatShrinkageUpdater;
class MatMinnUpdater;

class MatShrinkageUpdater {
public:
	MatShrinkageUpdater(int num_iter, const MatShrinkageParams& params, const MatShrinkageInits& inits) {}
	virtual ~MatShrinkageUpdater() = default;
	virtual void updatePrec(
		Eigen::Ref<Eigen::VectorXd> prior_prec,
		Eigen::Ref<Eigen::MatrixXd> coef, Eigen::Ref<Eigen::MatrixXd> sig_lower,
		Eigen::Ref<Eigen::MatrixXd> prior_mean,
		BHRNG& rng
	) {}
};

class MatMinnUpdater : public MatShrinkageUpdater {
public:
	MatMinnUpdater(int num_iter, const MatMinnParams& params, const MatMinnInits& inits)
	: MatShrinkageUpdater(num_iter, params, inits),
		shp(params._shp), rate(params._rate), kappa(inits._kappa) {}
	virtual ~MatMinnUpdater() = default;
	void updatePrec(
		Eigen::Ref<Eigen::VectorXd> prior_prec,
		Eigen::Ref<Eigen::MatrixXd> coef, Eigen::Ref<Eigen::MatrixXd> sig_lower,
		Eigen::Ref<Eigen::MatrixXd> prior_mean,
		BHRNG& rng
	) override {
		minnesota_kappa(kappa, prior_mean, prior_prec, coef, sig_lower, shp, rate, rng);
	}

private:
	double shp, rate, kappa;
};

inline std::unique_ptr<MatShrinkageUpdater> initialize_matshrinkageupdater(int num_iter, LIST& param_prior, LIST& param_init, int prior_type) {
	std::unique_ptr<MatShrinkageUpdater> shrinkage_ptr;
	switch (prior_type) {
		case 1: {
			MatMinnParams params(param_prior);
			MatMinnInits inits(param_init);
			shrinkage_ptr = std::make_unique<MatMinnUpdater>(num_iter, params, inits);
			return shrinkage_ptr;
		}
	}
	return shrinkage_ptr;
}

} // namespace baymar

#endif // BAYMAR_BAYES_SHRINKAGE_SHRINKAGE_H