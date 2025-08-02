#ifndef BAYMAR_BAYES_SHRINKAGE_SHRINKAGE_H
#define BAYMAR_BAYES_SHRINKAGE_SHRINKAGE_H

#include "./config.h"

namespace baymar {

class MatShrinkageUpdater;
class MatMinnUpdater;
class MatHsUpdater;

class MatShrinkageUpdater {
public:
	MatShrinkageUpdater(int num_iter, const MatShrinkageParams& params, const MatShrinkageInits& inits) {}
	virtual ~MatShrinkageUpdater() = default;
	virtual void initPrec(Eigen::Ref<Eigen::VectorXd> prior_prec) {}
	virtual void updatePrec(
		Eigen::Ref<Eigen::VectorXd> prior_prec,
		Eigen::Ref<Eigen::MatrixXd> coef, Eigen::Ref<Eigen::MatrixXd> sig_lower,
		Eigen::Ref<Eigen::MatrixXd> prior_mean,
		BVHAR_BHRNG& rng
	) {}
};

class MatMinnUpdater : public MatShrinkageUpdater {
public:
	MatMinnUpdater(int num_iter, const MatMinnParams& params, const MatShrinkageInits& inits)
	: MatShrinkageUpdater(num_iter, params, inits), kappa(params._kappa) {}
	virtual ~MatMinnUpdater() = default;
	void initPrec(Eigen::Ref<Eigen::VectorXd> prior_prec) override {
		prior_prec.array() /= kappa;
	}

private:
	double kappa;
};

class MatHierMinnUpdater : public MatShrinkageUpdater {
	public:
		MatHierMinnUpdater(int num_iter, const MatHierMinnParams& params, const MatHierMinnInits& inits)
		: MatShrinkageUpdater(num_iter, params, inits),
			shp(params._shp), rate(params._rate), kappa(inits._kappa) {}
		virtual ~MatHierMinnUpdater() = default;
		void initPrec(Eigen::Ref<Eigen::VectorXd> prior_prec) override {
			prior_prec.array() /= kappa;
		}
		void updatePrec(
			Eigen::Ref<Eigen::VectorXd> prior_prec,
			Eigen::Ref<Eigen::MatrixXd> coef, Eigen::Ref<Eigen::MatrixXd> sig_lower,
			Eigen::Ref<Eigen::MatrixXd> prior_mean,
			BVHAR_BHRNG& rng
		) override {
			minnesota_kappa(kappa, prior_mean, prior_prec, coef, sig_lower, shp, rate, rng);
		}
	
	private:
		double shp, rate, kappa;
	};

class MatHsUpdater : public MatShrinkageUpdater {
public:
	MatHsUpdater(int num_iter, const MatShrinkageParams& params, const MatGlInits& inits)
	: MatShrinkageUpdater(num_iter, params, inits),
		local_lev(inits._local), global_lev(inits._global),
		latent_local(Eigen::VectorXd::Zero(local_lev.size())),
		latent_global(0.0) {}
	virtual ~MatHsUpdater() = default;
	
	void initPrec(Eigen::Ref<Eigen::VectorXd> prior_prec) override {
		prior_prec.array() /= (global_lev * local_lev.array()).square();
	}

	void updatePrec(
		Eigen::Ref<Eigen::VectorXd> prior_prec,
		Eigen::Ref<Eigen::MatrixXd> coef, Eigen::Ref<Eigen::MatrixXd> sig_lower,
		Eigen::Ref<Eigen::MatrixXd> prior_mean,
		BVHAR_BHRNG& rng
	) override {
		bvhar::horseshoe_latent(latent_local, local_lev, rng);
		bvhar::horseshoe_latent(latent_global, global_lev, rng);
		horseshoe_sparsity(local_lev, global_lev, prior_prec, coef, sig_lower, latent_local, latent_global, rng);
		prior_prec = 1 / (global_lev * local_lev.array()).square();
	}

private:
	Eigen::VectorXd local_lev;
	double global_lev;
	Eigen::VectorXd latent_local;
	double latent_global;
};

inline std::unique_ptr<MatShrinkageUpdater> initialize_matshrinkageupdater(int num_iter, BVHAR_LIST& param_prior, BVHAR_LIST& param_init, int prior_type) {
	std::unique_ptr<MatShrinkageUpdater> shrinkage_ptr;
	switch (prior_type) {
		case 1: {
			MatMinnParams params(param_prior);
			MatShrinkageInits inits(param_init);
			shrinkage_ptr = std::make_unique<MatMinnUpdater>(num_iter, params, inits);
			return shrinkage_ptr;
		}
		case 3: {
			MatShrinkageParams params(param_prior);
			MatGlInits inits(param_init);
			shrinkage_ptr = std::make_unique<MatHsUpdater>(num_iter, params, inits);
			return shrinkage_ptr;
		}
		case 4: {
			MatHierMinnParams params(param_prior);
			MatHierMinnInits inits(param_init);
			shrinkage_ptr = std::make_unique<MatHierMinnUpdater>(num_iter, params, inits);
			return shrinkage_ptr;
		}
		default: {
			BVHAR_STOP("Not defined yet");
		}
	}
	return shrinkage_ptr;
}

} // namespace baymar

#endif // BAYMAR_BAYES_SHRINKAGE_SHRINKAGE_H