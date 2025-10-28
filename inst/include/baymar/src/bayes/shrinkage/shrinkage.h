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
	virtual void updateRecords(int id) {}
	virtual void appendRowRecords(BVHAR_LIST& list) {}
	virtual void appendColRecords(BVHAR_LIST& list) {}
	virtual void appendExogenRowRecords(BVHAR_LIST& list) {}
	virtual void appendExogenColRecords(BVHAR_LIST& list) {}
	virtual void appendFactorRowRecords(BVHAR_LIST& list) {}
	virtual void appendFactorColRecords(BVHAR_LIST& list) {}
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
			shp(params._shp), rate(params._rate), kappa(inits._kappa),
			kappa_record(Eigen::VectorXd::Zero(num_iter + 1)) {}
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

		void updateRecords(int id) override {
			kappa_record[id] = kappa;
		}

		void appendRowRecords(BVHAR_LIST& list) override {
			list["kappaR_record"] = kappa_record;
		}

		void appendColRecords(BVHAR_LIST& list) override {
			list["kappaC_record"] = kappa_record;
		}

		void appendExogenRowRecords(BVHAR_LIST& list) override {
			list["kappaXr_record"] = kappa_record;
		}

		void appendExogenColRecords(BVHAR_LIST& list) override {
			list["kappaXc_record"] = kappa_record;
		}

		void appendFactorRowRecords(BVHAR_LIST& list) override {
			list["kappaFr_record"] = kappa_record;
		}

		void appendFactorColRecords(BVHAR_LIST& list) override {
			list["kappaFc_record"] = kappa_record;
		}
	
	private:
		double shp, rate, kappa;
		Eigen::VectorXd kappa_record;
	};

class MatHsUpdater : public MatShrinkageUpdater {
public:
	MatHsUpdater(int num_iter, const MatShrinkageParams& params, const MatGlInits& inits)
	: MatShrinkageUpdater(num_iter, params, inits),
		local_lev(inits._local), global_lev(inits._global),
		latent_local(Eigen::VectorXd::Zero(local_lev.size())),
		latent_global(0.0),
		global_record(Eigen::VectorXd::Zero(num_iter + 1)),
		local_record(Eigen::MatrixXd::Zero(num_iter + 1, local_lev.size())) {}
	virtual ~MatHsUpdater() = default;
	
	void initPrec(Eigen::Ref<Eigen::VectorXd> prior_prec) override {
		prior_prec = global_lev * local_lev;
	}

	void updatePrec(
		Eigen::Ref<Eigen::VectorXd> prior_prec,
		Eigen::Ref<Eigen::MatrixXd> coef, Eigen::Ref<Eigen::MatrixXd> sig_lower,
		Eigen::Ref<Eigen::MatrixXd> prior_mean,
		BVHAR_BHRNG& rng
	) override {
		// bvhar::horseshoe_latent(latent_local, local_lev, rng);
		// bvhar::horseshoe_latent(latent_global, global_lev, rng);
		horseshoe_sparsity(local_lev, global_lev, prior_mean, coef, sig_lower, latent_local, latent_global, rng);
		prior_prec = global_lev * local_lev;
	}

	void updateRecords(int id) override {
		local_record.row(id) = local_lev;
		global_record[id] = global_lev;
	}

	void appendRowRecords(BVHAR_LIST& list) override {
		list["lambdaR_record"] = local_record;
		list["tauR_record"] = global_record;
	}

	void appendColRecords(BVHAR_LIST& list) override {
		list["lambdaC_record"] = local_record;
		list["tauC_record"] = global_record;
	}

	void appendExogenRowRecords(BVHAR_LIST& list) override {
		list["lambdaXr_record"] = local_record;
		list["tauXr_record"] = global_record;
	}

	void appendExogenColRecords(BVHAR_LIST& list) override {
		list["lambdaXc_record"] = local_record;
		list["tauXc_record"] = global_record;
	}

	void appendFactorRowRecords(BVHAR_LIST& list) override {
		list["lambdaFr_record"] = local_record;
		list["tauFr_record"] = global_record;
	}
	
	void appendFactorColRecords(BVHAR_LIST& list) override {
		list["lambdaFc_record"] = local_record;
		list["tauFc_record"] = global_record;
	}

private:
	Eigen::VectorXd local_lev;
	double global_lev;
	Eigen::VectorXd latent_local;
	double latent_global;
	Eigen::VectorXd global_record;
	Eigen::MatrixXd local_record;
};

inline std::unique_ptr<MatShrinkageUpdater> initialize_matshrinkageupdater(int num_iter, BVHAR_LIST& param_prior, BVHAR_LIST& param_init, int prior_type, const BVHAR_STRING& prefix = "") {
	std::unique_ptr<MatShrinkageUpdater> shrinkage_ptr;
	if (prior_type == 0) {
		// Should check when using pybind11: BVHAR_STRING is py::str -> change this to std::string?
		if (BVHAR_CONTAINS(param_init, (prefix + "local_sparsity").c_str())) {
			prior_type = 3;
		} else if (BVHAR_CONTAINS(param_init, (prefix + "kappa").c_str())) {
			prior_type = 4;
		}
	}
	switch (prior_type) {
		case 1: {
			MatMinnParams params(param_prior);
			MatShrinkageInits inits(param_init);
			shrinkage_ptr = std::make_unique<MatMinnUpdater>(num_iter, params, inits);
			return shrinkage_ptr;
		}
		case 3: {
			MatShrinkageParams params(param_prior);
			MatGlInits inits(param_init, prefix);
			shrinkage_ptr = std::make_unique<MatHsUpdater>(num_iter, params, inits);
			return shrinkage_ptr;
		}
		case 4: {
			MatHierMinnParams params(param_prior, prefix);
			MatHierMinnInits inits(param_init, prefix);
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