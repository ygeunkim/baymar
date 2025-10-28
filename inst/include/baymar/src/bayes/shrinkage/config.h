#ifndef BAYMAR_BAYES_SHRINKAGE_CONFIG_H
#define BAYMAR_BAYES_SHRINKAGE_CONFIG_H

#include "../misc/draw.h"

namespace baymar {

struct MatShrinkageParams;
struct MatMinnParams;
struct MatHierMinnParams;
struct MatShrinkageInits;
struct MatHierMinnInits;
struct MatGlInits;

struct MatShrinkageParams {
	MatShrinkageParams() {}
	MatShrinkageParams(BVHAR_LIST& priors) {}
};

struct MatMinnParams : public MatShrinkageParams {
	double _kappa;

	MatMinnParams(BVHAR_LIST& priors, const BVHAR_STRING& prefix = "")
	: MatShrinkageParams(priors),
		_kappa(BVHAR_CAST_DOUBLE(priors[prefix + "kappa"])) {}
};

struct MatHierMinnParams : public MatShrinkageParams {
	double _shp, _rate;

	MatHierMinnParams(BVHAR_LIST& priors, const BVHAR_STRING& prefix = "")
	: MatShrinkageParams(priors),
		_shp(BVHAR_CAST_DOUBLE(priors[prefix + "shape"])),
		_rate(BVHAR_CAST_DOUBLE(priors[prefix + "rate"])) {}
};

struct MatShrinkageInits {
	MatShrinkageInits() {}
	MatShrinkageInits(BVHAR_LIST& init) {}
	MatShrinkageInits(BVHAR_LIST& init, int num_design) {}
	MatShrinkageInits(BVHAR_BHRNG& rng) {}
};

struct MatHierMinnInits : public MatShrinkageInits {
	double _kappa;

	MatHierMinnInits(BVHAR_LIST& init, const BVHAR_STRING& prefix = "")
	: MatShrinkageInits(init),
		_kappa(BVHAR_CAST_DOUBLE(init[prefix + "kappa"])) {}
	
	MatHierMinnInits(BVHAR_BHRNG& rng)
	: MatShrinkageInits(rng),
		_kappa(bvhar::unif_rand(.001, 1, rng)) {}
};

struct MatGlInits : public MatShrinkageInits {
	Eigen::VectorXd _local;
	double _global;
	
	MatGlInits(BVHAR_LIST& init, const BVHAR_STRING& prefix = "")
	: MatShrinkageInits(init),
		_local(BVHAR_CAST<Eigen::VectorXd>(init[prefix + "local_sparsity"])),
		_global(BVHAR_CAST_DOUBLE(init[prefix + "global_sparsity"])) {}
	
	MatGlInits(int dim)
	: _local(Eigen::VectorXd::Constant(dim, 1.0)), _global(1.0) {}
};

} // namespace baymar

#endif // BAYMAR_BAYES_SHRINKAGE_CONFIG_H