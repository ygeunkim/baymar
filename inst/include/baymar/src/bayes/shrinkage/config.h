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
	MatShrinkageParams(LIST& priors) {}
};

struct MatMinnParams : public MatShrinkageParams {
	double _kappa;

	MatMinnParams(LIST& priors)
	: MatShrinkageParams(priors),
		_kappa(CAST_DOUBLE(priors["kappa"])) {}
};

struct MatHierMinnParams : public MatShrinkageParams {
	double _shp, _rate;

	MatHierMinnParams(LIST& priors)
	: MatShrinkageParams(priors),
		_shp(CAST_DOUBLE(priors["shape"])),
		_rate(CAST_DOUBLE(priors["rate"])) {}
};

struct MatShrinkageInits {
	MatShrinkageInits() {}
	MatShrinkageInits(LIST& init) {}
	MatShrinkageInits(LIST& init, int num_design) {}
	MatShrinkageInits(BHRNG& rng) {}
};

struct MatHierMinnInits : public MatShrinkageInits {
	double _kappa;

	MatHierMinnInits(LIST& init)
	: MatShrinkageInits(init),
		_kappa(CAST_DOUBLE(init["kappa"])) {}
	
	MatHierMinnInits(BHRNG& rng)
	: MatShrinkageInits(rng),
		_kappa(bvhar::unif_rand(.001, 1, rng)) {}
};

struct MatGlInits : public MatShrinkageInits {
	Eigen::VectorXd _local;
	double _global;
	
	MatGlInits(LIST& init)
	: MatShrinkageInits(init),
		_local(CAST<Eigen::VectorXd>(init["local_sparsity"])),
		_global(CAST_DOUBLE(init["global_sparsity"])) {}
};

} // namespace baymar

#endif // BAYMAR_BAYES_SHRINKAGE_CONFIG_H