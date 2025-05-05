#ifndef BAYMAR_BAYES_SHRINKAGE_CONFIG_H
#define BAYMAR_BAYES_SHRINKAGE_CONFIG_H

#include "../misc/draw.h"

namespace baymar {

struct MatShrinkageParams;
struct MatMinnParams;
struct MatShrinkageInits;
struct MatMinnInits;

struct MatShrinkageParams {
	MatShrinkageParams() {}
	MatShrinkageParams(LIST& priors) {}
};

struct MatMinnParams : public MatShrinkageParams {
	double _shp, _rate;

	MatMinnParams(LIST& priors)
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

struct MatMinnInits : public MatShrinkageInits {
	double _kappa;

	MatMinnInits(LIST& init)
	: MatShrinkageInits(init),
		_kappa(CAST_DOUBLE(init["kappa"])) {}
	
	MatMinnInits(BHRNG& rng)
	: MatShrinkageInits(rng),
		_kappa(bvhar::unif_rand(.001, 1, rng)) {}
};

} // namespace baymar

#endif // BAYMAR_BAYES_SHRINKAGE_CONFIG_H