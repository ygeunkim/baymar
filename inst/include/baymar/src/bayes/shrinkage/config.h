#ifndef BAYMAR_BAYES_SHRINKAGE_CONFIG_H
#define BAYMAR_BAYES_SHRINKAGE_CONFIG_H

#include "../misc/draw.h"

namespace baecon {
namespace baymar {

struct MatShrinkageParams;
struct MatMinnParams;
struct MatHierMinnParams;
struct MatSsvsParams;
struct MatShrinkageInits;
struct MatHierMinnInits;
struct MatSsvsInits;
struct MatGlInits;

struct MatShrinkageParams {
	MatShrinkageParams() {}
	MatShrinkageParams(BVHAR_LIST& priors) {}
};

struct MatMinnParams : public MatShrinkageParams {
	double _kappa;

	MatMinnParams(BVHAR_LIST& priors, const BVHAR_STRING& prefix = "", const BVHAR_STRING& suffix = "")
	: MatShrinkageParams(priors),
		_kappa(BVHAR_CAST_DOUBLE(priors[prefix + "kappa" + suffix])) {}
};

struct MatSsvsParams : public MatShrinkageParams {
	// Eigen::VectorXd _s1, _s2;
	double _s1, _s2;
	double _slab_shape, _slab_scl;
	int _grid_size;

	MatSsvsParams(BVHAR_LIST& priors, const BVHAR_STRING& prefix = "", const BVHAR_STRING& suffix = "")
	: MatShrinkageParams(priors),
		// _s1(BVHAR_CAST<Eigen::VectorXd>(priors[prefix + "s1" + suffix])),
		// _s2(BVHAR_CAST<Eigen::VectorXd>(priors[prefix + "s2" + suffix])),
		_s1(BVHAR_CAST_DOUBLE(priors[prefix + "s1" + suffix])),
		_s2(BVHAR_CAST_DOUBLE(priors[prefix + "s2" + suffix])),
		_slab_shape(BVHAR_CAST_DOUBLE(priors[prefix + "slab_shape" + suffix])),
		_slab_scl(BVHAR_CAST_DOUBLE(priors[prefix + "slab_scl" + suffix])),
		_grid_size(BVHAR_CAST_INT(priors[prefix + "grid_size" + suffix])) {}
};

struct MatHierMinnParams : public MatShrinkageParams {
	double _shp, _rate;

	MatHierMinnParams(BVHAR_LIST& priors, const BVHAR_STRING& prefix = "", const BVHAR_STRING& suffix = "")
	: MatShrinkageParams(priors),
		_shp(BVHAR_CAST_DOUBLE(priors[prefix + "shape" + suffix])),
		_rate(BVHAR_CAST_DOUBLE(priors[prefix + "rate" + suffix])) {}
};

struct MatShrinkageInits {
	MatShrinkageInits() {}
	MatShrinkageInits(BVHAR_LIST& init) {}
	MatShrinkageInits(BVHAR_LIST& init, int num_design) {}
	MatShrinkageInits(BVHAR_BHRNG& rng) {}
};

struct MatHierMinnInits : public MatShrinkageInits {
	double _kappa;

	MatHierMinnInits(BVHAR_LIST& init, const BVHAR_STRING& prefix = "", const BVHAR_STRING& suffix = "")
	: MatShrinkageInits(init),
		_kappa(BVHAR_CAST_DOUBLE(init[prefix + "kappa" + suffix])) {}
	
	MatHierMinnInits(BVHAR_BHRNG& rng)
	: MatShrinkageInits(rng),
		_kappa(bvhar::unif_rand(.001, 1, rng)) {}
};

struct MatSsvsInits : public MatShrinkageInits {
	Eigen::VectorXd _dummy, _weight, _slab;
	double _spike_scl;
	// Eigen::VectorXd _dummy, _weight;
	// double _slab, _spike_scl;

	MatSsvsInits(BVHAR_LIST& init, const BVHAR_STRING& prefix = "", const BVHAR_STRING& suffix = "")
	: MatShrinkageInits(init),
		_dummy(BVHAR_CAST<Eigen::VectorXd>(init[prefix + "dummy" + suffix])),
		_weight(BVHAR_CAST<Eigen::VectorXd>(init[prefix + "mixture" + suffix])),
		_slab(BVHAR_CAST<Eigen::VectorXd>(init[prefix + "slab" + suffix])),
		// _slab(BVHAR_CAST_DOUBLE(init[prefix + "slab" + suffix])),
		_spike_scl(BVHAR_CAST_DOUBLE(init[prefix + "spike_scl" + suffix])) {}
};

struct MatGlInits : public MatShrinkageInits {
	Eigen::VectorXd _local;
	double _global;
	
	MatGlInits(BVHAR_LIST& init, const BVHAR_STRING& prefix = "", const BVHAR_STRING& suffix = "")
	: MatShrinkageInits(init),
		_local(BVHAR_CAST<Eigen::VectorXd>(init[prefix + "local_sparsity" + suffix])),
		_global(BVHAR_CAST_DOUBLE(init[prefix + "global_sparsity" + suffix])) {}
	
	MatGlInits(int dim)
	: _local(Eigen::VectorXd::Constant(dim, 1.0)), _global(1.0) {}
};

} // namespace baymar
} // namespace baecon

#endif // BAYMAR_BAYES_SHRINKAGE_CONFIG_H