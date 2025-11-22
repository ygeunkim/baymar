#ifndef BAYMAR_MATH_RANDOM_H
#define BAYMAR_MATH_RANDOM_H

#include <bvhar/utils>

namespace baecon {
namespace baymar {

// L of X ~ IW(S, nu)
// with X(1, 1) = 1
inline Eigen::MatrixXd sim_iw_tri_restr(const Eigen::MatrixXd& mat_scale, double shape, BVHAR_BHRNG& rng) {
	int dim = mat_scale.cols();
	// Eigen::MatrixXd permuation = Eigen::MatrixXd::Identity(dim, dim);
	// permuation(0, 0) = 0;
	// permuation(dim - 1, dim - 1) = 0;
	// permuation(0, dim - 1) = 1;
	// permuation(dim - 1, 0) = 1;
	// Eigen::MatrixXd S_trans = mat_scale;
	// S_trans.row(0).swap(S_trans.row(dim - 1));
	// S_trans.col(0).swap(S_trans.col(dim - 1));
  Eigen::MatrixXd mat_bartlett = Eigen::MatrixXd::Zero(dim, dim);
	for (int j = 0; j < dim; ++j) {
		for (int i = 0; i < j; ++i) {
			mat_bartlett(i, j) = bvhar::normal_rand(rng);
		}
		if (j > 0) {
			mat_bartlett(j, j) = sqrt(bvhar::chisq_rand(shape - dim + j + 1, rng));
		}
	}
	Eigen::MatrixXd chol_scale = mat_scale.llt().matrixL();
	mat_bartlett(0, 0) = chol_scale(0, 0); // (L Q^{-T})(1, 1) = 1
	return mat_bartlett.transpose().triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(chol_scale);
}

} // namespace baymar
} // namespace baecon

#endif // BAYMAR_MATH_RANDOM_H