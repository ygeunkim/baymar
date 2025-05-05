#ifndef BAYMAR_BAYES_MNIW_CONFIG_H
#define BAYMAR_BAYES_MNIW_CONFIG_H

#include <bvhar/triangular> // add another one for base in bvhar later
#include "../shrinkage/shrinkage.h"

namespace baymar {

struct MatMniwParams;
struct MatMniwInits;

struct MatMniwParams {
	std::vector<Eigen::SparseMatrix<double>> _x;
	std::vector<Eigen::MatrixXd> _y;
	int _iter, _row, _col, _design;
	Eigen::MatrixXd _row_mean, _row_iw_scl;
	Eigen::MatrixXd _col_mean, _col_iw_scl;
	Eigen::VectorXd _row_prec, _col_prec;
	double _row_iw_df, _col_iw_df;

	MatMniwParams(int num_iter, std::vector<Eigen::SparseMatrix<double>>& x, std::vector<Eigen::MatrixXd>& y, LIST& priors)
	: _x(x), _y(y),
		_iter(num_iter), _row(y[0].rows()), _col(y[0].cols()), _design(y.size()),
		_row_mean(CAST<Eigen::MatrixXd>(priors["row_prior_mean"])), _row_iw_scl(CAST<Eigen::MatrixXd>(priors["row_iw_scl"])),
		_col_mean(CAST<Eigen::MatrixXd>(priors["col_prior_mean"])), _col_iw_scl(CAST<Eigen::MatrixXd>(priors["col_iw_scl"])),
		_row_prec(CAST<Eigen::MatrixXd>(priors["row_prior_prec"]).diagonal()), _col_prec(CAST<Eigen::MatrixXd>(priors["col_prior_prec"]).diagonal()),
		_row_iw_df(CAST_DOUBLE(priors["row_iw_df"])), _col_iw_df(CAST_DOUBLE(priors["col_iw_df"])) {}
};

struct MatMniwInits {
	Eigen::MatrixXd _init_row_coef, _init_row_lower, _init_col_coef, _init_col_lower;

	MatMniwInits(LIST& init)
	: _init_row_coef(CAST<Eigen::MatrixXd>(init["row_init_coef"])),
		_init_row_lower(CAST<Eigen::MatrixXd>(init["row_init_lower"])),
		_init_col_coef(CAST<Eigen::MatrixXd>(init["col_init_coef"])),
		_init_col_lower(CAST<Eigen::MatrixXd>(init["col_init_lower"])) {}
};

} // namespace baymar

#endif // BAYMAR_BAYES_MNIW_CONFIG_H