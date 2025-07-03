#ifndef BAYMAR_BAYES_MDFM_CONFIG_H
#define BAYMAR_BAYES_MDFM_CONFIG_H

// #include <bvhar/base>
// #include "../misc/draw.h"
// #include "../../math/design.h"
#include "../mniw/config.h"

namespace baymar {

struct MatDfmParams;
struct MatDfmVarParams;
struct MatDfmVarInits;
struct MatDfmRecords;

struct MatDfmParams : public MatMniwParams {
	int _nrow_factor, _ncol_factor, _size_factor, _lag;

	MatDfmParams(int num_iter, std::vector<Eigen::MatrixXd>& y, LIST& priors)
	: MatMniwParams(num_iter, y, priors),
		_nrow_factor(CAST_INT(priors["nrow_factor"])), _ncol_factor(CAST_INT(priors["ncol_factor"])), _size_factor(_nrow_factor * _ncol_factor),
		_lag(CAST_INT(priors["lag"])) {}
};

struct MatDfmVarParams : public MatDfmParams {
	Eigen::VectorXd _sig_shp, _sig_scl;
	Eigen::VectorXd _mean, _prec;

	MatDfmVarParams(int num_iter, std::vector<Eigen::MatrixXd>& y, LIST& priors)
	: MatDfmParams(num_iter, y, priors),
		_sig_shp(CAST<Eigen::VectorXd>(priors["shape"])),
		_sig_scl(CAST<Eigen::VectorXd>(priors["scale"])),
		_mean(Eigen::VectorXd::Zero(_lag)), _prec(Eigen::VectorXd::Ones(_lag)) {}
};

struct MatDfmVarInits : public MatMniwInits {
	Eigen::MatrixXd _init_factor_coef;
	Eigen::VectorXd _init_factor_prec;

	MatDfmVarInits(LIST& init)
	: MatMniwInits(init),
		_init_factor_coef(CAST<Eigen::MatrixXd>(init["factor_arcoef_init"])),
		_init_factor_prec(CAST<Eigen::VectorXd>(init["factor_arprec_init"])) {}
};

struct MatDfmRecords : public MatMniwRecords {
	Eigen::MatrixXd factor_record;

	MatDfmRecords(
		int num_iter, int num_row, int num_col, int nrow_row_coef, int nrow_col_coef,
		int num_design, int size_factor
	)
	: MatMniwRecords(num_iter, num_row, num_col, nrow_row_coef, nrow_col_coef),
		factor_record(Eigen::MatrixXd::Zero(num_iter + 1, num_design * size_factor)) {}
	
	void assignRecords(
		int id,
		const Eigen::MatrixXd row_coef, const Eigen::MatrixXd row_sig_lower,
		const Eigen::MatrixXd col_coef, const Eigen::MatrixXd col_sig_lower,
		std::vector<Eigen::MatrixXd>& factor_mat,
		int nrow_row_coef, int num_row, int nrow_col_coef, int num_col,
		int num_design, int size_factor
	) {
		MatMniwRecords::assignRecords(
			id,
			row_coef, row_sig_lower, col_coef, col_sig_lower,
			nrow_row_coef, num_row, 0, 0,
			nrow_col_coef, num_col, 0, 0
		);
		for (int i = 0; i < num_design; ++i) {
			// f_{11, p + 1}, f_{21, p + 1}, ..., f_{p1p2, p + 1}, f_{11, p + 2}, ..., f_{p1p2, T}
			factor_record.row(id).segment(i * size_factor, size_factor) = factor_mat[i].reshaped();
		}
	}

	LIST returnListRecords(int nrow_row_coef, int num_row, int nrow_col_coef, int num_col, int num_design, int size_factor) {
		LIST res = MatMniwRecords::returnListRecords(nrow_row_coef, num_row, 0, 0, nrow_col_coef, num_col, 0, 0);
		res["F_record"] = factor_record;
		return res;
	}
};

} // namespace baymar

#endif // BAYMAR_BAYES_MDFM_CONFIG_H