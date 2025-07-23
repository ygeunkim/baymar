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
struct MatDfmVarRecords;

struct MatDfmParams {
	int _nrow_factor, _ncol_factor, _size_factor, _lag;

	MatDfmParams(BVHAR_LIST& priors)
	: _nrow_factor(BVHAR_CAST_INT(priors["nrow_factor"])), _ncol_factor(BVHAR_CAST_INT(priors["ncol_factor"])), _size_factor(_nrow_factor * _ncol_factor),
		_lag(BVHAR_CAST_INT(priors["lag"])) {}
};

struct MatDfmVarParams : public MatDfmParams {
	Eigen::VectorXd _sig_shp, _sig_scl;
	Eigen::VectorXd _mean, _prec;

	MatDfmVarParams(BVHAR_LIST& priors)
	: MatDfmParams(priors),
		_sig_shp(BVHAR_CAST<Eigen::VectorXd>(priors["shape"])),
		_sig_scl(BVHAR_CAST<Eigen::VectorXd>(priors["scale"])),
		_mean(Eigen::VectorXd::Zero(_lag)), _prec(Eigen::VectorXd::Ones(_lag)) {}
};

struct MatDfmVarInits : public MatMniwInits {
	Eigen::MatrixXd _init_factor_coef;
	Eigen::VectorXd _init_factor_prec;

	MatDfmVarInits(BVHAR_LIST& init)
	: MatMniwInits(init),
		_init_factor_coef(BVHAR_CAST<Eigen::MatrixXd>(init["factor_arcoef_init"])),
		_init_factor_prec(BVHAR_CAST<Eigen::VectorXd>(init["factor_arprec_init"])) {}
};

struct MatDfmRecords {
	Eigen::MatrixXd factor_record;

	MatDfmRecords(int num_iter, int num_design, int size_factor)
	: factor_record(Eigen::MatrixXd::Zero(num_iter + 1, num_design * size_factor)) {}
	
	MatDfmRecords(const Eigen::MatrixXd& factor_record)
	: factor_record(factor_record) {}
	
	virtual ~MatDfmRecords() = default;
	
	void assignRecords(
		int id,
		std::vector<Eigen::MatrixXd>& factor_mat,
		int num_design, int size_factor
	) {
		for (int i = 0; i < num_design; ++i) {
			// f_{11, p + 1}, f_{21, p + 1}, ..., f_{p1p2, p + 1}, f_{11, p + 2}, ..., f_{p1p2, T}
			factor_record.row(id).segment(i * size_factor, size_factor) = factor_mat[i].reshaped();
		}
	}

	virtual void assignRecords(
		int id,
		std::vector<Eigen::MatrixXd>& factor_mat,
		const Eigen::MatrixXd& factor_coef, const Eigen::VectorXd& factor_prec,
		int num_design, int size_factor
	) = 0;

	// BVHAR_LIST returnListRecords(int nrow_row_coef, int num_row, int nrow_col_coef, int num_col, int num_design, int size_factor) {
	// 	BVHAR_LIST res = MatMniwRecords::returnListRecords(nrow_row_coef, num_row, 0, 0, nrow_col_coef, num_col, 0, 0);
	// 	// res["F_record"] = factor_record;
	// 	return res;
	// }

	virtual void appendRecords(BVHAR_LIST& list) {
		list["F_record"] = factor_record;
	}

	// MatDfmRecords returnDfmRecords(int num_iter, int num_burn, int thin) const {
	// 	return MatDfmRecords(
	// 		bvhar::thin_record(row_coef_record, num_iter, num_burn, thin).derived(),
	// 		bvhar::thin_record(row_sigma_record, num_iter, num_burn, thin).derived(),
	// 		bvhar::thin_record(col_coef_record, num_iter, num_burn, thin).derived(),
	// 		bvhar::thin_record(col_sigma_record, num_iter, num_burn, thin).derived(),
	// 		bvhar::thin_record(factor_record, num_iter, num_burn, thin).derived()
	// 	);
	// }

	virtual MatDfmVarRecords returnDfmVarRecords(int num_iter, int num_burn, int thin) const = 0;

	template <typename RecordType = MatDfmRecords>
	RecordType returnRecords(int num_iter, int num_burn, int thin) const;
};

struct MatDfmVarRecords : public MatDfmRecords {
	Eigen::MatrixXd factor_coef_record;
	Eigen::MatrixXd factor_prec_record;

	MatDfmVarRecords(int num_iter, int num_design, int size_factor, int lag)
	: MatDfmRecords(num_iter, num_design, size_factor),
		factor_coef_record(Eigen::MatrixXd::Zero(num_iter + 1, size_factor * lag)),
		factor_prec_record(Eigen::MatrixXd::Zero(num_iter + 1, size_factor)) {}
	
	MatDfmVarRecords(const Eigen::MatrixXd& factor_record, const Eigen::MatrixXd& factor_coef_record, const Eigen::MatrixXd& factor_prec_record)
	: MatDfmRecords(factor_record),
		factor_coef_record(factor_coef_record), factor_prec_record(factor_prec_record) {}
	
	virtual ~MatDfmVarRecords() = default;
	
	void assignRecords(
		int id,
		std::vector<Eigen::MatrixXd>& factor_mat,
		const Eigen::MatrixXd& factor_coef, const Eigen::VectorXd& factor_prec,
		int num_design, int size_factor
	) override {
		MatDfmRecords::assignRecords(id, factor_mat, num_design, size_factor);
		factor_coef_record.row(id) = factor_coef.reshaped();
		factor_prec_record.row(id) = factor_prec;
	}

	void appendRecords(BVHAR_LIST& list) override {
		list["F_record"] = factor_record;
		list["Rho_record"] = factor_coef_record;
		list["Lambda_record"] = factor_prec_record;
	}

	MatDfmVarRecords returnDfmVarRecords(int num_iter, int num_burn, int thin) const override {
		return MatDfmVarRecords(
			bvhar::thin_record(factor_record, num_iter, num_burn, thin).derived(),
			bvhar::thin_record(factor_coef_record, num_iter, num_burn, thin).derived(),
			bvhar::thin_record(factor_prec_record, num_iter, num_burn, thin).derived()
		);
	}
};

// template <>
// inline MatDfmRecords MatDfmRecords::returnRecords(int num_iter, int num_burn, int thin) const {
//   return returnDfmRecords(num_iter, num_burn, thin);
// }

template <>
inline MatDfmVarRecords MatDfmRecords::returnRecords(int num_iter, int num_burn, int thin) const {
  return returnDfmVarRecords(num_iter, num_burn, thin);
}

} // namespace baymar

#endif // BAYMAR_BAYES_MDFM_CONFIG_H