#ifndef BAYMAR_BAYES_MDFM_CONFIG_H
#define BAYMAR_BAYES_MDFM_CONFIG_H

#include <bvhar/base>
#include "../misc/draw.h"
#include "../../math/design.h"
#include "../mniw/config.h"

namespace baymar {

struct MatDfmParams;
struct MatDfmVarParams;
struct MatDfmMarParams;
struct MatDfmInits;
struct MatDfmVarInits;
struct MatDfmMarInits;
struct MatDfmRecords;
struct MatDfmVarRecords;
struct MatDfmMarRecords;

struct MatDfmParams {
	int _nrow_factor, _ncol_factor, _size_factor, _lag;

	MatDfmParams(int lag, int nrow_factor, int ncol_factor)
	: _nrow_factor(nrow_factor), _ncol_factor(ncol_factor), _size_factor(_nrow_factor * _ncol_factor),
		_lag(lag) {}

	MatDfmParams(BVHAR_LIST& priors)
	: _nrow_factor(BVHAR_CAST_INT(priors["nrow_factor"])), _ncol_factor(BVHAR_CAST_INT(priors["ncol_factor"])), _size_factor(_nrow_factor * _ncol_factor),
		_lag(BVHAR_CAST_INT(priors["lag"])) {}
};

struct MatDfmVarParams : public MatDfmParams {
	Eigen::VectorXd _sig_shp, _sig_scl;
	Eigen::VectorXd _mean, _prec;

	MatDfmVarParams(int lag, int nrow_factor, int ncol_factor)
	: MatDfmParams(lag, nrow_factor, ncol_factor),
		_sig_shp(Eigen::VectorXd::Constant(_size_factor, 2.0)),
		_sig_scl(Eigen::VectorXd::Ones(_size_factor)),
		_mean(Eigen::VectorXd::Zero(_lag)), _prec(Eigen::VectorXd::Ones(_lag)) {}

	MatDfmVarParams(BVHAR_LIST& priors)
	: MatDfmParams(priors),
		// _sig_shp(BVHAR_CAST<Eigen::VectorXd>(priors["shape"])),
		// _sig_scl(BVHAR_CAST<Eigen::VectorXd>(priors["scale"])),
		_mean(Eigen::VectorXd::Zero(_lag)), _prec(Eigen::VectorXd::Ones(_lag)) {
		_sig_shp = BVHAR_CAST<Eigen::VectorXd>(priors["shape"]).size() == 1
			? Eigen::VectorXd::Constant(_size_factor, BVHAR_CAST_INT(priors["shape"]))
			: BVHAR_CAST<Eigen::VectorXd>(priors["shape"]);
		_sig_scl = BVHAR_CAST<Eigen::VectorXd>(priors["scale"]).size() == 1
			? Eigen::VectorXd::Constant(_size_factor, BVHAR_CAST_INT(priors["scale"]))
			: BVHAR_CAST<Eigen::VectorXd>(priors["scale"]);
	}
};

struct MatDfmMarParams : public MatDfmParams {
	std::vector<Eigen::MatrixXd> empty_y;
	MatMniwParams mniw_params;

	MatDfmMarParams(BVHAR_LIST& priors)
	: MatDfmParams(priors), empty_y(),
		mniw_params(0, empty_y, priors) {}
};

struct MatDfmInits {
	MatDfmInits() {}
};

struct MatDfmVarInits : public MatDfmInits {
	Eigen::MatrixXd _init_factor_coef;
	Eigen::VectorXd _init_factor_prec;

	MatDfmVarInits(int size_factor, int lag)
	: _init_factor_coef(Eigen::MatrixXd::Zero(size_factor, lag)),
		_init_factor_prec(Eigen::VectorXd::Ones(size_factor)) {}

	MatDfmVarInits(BVHAR_LIST& init)
	: _init_factor_coef(BVHAR_CAST<Eigen::MatrixXd>(init["factor_arcoef_init"])),
		_init_factor_prec(BVHAR_CAST<Eigen::VectorXd>(init["factor_arprec_init"])) {}
};

struct MatDfmMarInits : public MatDfmInits {
	MatMniwInits mniw_init;

	MatDfmMarInits(BVHAR_LIST& init)
	: mniw_init(init) {}
};

struct MatDfmRecords {
	Eigen::MatrixXd factor_record;

	MatDfmRecords() {}

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
	) {
		assignRecords(id, factor_mat, num_design, size_factor);
	}

	virtual void assignRecords(
		int id,
		std::vector<Eigen::MatrixXd>& factor_mat,
		const Eigen::MatrixXd& row_coef, const Eigen::MatrixXd& row_sig_lower,
		const Eigen::MatrixXd& col_coef, const Eigen::MatrixXd& col_sig_lower,
		int num_design,
		int nrow_row_coef, int num_row, int nrow_col_coef, int num_col
	) {
		assignRecords(id, factor_mat, num_design, num_row * num_col);
	}

	// BVHAR_LIST returnListRecords(int nrow_row_coef, int num_row, int nrow_col_coef, int num_col, int num_design, int size_factor) {
	// 	BVHAR_LIST res = MatMniwRecords::returnListRecords(nrow_row_coef, num_row, 0, 0, nrow_col_coef, num_col, 0, 0);
	// 	// res["F_record"] = factor_record;
	// 	return res;
	// }

	virtual void appendRecords(BVHAR_LIST& list) {
		list["F_record"] = factor_record;
	}

	virtual void updateParams(const int id, Eigen::Ref<Eigen::MatrixXd> factor_coef, Eigen::Ref<Eigen::VectorXd> factor_sig, const int lag) {}

	virtual void updateParams(
		const int id, Eigen::Ref<Eigen::MatrixXd> row_coef, Eigen::Ref<Eigen::MatrixXd> row_sig_lower,
		Eigen::Ref<Eigen::MatrixXd> col_coef, Eigen::Ref<Eigen::MatrixXd> col_sig_lower
	) {}

	MatDfmRecords returnDfmRecords(int num_iter, int num_burn, int thin) const {
		return MatDfmRecords(
			bvhar::thin_record(factor_record, num_iter, num_burn, thin).derived()
		);
	}

	virtual MatDfmVarRecords returnDfmVarRecords(int num_iter, int num_burn, int thin) const;

	template <typename RecordType = MatDfmRecords>
	RecordType returnRecords(int num_iter, int num_burn, int thin) const;
};

struct MatDfmVarRecords : public MatDfmRecords {
	Eigen::MatrixXd factor_coef_record;
	Eigen::MatrixXd factor_prec_record;

	MatDfmVarRecords() {}

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
		factor_prec_record.row(id) = factor_prec; // this is sigma, not precision -> change the name
	}

	void appendRecords(BVHAR_LIST& list) override {
		list["F_record"] = factor_record;
		list["Rho_record"] = factor_coef_record;
		list["Lambda_record"] = factor_prec_record;
	}

	void updateParams(const int id, Eigen::Ref<Eigen::MatrixXd> factor_coef, Eigen::Ref<Eigen::VectorXd> factor_sig, const int lag) override {
		Eigen::MatrixXd temp_coef = bvhar::unvectorize(factor_coef_record.row(id).transpose(), lag);
		int size_factor = factor_sig.size();
		for (int i = 0; i < lag; ++i) {
			// factor_coef.middleRows(i * size_factor, size_factor) = factor_coef_record.row(id).segment(i * size_factor, size_factor).asDiagonal();
			factor_coef.middleRows(i * size_factor, size_factor) = temp_coef.col(i).asDiagonal();
		}
		// factor_sig.array() = 1 / factor_prec_record.row(id).array();
		factor_sig = factor_prec_record.row(id).transpose();
	}

	MatDfmVarRecords returnDfmVarRecords(int num_iter, int num_burn, int thin) const override {
		return MatDfmVarRecords(
			bvhar::thin_record(factor_record, num_iter, num_burn, thin).derived(),
			bvhar::thin_record(factor_coef_record, num_iter, num_burn, thin).derived(),
			bvhar::thin_record(factor_prec_record, num_iter, num_burn, thin).derived()
		);
	}
};

struct MatDfmMarRecords : public MatDfmRecords {
	MatMniwRecords mniw_record;

	MatDfmMarRecords(int num_iter, int num_design, int nrow_factor, int ncol_factor, int lag)
	: MatDfmRecords(num_iter, num_design, nrow_factor * ncol_factor),
		mniw_record(num_iter, nrow_factor, ncol_factor, nrow_factor * lag, ncol_factor * lag) {}

	MatDfmMarRecords(
		const Eigen::MatrixXd& factor_record,
		const Eigen::MatrixXd& coef_row_record, const Eigen::MatrixXd& row_sigma_record,
		const Eigen::MatrixXd& coef_col_record, const Eigen::MatrixXd& col_sigma_record
	)
	: MatDfmRecords(factor_record),
		mniw_record(coef_row_record, row_sigma_record, coef_col_record, col_sigma_record) {}

	void assignRecords(
		int id,
		std::vector<Eigen::MatrixXd>& factor_mat,
		const Eigen::MatrixXd& row_coef, const Eigen::MatrixXd& row_sig_lower,
		const Eigen::MatrixXd& col_coef, const Eigen::MatrixXd& col_sig_lower,
		int num_design,
		int nrow_row_coef, int num_row, int nrow_col_coef, int num_col
	) override {
		MatDfmRecords::assignRecords(id, factor_mat, num_design, num_row * num_col);
		mniw_record.assignRecords(
			id,
			row_coef, row_sig_lower,
			col_coef, col_sig_lower,
			nrow_row_coef, num_row, 0, 0,
			nrow_col_coef, num_col, 0, 0
		);
	}

	void appendRecords(BVHAR_LIST& list) override {
		list["F_record"] = factor_record;
		list["FA_record"] = mniw_record.row_coef_record;
		list["OmegaR_record"] = mniw_record.row_sigma_record;
		list["FB_record"] = mniw_record.col_coef_record;
		list["OmegaC_record"] = mniw_record.col_sigma_record;
	}

	void updateParams(
		const int id, Eigen::Ref<Eigen::MatrixXd> row_coef, Eigen::Ref<Eigen::MatrixXd> row_sig_lower,
		Eigen::Ref<Eigen::MatrixXd> col_coef, Eigen::Ref<Eigen::MatrixXd> col_sig_lower
	) override {
		mniw_record.updateParams(
			id, row_coef, row_sig_lower, col_coef, col_sig_lower,
			row_coef.rows(), row_coef.cols(),
			col_coef.rows(), col_coef.cols()
		);
	}
};

inline MatDfmVarRecords MatDfmRecords::returnDfmVarRecords(int num_iter, int num_burn, int thin) const {
	return MatDfmVarRecords();
}

template <>
inline MatDfmRecords MatDfmRecords::returnRecords(int num_iter, int num_burn, int thin) const {
  return returnDfmRecords(num_iter, num_burn, thin);
}

template <>
inline MatDfmVarRecords MatDfmRecords::returnRecords(int num_iter, int num_burn, int thin) const {
  return returnDfmVarRecords(num_iter, num_burn, thin);
}

inline void initialize_matdfm_record(
	std::unique_ptr<MatDfmRecords>& record, int chain_id, BVHAR_LIST& dfm_record,
	BVHAR_STRING& factor_name,
	BVHAR_OPTIONAL<BVHAR_STRING> rho_name = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_STRING> lambda_name = BVHAR_NULLOPT,
	BVHAR_OPTIONAL<BVHAR_STRING> fa_name = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_STRING> fb_name = BVHAR_NULLOPT,
	BVHAR_OPTIONAL<BVHAR_STRING> omegar_name = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_STRING> omegac_name = BVHAR_NULLOPT
) {
	BVHAR_PY_LIST factor_list = dfm_record[factor_name];
	if (rho_name && lambda_name) {
		BVHAR_PY_LIST factor_coef_list = dfm_record[*rho_name];
		BVHAR_PY_LIST factor_prec_list = dfm_record[*lambda_name];
		record = std::make_unique<MatDfmVarRecords>(
			BVHAR_CAST<Eigen::MatrixXd>(factor_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(factor_coef_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(factor_prec_list[chain_id])
		);
	} else if (fa_name && fb_name && omegar_name && omegac_name) {
		BVHAR_PY_LIST row_coef_list = dfm_record[*fa_name];
		BVHAR_PY_LIST col_coef_list = dfm_record[*fb_name];
		BVHAR_PY_LIST row_sig_list = dfm_record[*omegar_name];
		BVHAR_PY_LIST col_sig_list = dfm_record[*omegac_name];
		record = std::make_unique<MatDfmMarRecords>(
			BVHAR_CAST<Eigen::MatrixXd>(factor_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(row_coef_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(row_sig_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(col_coef_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(col_sig_list[chain_id])
		);
	} else {
		record = std::make_unique<MatDfmRecords>(
			BVHAR_CAST<Eigen::MatrixXd>(factor_list[chain_id])
		);
	}
}

} // namespace baymar

#endif // BAYMAR_BAYES_MDFM_CONFIG_H