#ifndef BAYMAR_BAYES_MNIW_CONFIG_H
#define BAYMAR_BAYES_MNIW_CONFIG_H

#include <bvhar/base>
#include "../shrinkage/shrinkage.h"
// #include "../mdfm/augment.h"

namespace baecon {
namespace baymar {

struct MatMniwParams;
struct MatMniwRegParams;
struct MatMniwInits;
struct MatMniwRecords;

struct MatMniwParams : public bvhar::McmcParams {
	std::vector<Eigen::MatrixXd> _y;
	int _row, _col, _design;
	Eigen::MatrixXd _row_mean, _row_iw_scl;
	Eigen::MatrixXd _col_mean, _col_iw_scl;
	Eigen::VectorXd _row_prec, _col_prec;
	double _row_iw_df, _col_iw_df;
	int _row_row_coef, _row_col_coef;

	MatMniwParams(int num_iter, std::vector<Eigen::MatrixXd>& y, BVHAR_LIST& priors, const BVHAR_STRING& prefix = "", const BVHAR_STRING& suffix = "")
	: bvhar::McmcParams(num_iter),
		_y(y), _row(y[0].rows()), _col(y[0].cols()), _design(y.size()),
		_row_mean(BVHAR_CAST<Eigen::MatrixXd>(priors[prefix + "row_prior_mean" + suffix])), _row_iw_scl(BVHAR_CAST<Eigen::MatrixXd>(priors[prefix + "row_iw_scl" + suffix])),
		_col_mean(BVHAR_CAST<Eigen::MatrixXd>(priors[prefix + "col_prior_mean" + suffix])), _col_iw_scl(BVHAR_CAST<Eigen::MatrixXd>(priors[prefix + "col_iw_scl" + suffix])),
		_row_prec(BVHAR_CAST<Eigen::VectorXd>(priors[prefix + "row_prior_prec" + suffix])), _col_prec(BVHAR_CAST<Eigen::VectorXd>(priors[prefix + "col_prior_prec" + suffix])),
		_row_iw_df(BVHAR_CAST_DOUBLE(priors[prefix + "row_iw_df" + suffix])), _col_iw_df(BVHAR_CAST_DOUBLE(priors[prefix + "col_iw_df" + suffix])),
		_row_row_coef(_row_mean.rows()), _row_col_coef(_col_mean.rows()) {}
};

struct MatMniwRegParams : public MatMniwParams {
	std::vector<Eigen::SparseMatrix<double>> _x;
	int _row_exogen, _col_exogen, _lag_exogen;
	int _row_factor, _col_factor;

	MatMniwRegParams(
		int num_iter, std::vector<Eigen::SparseMatrix<double>>& x, std::vector<Eigen::MatrixXd>& y,
		BVHAR_LIST& priors,
		BVHAR_OPTIONAL<int> exogen_rows = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_cols = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_lag = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<int> factor_rows = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> factor_cols = BVHAR_NULLOPT
	)
	: MatMniwParams(num_iter, y, priors, ""),
		_x(x),
		_row_exogen(exogen_rows ? *exogen_rows : 0), _col_exogen(exogen_cols ? *exogen_cols : 0), _lag_exogen(exogen_lag ? *exogen_lag : 0),
		_row_factor(factor_rows ? *factor_rows : 0), _col_factor(factor_cols ? *factor_cols : 0) {
		_row_row_coef -= (_row_exogen + _row_factor);
		_row_col_coef -= (_col_exogen + _col_factor);
	}

	MatMniwRegParams(
		int num_iter, std::vector<Eigen::SparseMatrix<double>>& x, std::vector<Eigen::MatrixXd>& y,
		BVHAR_LIST& priors, const BVHAR_STRING& prefix, const BVHAR_STRING& suffix
	)
	: MatMniwParams(num_iter, y, priors, prefix, suffix),
		_x(x), _row_exogen(0), _col_exogen(0), _lag_exogen(0), _row_factor(0), _col_factor(0) {}
};

struct MatMniwInits {
	Eigen::MatrixXd _init_row_coef, _init_row_lower, _init_col_coef, _init_col_lower;

	MatMniwInits(BVHAR_LIST& init, const BVHAR_STRING& prefix = "", const BVHAR_STRING& suffix = "")
	: _init_row_coef(BVHAR_CAST<Eigen::MatrixXd>(init[prefix + "row_init_coef" + suffix])),
		_init_row_lower(BVHAR_CAST<Eigen::MatrixXd>(init[prefix + "row_init_lower" + suffix])),
		_init_col_coef(BVHAR_CAST<Eigen::MatrixXd>(init[prefix + "col_init_coef" + suffix])),
		_init_col_lower(BVHAR_CAST<Eigen::MatrixXd>(init[prefix + "col_init_lower" + suffix])) {}
};

struct MatMniwRecords {
	Eigen::MatrixXd row_coef_record;
	Eigen::MatrixXd row_sigma_record;
	Eigen::MatrixXd col_coef_record;
	Eigen::MatrixXd col_sigma_record;

	MatMniwRecords()
	: row_coef_record(), row_sigma_record(), col_coef_record(), col_sigma_record() {}

	MatMniwRecords(int num_iter, int num_row, int num_col, int nrow_row_coef, int nrow_col_coef)
	: row_coef_record(num_iter + 1, nrow_row_coef * num_row),
		row_sigma_record(num_iter + 1, num_row * (num_row + 1) / 2),
		col_coef_record(num_iter + 1, nrow_col_coef * num_col),
		col_sigma_record(num_iter + 1, num_col * (num_col + 1) / 2) {}
	
	MatMniwRecords(
		const Eigen::MatrixXd& coef_row_record, const Eigen::MatrixXd& row_sigma_record,
		const Eigen::MatrixXd& coef_col_record, const Eigen::MatrixXd& col_sigma_record
	)
	: row_coef_record(coef_row_record), row_sigma_record(row_sigma_record),
		col_coef_record(coef_col_record), col_sigma_record(col_sigma_record) {}
	
	MatMniwRecords(
		const Eigen::MatrixXd& coef_row_record, const Eigen::MatrixXd& row_sigma_record,
		const Eigen::MatrixXd& coef_col_record, const Eigen::MatrixXd& col_sigma_record,
		const Eigen::MatrixXd& exogen_row_coef_record, const Eigen::MatrixXd& exogen_col_coef_record
	)
	: row_coef_record(Eigen::MatrixXd::Zero(coef_row_record.rows(), coef_row_record.cols() + exogen_row_coef_record.cols())),
		row_sigma_record(row_sigma_record),
		col_coef_record(Eigen::MatrixXd::Zero(coef_col_record.rows(), coef_col_record.cols() + exogen_col_coef_record.cols())),
		col_sigma_record(col_sigma_record) {
		row_coef_record << coef_row_record, exogen_row_coef_record;
		col_coef_record << coef_col_record, exogen_col_coef_record;
	}

	MatMniwRecords(
		const Eigen::MatrixXd& coef_row_record, const Eigen::MatrixXd& row_sigma_record,
		const Eigen::MatrixXd& coef_col_record, const Eigen::MatrixXd& col_sigma_record,
		const Eigen::MatrixXd& exogen_row_coef_record, const Eigen::MatrixXd& exogen_col_coef_record,
		const Eigen::MatrixXd& factor_row_coef_record, const Eigen::MatrixXd& factor_col_coef_record
	)
	: row_coef_record(Eigen::MatrixXd::Zero(coef_row_record.rows(), coef_row_record.cols() + exogen_row_coef_record.cols() + factor_row_coef_record.cols())),
		row_sigma_record(row_sigma_record),
		col_coef_record(Eigen::MatrixXd::Zero(coef_col_record.rows(), coef_col_record.cols() + exogen_col_coef_record.cols() + factor_col_coef_record.cols())),
		col_sigma_record(col_sigma_record) {
		row_coef_record << coef_row_record, exogen_row_coef_record, factor_row_coef_record;
		col_coef_record << coef_col_record, exogen_col_coef_record, factor_col_coef_record;
	}

	void assignRecords(
		int id,
		const Eigen::MatrixXd& row_coef, const Eigen::MatrixXd& row_sig_lower,
		const Eigen::MatrixXd& col_coef, const Eigen::MatrixXd& col_sig_lower,
		int nrow_row_coef, int num_row, int nrow_row_exogen, int nrow_factor,
		int nrow_col_coef, int num_col, int nrow_col_exogen, int ncol_factor
	) {
		// double sign00, fro_norm;
		// if (nrow_row_coef > 0 && nrow_col_coef > 0) {
		// 	int lag = nrow_row_coef / num_row;
		// 	Eigen::MatrixXd mar_row_coef = row_coef.topRows(nrow_row_coef);
		// 	Eigen::MatrixXd mar_col_coef = col_coef.topRows(nrow_col_coef);
		// 	for (int i = 0; i < lag; ++i) {
		// 		// A_i / c_1 & B_i / c_1, c_1 = sign(A(0,0)) ||A||_F
		// 		sign00 = mar_row_coef(i * num_row, 0) > 0 ? 1.0 : -1.0;
		// 		fro_norm = mar_row_coef.middleRows(i * num_row, num_row).norm();
		// 		mar_row_coef.middleRows(i * num_row, num_row) /= sign00 * fro_norm;
		// 		mar_col_coef.middleRows(i * num_col, num_col) *= sign00 * fro_norm;
		// 	}
		// 	row_coef_record.row(id).head(nrow_row_coef * num_row) = mar_row_coef.reshaped();
		// 	col_coef_record.row(id).head(nrow_col_coef * num_col) = mar_col_coef.reshaped();
		// }
		row_coef_record.row(id).head(nrow_row_coef * num_row) = row_coef.topRows(nrow_row_coef).reshaped();
		col_coef_record.row(id).head(nrow_col_coef * num_col) = col_coef.topRows(nrow_col_coef).reshaped();
		// Should add restriction to exogen part later
		if (nrow_row_exogen > 0 && nrow_col_exogen > 0) {
			row_coef_record.row(id).segment(nrow_row_coef * num_row, nrow_row_exogen * num_row) = row_coef.middleRows(nrow_row_coef, nrow_row_exogen).reshaped();
			col_coef_record.row(id).segment(nrow_col_coef * num_col, nrow_col_exogen * num_col) = col_coef.middleRows(nrow_col_coef, nrow_col_exogen).reshaped();
		}
		// if (nrow_factor > 0 && ncol_factor > 0) {
		// 	Eigen::MatrixXd row_factor_coef = row_coef.bottomRows(nrow_factor);
		// 	Eigen::MatrixXd col_factor_coef = col_coef.bottomRows(ncol_factor);
		// 	sign00 = row_factor_coef(0, 0) > 0 ? 1.0 : -1.0;
		// 	fro_norm = row_factor_coef.norm();
		// 	row_factor_coef /= sign00 * fro_norm;
		// 	col_factor_coef *= sign00 * fro_norm;
		// 	row_coef_record.row(id).tail(nrow_factor * num_row) = row_factor_coef.reshaped();
		// 	col_coef_record.row(id).tail(ncol_factor * num_col) = col_factor_coef.reshaped();
		// }
		if (nrow_factor > 0 && ncol_factor > 0) {
			row_coef_record.row(id).tail(nrow_factor * num_row) = row_coef.bottomRows(nrow_factor).reshaped();
			col_coef_record.row(id).tail(ncol_factor * num_col) = col_coef.bottomRows(ncol_factor).reshaped();
		}
		// fro_norm = (row_sig_lower * row_sig_lower.transpose()).norm();
		int lower_id = 0;
		for (int j = 0; j < row_sig_lower.cols(); ++j) {
			for (int i = j; i < row_sig_lower.cols(); ++i) {
				// If 3x3: (0, 0) -> (1, 0) -> (2, 0) -> (1, 1) -> (2, 1) -> (2, 2)
				// Can be assigned in R:
				// matrix[lower.tri(matrix, diag = TRUE)] <- row_vector
				// matrix[upper.tri(matrix, diag = FALSE)] <- matrix[lower.tri(matrix, diag = FALSE)]
				row_sigma_record(id, lower_id++) = row_sig_lower.row(i).head(j + 1).dot(row_sig_lower.row(j).head(j + 1));
				// row_sigma_record(id, lower_id++) = row_sig_lower.row(i).head(j + 1).dot(row_sig_lower.row(j).head(j + 1)) / fro_norm;
			}
		}
		lower_id = 0;
		for (int j = 0; j < col_sig_lower.cols(); ++j) {
			for (int i = j; i < col_sig_lower.cols(); ++i) {
				col_sigma_record(id, lower_id++) = col_sig_lower.row(i).head(j + 1).dot(col_sig_lower.row(j).head(j + 1));
				// col_sigma_record(id, lower_id++) = col_sig_lower.row(i).head(j + 1).dot(col_sig_lower.row(j).head(j + 1)) * fro_norm;
			}
		}
	}

	void updateParams(
		const int id,
		Eigen::Ref<Eigen::MatrixXd> row_coef, Eigen::Ref<Eigen::MatrixXd> row_sig_lower,
		Eigen::Ref<Eigen::MatrixXd> col_coef, Eigen::Ref<Eigen::MatrixXd> col_sig_lower,
		int nrow_row_coef, int num_row,
		int nrow_col_coef, int num_col
	) {
		row_coef = bvhar::unvectorize(row_coef_record.row(id).head(nrow_row_coef * num_row).transpose(), num_row);
		col_coef = bvhar::unvectorize(col_coef_record.row(id).head(nrow_col_coef * num_col).transpose(), num_col);
		fill_lower(row_sig_lower, row_sigma_record.row(id).transpose());
		fill_lower(col_sig_lower, col_sigma_record.row(id).transpose());
	}

	BVHAR_LIST returnListRecords(
		int nrow_row_coef, int num_row, int nrow_row_exogen, int nrow_factor,
		int nrow_col_coef, int num_col, int nrow_col_exogen, int ncol_factor
	) {
		BVHAR_LIST res = BVHAR_CREATE_LIST(
			BVHAR_NAMED("A_record") = row_coef_record.leftCols(num_row * nrow_row_coef),
			BVHAR_NAMED("SigmaR_record") = row_sigma_record,
			BVHAR_NAMED("B_record") = col_coef_record.leftCols(num_col * nrow_col_coef),
			BVHAR_NAMED("SigmaC_record") = col_sigma_record
		);
		if (nrow_row_exogen > 0) {
			res["C_record"] = row_coef_record.middleCols(num_row * nrow_row_coef, num_row * nrow_row_exogen);
			res["D_record"] = col_coef_record.middleCols(num_col * nrow_col_coef, num_col * nrow_col_exogen);
		}
		if (nrow_factor > 0) {
			res["G_record"] = row_coef_record.rightCols(num_row * nrow_factor);
			res["H_record"] = col_coef_record.rightCols(num_col * ncol_factor);
		}
		// return BVHAR_CREATE_LIST(
		// 	BVHAR_NAMED("A_record") = row_coef_record,
		// 	BVHAR_NAMED("SigmaR_record") = row_sigma_record,
		// 	BVHAR_NAMED("B_record") = col_coef_record,
		// 	BVHAR_NAMED("SigmaC_record") = col_sigma_record
		// );
		return res;
	}

	MatMniwRecords returnMniwRecords(int num_iter, int num_burn, int thin) const {
		return MatMniwRecords(
			bvhar::thin_record(row_coef_record, num_iter, num_burn, thin).derived(),
			bvhar::thin_record(row_sigma_record, num_iter, num_burn, thin).derived(),
			bvhar::thin_record(col_coef_record, num_iter, num_burn, thin).derived(),
			bvhar::thin_record(col_sigma_record, num_iter, num_burn, thin).derived()
		);
	}

	template <typename RecordType = MatMniwRecords>
	RecordType returnRecords(int num_iter, int num_burn, int thin) const;
};

template <>
inline MatMniwRecords MatMniwRecords::returnRecords(int num_iter, int num_burn, int thin) const {
	return returnMniwRecords(num_iter, num_burn, thin);
}

inline void initialize_matmniw_record(
	std::unique_ptr<MatMniwRecords>& record, int chain_id, BVHAR_LIST& fit_record,
	BVHAR_STRING& a_name, BVHAR_STRING& sigr_name, BVHAR_STRING& b_name, BVHAR_STRING& sigc_name,
	BVHAR_OPTIONAL<BVHAR_STRING> c_name = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_STRING> d_name = BVHAR_NULLOPT,
	BVHAR_OPTIONAL<BVHAR_STRING> g_name = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_STRING> h_name = BVHAR_NULLOPT
) {
	BVHAR_PY_LIST row_coef_list = BVHAR_CAST_PY_LIST(fit_record[a_name]);
	BVHAR_PY_LIST row_sigma_list = BVHAR_CAST_PY_LIST(fit_record[sigr_name]);
	BVHAR_PY_LIST col_coef_list = BVHAR_CAST_PY_LIST(fit_record[b_name]);
	BVHAR_PY_LIST col_sigma_list = BVHAR_CAST_PY_LIST(fit_record[sigc_name]);
	if ((c_name && d_name) && (g_name && h_name)) {
		BVHAR_PY_LIST exogen_row_list = BVHAR_CAST_PY_LIST(fit_record[*c_name]);
		BVHAR_PY_LIST exogen_col_list = BVHAR_CAST_PY_LIST(fit_record[*d_name]);
		BVHAR_PY_LIST factor_row_list = BVHAR_CAST_PY_LIST(fit_record[*g_name]);
		BVHAR_PY_LIST factor_col_list = BVHAR_CAST_PY_LIST(fit_record[*h_name]);
		record = std::make_unique<MatMniwRecords>(
			BVHAR_CAST<Eigen::MatrixXd>(row_coef_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(row_sigma_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(col_coef_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(col_sigma_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(exogen_row_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(exogen_col_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(factor_row_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(factor_col_list[chain_id])
		);
	} else if (c_name && d_name) {
		BVHAR_PY_LIST exogen_row_list = BVHAR_CAST_PY_LIST(fit_record[*c_name]);
		BVHAR_PY_LIST exogen_col_list = BVHAR_CAST_PY_LIST(fit_record[*d_name]);
		record = std::make_unique<MatMniwRecords>(
			BVHAR_CAST<Eigen::MatrixXd>(row_coef_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(row_sigma_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(col_coef_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(col_sigma_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(exogen_row_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(exogen_col_list[chain_id])
		);
	} else {
		record = std::make_unique<MatMniwRecords>(
			BVHAR_CAST<Eigen::MatrixXd>(row_coef_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(row_sigma_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(col_coef_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(col_sigma_list[chain_id])
		);
	}
}

} // namespace baymar
} // namespace baecon

#endif // BAYMAR_BAYES_MNIW_CONFIG_H