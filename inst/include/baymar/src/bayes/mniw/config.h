#ifndef BAYMAR_BAYES_MNIW_CONFIG_H
#define BAYMAR_BAYES_MNIW_CONFIG_H

#include <bvhar/base>
#include "../shrinkage/shrinkage.h"

namespace baymar {

struct MatMniwParams;
struct MatMniwInits;
struct MatMniwRecords;

struct MatMniwParams : public bvhar::McmcParams {
	std::vector<Eigen::SparseMatrix<double>> _x;
	std::vector<Eigen::MatrixXd> _y;
	int _row, _col, _design;
	Eigen::MatrixXd _row_mean, _row_iw_scl;
	Eigen::MatrixXd _col_mean, _col_iw_scl;
	Eigen::VectorXd _row_prec, _col_prec;
	double _row_iw_df, _col_iw_df;

	MatMniwParams(int num_iter, std::vector<Eigen::SparseMatrix<double>>& x, std::vector<Eigen::MatrixXd>& y, LIST& priors)
	: bvhar::McmcParams(num_iter),
		_x(x), _y(y),
		_row(y[0].rows()), _col(y[0].cols()), _design(y.size()),
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

struct MatMniwRecords {
	Eigen::MatrixXd row_coef_record;
	Eigen::MatrixXd row_sigma_record;
	Eigen::MatrixXd col_coef_record;
	Eigen::MatrixXd col_sigma_record;

	MatMniwRecords(int num_iter, int num_row, int num_col, int nrow_row_coef, int nrow_col_coef)
	: row_coef_record(num_iter + 1, nrow_row_coef * num_row),
		row_sigma_record(num_iter + 1, num_row * (num_row + 1) / 2),
		col_coef_record(num_iter + 1, nrow_col_coef * num_col),
		col_sigma_record(num_iter + 1, num_col * (num_col + 1) / 2) {}
	
	MatMniwRecords(
		const Eigen::MatrixXd& row_coef_record, const Eigen::MatrixXd& row_sigma_record,
		const Eigen::MatrixXd& col_coef_record, const Eigen::MatrixXd& col_sigma_record
	)
	: row_coef_record(row_coef_record), row_sigma_record(row_sigma_record),
		col_coef_record(col_coef_record), col_sigma_record(col_sigma_record) {}
	
	void assignRecords(
		int id,
		const Eigen::MatrixXd row_coef, const Eigen::MatrixXd row_sig_lower,
		const Eigen::MatrixXd col_coef, const Eigen::MatrixXd col_sig_lower
	) {
		row_coef_record.row(id) = row_coef.reshaped();
		col_coef_record.row(id) = col_coef.reshaped();
		int lower_id = 0;
		for (int j = 0; j < row_sig_lower.cols(); ++j) {
			for (int i = j; i < row_sig_lower.cols(); ++i) {
				// If 3x3: (0, 0) -> (1, 0) -> (2, 0) -> (1, 1) -> (2, 1) -> (2, 2)
				// Can be assigned in R:
				// matrix[lower.tri(matrix, diag = TRUE)] <- row_vector
				// matrix[upper.tri(matrix, diag = FALSE)] <- matrix[lower.tri(matrix, diag = FALSE)]
				row_sigma_record(id, lower_id++) = row_sig_lower.row(i).head(j + 1).dot(row_sig_lower.row(j).head(j + 1));
			}
		}
		lower_id = 0;
		for (int j = 0; j < col_sig_lower.cols(); ++j) {
			for (int i = j; i < col_sig_lower.cols(); ++i) {
				col_sigma_record(id, lower_id++) = col_sig_lower.row(i).head(j + 1).dot(col_sig_lower.row(j).head(j + 1));
			}
		}
	}

	LIST returnListRecords() {
		return CREATE_LIST(
			NAMED("A_record") = row_coef_record,
			NAMED("SigmaR_record") = row_sigma_record,
			NAMED("B_record") = col_coef_record,
			NAMED("SigmaC_record") = col_sigma_record
		);
	}
};

} // namespace baymar

#endif // BAYMAR_BAYES_MNIW_CONFIG_H