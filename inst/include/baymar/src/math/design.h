#ifndef BAYMAR_MATH_DESIGN_H
#define BAYMAR_MATH_DESIGN_H

#include <bvhar/utils>

namespace baymar {

inline Eigen::SparseMatrix<double> build_blk_design(const std::vector<Eigen::MatrixXd>& y, int lag) {
	int num_row = y[0].rows();
	int num_col = y[0].cols();
	Eigen::SparseMatrix<double> x(num_row * lag, num_col * lag);
	Eigen::MatrixXd dense_x = x.toDense();
	for (int i = 0; i < lag; ++i) {
		dense_x.block(i * num_row, i * num_col, num_row, num_col) = y[i];
	}
	return dense_x.sparseView();
}

// Y_{p + 1}, ..., Y_T
inline std::vector<Eigen::MatrixXd> build_mar_response(const std::vector<Eigen::MatrixXd>& y, int lag) {
	int num_design = y.size() - lag;
	std::vector<Eigen::MatrixXd> response(num_design);
	for (int i = 0; i < num_design; ++i) {
		response[i] = y[i + lag];
	}
	return response;
}

// diag(Y_{t - 1}, ..., Y_{t - p}), t = p + 1, ..., T
// @param y Y_1, ..., Y_T
// @param lag MAR lag
inline std::vector<Eigen::SparseMatrix<double>> build_mar_design(const std::vector<Eigen::MatrixXd>& y, int lag,
																																 int nrow_factor = 0, int ncol_factor = 0) {
	int num_design = y.size() - lag;
	int num_row = y[0].rows();
	int num_col = y[0].cols();
	std::vector<Eigen::SparseMatrix<double>> x(num_design); // t = p + 1, ..., T
	// Eigen::MatrixXd dense_x = Eigen::MatrixXd::Zero(num_row * lag + nrow_factor, num_col * lag + ncol_factor);
	Eigen::MatrixXd dense_x = Eigen::MatrixXd::Zero(num_row * lag, num_col * lag);
	for (int i = 0; i < num_design; ++i) {
		for (int j = 0; j < lag; ++j) {
			dense_x.block(j * num_row, j * num_col, num_row, num_col) = y[lag + i - j - 1]; // diag(Y_{t - 1}, ..., Y_{t - p})
		}
		x[i] = dense_x.sparseView();
	}
	return x;
}

// diag(Y_{t - 1}, ..., Y_{t - p}, X_t, ..., X_{t - s}), t = p + 1, ..., T
// @param exogen X_1, ..., X_T
// @param exogen_lag s
inline std::vector<Eigen::SparseMatrix<double>> build_mar_design(const std::vector<Eigen::MatrixXd>& y,
																																 const std::vector<Eigen::MatrixXd>& exogen,
																																 int lag, int exogen_lag,
																																 int nrow_factor = 0, int ncol_factor = 0) {
	int num_design = y.size() - lag;
	int num_row = y[0].rows();
	int num_col = y[0].cols();
	int nrow_exogen = exogen[0].rows();
	int ncol_exogen = exogen[0].cols();
	std::vector<Eigen::SparseMatrix<double>> design(num_design); // t = p + 1, ..., T
	// Eigen::MatrixXd dense_x = Eigen::MatrixXd::Zero(num_row * lag + nrow_exogen * (exogen_lag + 1) + nrow_factor, num_col * lag + ncol_exogen * (exogen_lag + 1) + ncol_factor);
	Eigen::MatrixXd dense_x = Eigen::MatrixXd::Zero(num_row * lag + nrow_exogen * (exogen_lag + 1), num_col * lag + ncol_exogen * (exogen_lag + 1));
	for (int i = 0; i < num_design; ++i) {
		for (int j = 0; j < lag; ++j) {
			dense_x.block(j * num_row, j * num_col, num_row, num_col) = y[lag + i - j - 1]; // Y_{t - 1}, ..., Y_{t - p}
		}
		for (int j = 0; j < exogen_lag; ++j) {
			dense_x.block(
				lag * num_row + j * nrow_exogen,
				lag * num_col + j * ncol_exogen,
				nrow_exogen,
				ncol_exogen
			) = exogen[lag + i - j]; // X_t, ..., X_{t - s}
		}
		design[i] = dense_x.sparseView();
	}
	return design;
}

// Y_1, ..., Y_T
inline std::vector<Eigen::MatrixXd> marmatrix_to_vector(const Eigen::MatrixXd& y, int num_row) {
	int num_data = y.rows() / num_row;
	std::vector<Eigen::MatrixXd> response(num_data);
	for (int i = 0; i < num_data; ++i) {
		response[i] = y.middleRows(num_row * i, num_row);
	}
	return response;
}

// y is (Y_1^T, ..., Y_T^T)^T
inline Eigen::MatrixXd build_dense_design(const Eigen::MatrixXd& y, int lag) {
	int num_row = y.rows() / lag;
	int num_col = y.cols();
	Eigen::MatrixXd dense_x = Eigen::MatrixXd::Zero(num_row * lag, num_col * lag);
	for (int i = 0; i < lag; ++i) {
		dense_x.block(i * num_row, i * num_col, num_row, num_col) = y.middleRows(i * num_row, num_row);
	}
	return dense_x;
}

inline void fill_lower(Eigen::Ref<Eigen::MatrixXd> lower_matrix, Eigen::Ref<const Eigen::VectorXd> lower_vec) {
	int dim = lower_matrix.cols();
  int id = 0;
	int len = 0;
	for (int i = 0; i < dim; ++i) {
		len = dim - i;
		lower_matrix.col(i).segment(i, len) = lower_vec.segment(id, len);
		id += len;
	}
}

// Get Y_{p + 1} = A^T X B
// @param y Y_{p + 1}
// @param x Initial diag(Y_1, ..., Y_p)
// @param row_coef A
// @param col_coef B
inline void update_mar(Eigen::Ref<Eigen::MatrixXd> y,
											 Eigen::SparseMatrix<double>& x,
											 Eigen::Ref<Eigen::MatrixXd> row_coef,
											 Eigen::Ref<Eigen::MatrixXd> col_coef) {
	y = row_coef.transpose() * x * col_coef;
}

// Change x to diag(Y_2, ..., Y_p, y)
// @param x diag(Y_1, ..., Yp)
// @param y y
// @param num_row Row of y
// @param num_col Column of y
inline void update_x(Eigen::SparseMatrix<double>& x, Eigen::Ref<Eigen::MatrixXd> y, int num_row, int num_col) {
	int lag = x.rows() / num_row;
	Eigen::MatrixXd dense_x = x.toDense();
	for (int i = 0; i < lag - 1; ++i) {
		dense_x.block(i * num_row, i * num_col, num_row, num_col) = dense_x.block((i + 1) * num_row, (i + 1) * num_col, num_row, num_col);
	}
	dense_x.block((lag - 1) * num_row, (lag - 1) * num_col, num_row, num_col) = y;
	x = dense_x.sparseView();
}

inline void append_x(Eigen::SparseMatrix<double>& x, Eigen::Ref<Eigen::MatrixXd> new_x, int num_row, int num_col) {
	Eigen::MatrixXd dense_x = x.toDense();
	dense_x.bottomRightCorner(num_row, num_col) = new_x;
	x = dense_x.sparseView();
}

} // namespace baymar

#endif // BAYMAR_MATH_DESIGN_H