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

// inline std::vector<Eigen::SparseMatrix<double>> build_mar_design(const std::vector<Eigen::MatrixXd>& y, int lag) {
// 	int num_row = y[0].rows();
// 	int num_col = y[0].cols();
// 	int num_design = y.size();
// 	std::vector<Eigen::SparseMatrix<double>> x(num_design);
// 	for (int i = 0; i < num_design; ++i) {
// 		Eigen::MatrixXd dense_x(num_row * lag, num_col * lag);
// 		for (int j = i; j < i + lag; ++j) {
// 			dense_x.block(j * num_row, j * num_col, num_row, num_col) = y[j];
// 		}
// 		x[i] = dense_x.sparseView();
// 	}
// 	return x;
// }

inline std::vector<Eigen::SparseMatrix<double>> build_mar_design(const Eigen::MatrixXd& y, int num_design, int num_row, int num_col, int lag) {
	// int num_row = y.rows() / num_design;
	// int num_col = y.cols() / num_design;
	std::vector<Eigen::SparseMatrix<double>> x(num_design);
	Eigen::MatrixXd dense_x = Eigen::MatrixXd(num_row * lag, num_col * lag);
	for (int i = 0; i < num_design; ++i) {
		// for (int j = i; j < i + lag; ++j) {
		// 	dense_x.block(j * num_row, j * num_col, num_row, num_col) = y.middleRows(j * num_row, num_row);
		// }
		for (int j = 0; j < lag; ++j) {
			dense_x.block(j * num_row, j * num_col, num_row, num_col) = y.middleRows((i + j) * num_row, num_row); // is this right?
		}
		x[i] = dense_x.sparseView();
	}
	return x;
}
// -> overloading with exogen and exogen_lag

// Y_{p + 1}, ..., Y_T
inline std::vector<Eigen::MatrixXd> marmatrix_to_vector(const Eigen::MatrixXd& y, int num_row, int lag) {
	int num_design = y.rows() / num_row - lag;
	std::vector<Eigen::MatrixXd> response(num_design);
	for (int i = 0; i < num_design; ++i) {
		response[i] = y.middleRows(num_row * (lag + i), num_row);
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

} // namespace baymar

#endif // BAYMAR_MATH_DESIGN_H