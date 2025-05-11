#ifndef BAYMAR_MATH_DESIGN_H
#define BAYMAR_MATH_DESIGN_H

#include <bvhar/utils>

namespace baymar {

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