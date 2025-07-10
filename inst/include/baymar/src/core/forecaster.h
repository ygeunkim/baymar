#ifndef BAYMAR_CORE_FORECASTER_H
#define BAYMAR_CORE_FORECASTER_H

#include <bvhar/base>
#include <bvhar/utils>

namespace baymar {

class MatExogenForecaster;
class MatErrorGenerator;
class MatGaussianErrorGenerator;

class MatExogenForecaster : public bvhar::ExogenForecaster<Eigen::MatrixXd, Eigen::MatrixXd> {
public:
	MatExogenForecaster(int lag, const Eigen::MatrixXd& exogen, int num_exogen, int num_row, int num_col)
	: bvhar::ExogenForecaster<Eigen::MatrixXd, Eigen::MatrixXd>(lag, exogen),
		nrow_exogen(exogen.rows() / num_exogen), ncol_exogen(exogen.cols()),
		nrow_row_exogen((lag + 1) * nrow_exogen), nrow_col_exogen((lag + 1) * ncol_exogen),
		num_row(num_row), num_col(num_col),
		row_coef(nrow_row_exogen, num_row), col_coef(nrow_col_exogen, num_col) {
		BVHAR_DEBUG_LOG(debug_logger, "Constructor: num_exogen={}, num_row={}, num_col={}", num_exogen, num_row, num_col);
		// last_pvec = Eigen::MatrixXd::Zero((lag + 1) * nrow_exogen, (lag + 1) * ncol_exogen);
		// exogen: rbind(X_{T - s}, ..., X_{T + h})
		// last_pvec = x_(T + h), ..., x_(T + h - s)
		last_pvec = Eigen::MatrixXd::Zero(nrow_exogen, ncol_exogen);
	}
	virtual ~MatExogenForecaster() = default;
	
	void appendForecast(Eigen::MatrixXd& point_forecast, const int h) override {
		BVHAR_DEBUG_LOG(debug_logger, "appendForecast(point_forecast, h) called");
		for (int i = 0; i < lag + 1; ++i) {
			last_pvec = exogen.middleRows((lag + h - i) * nrow_exogen, nrow_exogen); // x_(T + h - i)
			point_forecast += row_coef.middleRows(i * nrow_exogen, nrow_exogen).transpose() * last_pvec * col_coef.middleRows(i * ncol_exogen, ncol_exogen);
		}
	}

protected:
	int nrow_exogen, ncol_exogen, nrow_row_exogen, nrow_col_exogen, num_row, num_col;
	Eigen::MatrixXd row_coef, col_coef;

	void updateCoef(Eigen::Ref<const Eigen::MatrixXd> row_coef_mat, Eigen::Ref<const Eigen::MatrixXd> col_coef_mat) {
		BVHAR_DEBUG_LOG(debug_logger, "updateCoef() called");
		row_coef = row_coef_mat;
		col_coef = col_coef_mat;
	}
};

class MatErrorGenerator : public bvhar::AutoregGenerator<Eigen::MatrixXd, Eigen::MatrixXd> {
public:
	MatErrorGenerator(int num_row, int num_col, unsigned int seed)
	: bvhar::AutoregGenerator<Eigen::MatrixXd, Eigen::MatrixXd>(seed) {
		error_term = Eigen::MatrixXd::Zero(num_row, num_col);
	}
	virtual ~MatErrorGenerator() = default;
};

class MatGaussianErrorGenerator : public MatErrorGenerator {
public:
	MatGaussianErrorGenerator(
		const Eigen::MatrixXd& error_mean,
		const Eigen::MatrixXd& error_sig_row, const Eigen::MatrixXd& error_sig_col,
		unsigned int seed
	)
	: MatErrorGenerator(error_sig_row.cols(), error_sig_col.cols(), seed),
		error_mean(error_mean), error_sig_row(error_sig_row), error_sig_col(error_sig_col) {}
	virtual ~MatGaussianErrorGenerator() = default;

	void appendError(Eigen::MatrixXd& point_forecast) override {
		error_term = bvhar::sim_mn(error_mean, error_sig_row, error_sig_col, false, rng);
	}

private:
	Eigen::MatrixXd error_mean, error_sig_row, error_sig_col;
};

} // namespace baymar

#endif // BAYMAR_CORE_FORECASTER_H