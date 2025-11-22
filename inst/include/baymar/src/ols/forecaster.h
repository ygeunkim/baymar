#ifndef BAYMAR_OLS_FORECASTER_H
#define BAYMAR_OLS_FORECASTER_H

#include "../core/forecaster.h"
#include "../math/design.h"

namespace baecon {
namespace baymar {

class MatForecaster;
class MarForecaster;

class MatForecaster : public bvhar::MultistepForecaster<Eigen::MatrixXd, Eigen::MatrixXd> {
public:
	MatForecaster(
		int step, const Eigen::MatrixXd& y, int lag,
		const Eigen::MatrixXd& row_coef, const Eigen::MatrixXd& col_coef,
		BVHAR_OPTIONAL<std::unique_ptr<MatExogenForecaster>> exogen_forecaster = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<std::unique_ptr<MatErrorGenerator>> dgp_updater = BVHAR_NULLOPT
	)
	: bvhar::MultistepForecaster<Eigen::MatrixXd, Eigen::MatrixXd>(step, y, lag),
		num_row(row_coef.cols()), num_col(col_coef.cols()),
		row_coef(row_coef), col_coef(col_coef) {
		initLagged();
		if (exogen_forecaster) {
			exogen_updater = std::move(*exogen_forecaster);
		}
		if (dgp_updater) {
			error_updater = std::move(*dgp_updater);
		}
	}
	virtual ~MatForecaster() = default;

protected:
	std::unique_ptr<MatExogenForecaster> exogen_updater;
	std::unique_ptr<MatErrorGenerator> error_updater;
	int num_row, num_col;
	Eigen::MatrixXd row_coef, col_coef;

	void initLagged() override {
		BVHAR_DEBUG_LOG(debug_logger, "initLagged() called");
		last_pvec = build_dense_design(response, num_row, lag);
		point_forecast = Eigen::MatrixXd::Zero(num_row, num_col);
		pred_save = Eigen::MatrixXd::Zero(step * num_row, num_col);
		tmp_vec = last_pvec.block(num_row, num_col, num_row * (lag - 1), num_col * (lag - 1));
	}

	void setRecursion() override {
		BVHAR_DEBUG_LOG(debug_logger, "setRecursion() called");
		last_pvec.bottomRightCorner(num_row * (lag - 1), num_col * (lag - 1)) = tmp_vec;
		last_pvec.topLeftCorner(num_row, num_col) = point_forecast;
	}

	void updateRecursion() override {
		BVHAR_DEBUG_LOG(debug_logger, "updateRecursion() called");
		tmp_vec = last_pvec.topLeftCorner(num_row * (lag - 1), num_col * (lag - 1));
	}

	void updatePred(const int h, const int i) override {
		BVHAR_DEBUG_LOG(debug_logger, "updatePred(h={}, i={}) called", h, i);
		computeMean();
		if (exogen_updater) {
			exogen_updater->appendForecast(point_forecast, h);
		}
		if (error_updater) {
			error_updater->appendError(point_forecast);
		}
		pred_save.middleRows(h * num_row, num_row) = point_forecast;
	}

	virtual void computeMean() = 0;
};

class MarForecaster : public MatForecaster {
public:
	MarForecaster(
		int step, const Eigen::MatrixXd& y, int lag,
		const Eigen::MatrixXd& row_coef, const Eigen::MatrixXd& col_coef,
		BVHAR_OPTIONAL<std::unique_ptr<MatExogenForecaster>> exogen_forecaster = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<std::unique_ptr<MatErrorGenerator>> dgp_updater = BVHAR_NULLOPT
	)
	: MatForecaster(step, y, lag, row_coef, col_coef, std::move(exogen_forecaster), std::move(dgp_updater)) {}
	virtual ~MarForecaster() = default;

	Eigen::MatrixXd getLastForecast() override {
		return this->doForecast().bottomRows(num_row);
	}

protected:
	void computeMean() override {
		BVHAR_DEBUG_LOG(debug_logger, "computeMean() called");
		point_forecast.setZero();
	// #ifdef _OPENMP
	// 	#pragma omp parallel for reduction(+:point_forecast)
	// #endif
		for (int i = 0; i < lag; ++i) {
			point_forecast += row_coef.middleRows(i * num_row, num_row).transpose() * last_pvec.block(i * num_row, i * num_col, num_row, num_col) * col_coef.middleRows(i * num_col, num_col);
		}
	}
};

} // namespace baymar
} // namespace baecon

#endif // BAYMAR_OLS_FORECASTER_H