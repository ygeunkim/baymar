#ifndef BAYMAR_BAYES_MNIW_FORECASTER_H
#define BAYMAR_BAYES_MNIW_FORECASTER_H

#include "./mniw.h"
#include "../../math/design.h"

namespace baymar {

class MatMniwForecaster;
class MatMniwForecastRun;

class MatMniwForecaster : public bvhar::BayesForecaster<Eigen::MatrixXd, Eigen::MatrixXd> {
public:
	MatMniwForecaster(const MatMniwRecords& records, int step, const Eigen::MatrixXd& y, int num_data, int lag, unsigned int seed)
	: bvhar::BayesForecaster<Eigen::MatrixXd, Eigen::MatrixXd>(step, y, lag, records.row_coef_record.rows(), seed),
		mat_record(std::make_unique<MatMniwRecords>(records)),
		num_row(y.rows() / num_data), num_col(y.cols()), nrow_row_coef(num_row * lag), nrow_col_coef(num_col * lag),
		row_coef(Eigen::MatrixXd::Zero(nrow_row_coef, num_row)),
		row_sig_lower(Eigen::MatrixXd::Identity(num_row, num_row)),
		col_coef(Eigen::MatrixXd::Zero(nrow_col_coef, num_col)),
		col_sig_lower(Eigen::MatrixXd::Identity(num_col, num_col)),
		error_mat(Eigen::MatrixXd::Zero(num_row, num_col)) {
		initLagged();
	}
	virtual ~MatMniwForecaster() = default;
	
	Eigen::MatrixXd getLastForecast() override {
		return this->doForecast().bottomRows(num_row);
	}

	Eigen::MatrixXd getLastForecast(const Eigen::MatrixXd& valid_vec) override {
		return this->doForecast(valid_vec).bottomRows(num_row);
	}

protected:
	std::unique_ptr<MatMniwRecords> mat_record;
	int num_row, num_col, nrow_row_coef, nrow_col_coef;
	Eigen::MatrixXd row_coef, row_sig_lower, col_coef, col_sig_lower, error_mat;

	void initLagged() override {
		last_pvec = build_dense_design(response, lag);
		point_forecast = Eigen::MatrixXd::Zero(num_row, num_col);
		pred_save = Eigen::MatrixXd::Zero(step * num_row, num_sim * num_col);
		tmp_vec = last_pvec.block(num_row, num_col, num_row * (lag - 1), num_col * (lag - 1));
	}

	void initRecursion(const Eigen::MatrixXd& obs_vec) override {
		last_pvec = obs_vec;
		point_forecast = obs_vec.topLeftCorner(num_row, num_col);
		tmp_vec = obs_vec.bottomRightCorner(num_row * (lag - 1), num_col * (lag - 1));
	}

	void setRecursion() override {
		last_pvec.bottomRightCorner(num_row * (lag - 1), num_col * (lag - 1)) = tmp_vec;
		last_pvec.topLeftCorner(num_row, num_col) = point_forecast;
	}

	void updatePred(const int h, const int i) override {
		computeMean();
		updateVariance();
		// point_forecast += error_mat;
		pred_save.block(h * num_row, i * num_col, num_row, num_col) = point_forecast + error_mat;
	}

	void updateRecursion() override {
		tmp_vec = last_pvec.topLeftCorner(num_row * (lag - 1), num_col * (lag - 1));
	}

	void computeMean() {
		// point_forecast = row_coef.transpose() * last_pvec.sparseView() * col_coef;
		point_forecast.setZero();
	// #ifdef _OPENMP
	// 	#pragma omp parallel for reduction(+:point_forecast)
	// #endif
		for (int i = 0; i < lag; ++i) {
			point_forecast += row_coef.middleRows(i * num_row, num_row).transpose() * last_pvec.block(i * num_row, i * num_col, num_row, num_col) * col_coef.middleRows(i * num_col, num_col);
		}
	}

	void updateParams(const int i) override {
		row_coef = bvhar::unvectorize(mat_record->row_coef_record.row(i).transpose(), num_row);
		col_coef = bvhar::unvectorize(mat_record->col_coef_record.row(i).transpose(), num_col);
		fill_lower(row_sig_lower, mat_record->row_sigma_record.row(i).transpose());
		fill_lower(col_sig_lower, mat_record->col_sigma_record.row(i).transpose());
	}

	void updateVariance() {
		for (int j = 0; j < num_col; ++j) {
			for (int i = 0; i < num_row; ++i) {
				error_mat(i, j) = bvhar::normal_rand(rng);
			}
		}
		error_mat = row_sig_lower * error_mat * col_sig_lower.transpose();
	}

	void updateLpl(int h, const Eigen::MatrixXd& valid_vec) override {}
};

inline std::vector<std::unique_ptr<MatMniwForecaster>> initialize_matmniwforecaster(
	int num_chains, int lag, int step, const Eigen::MatrixXd& y, int num_data,
	LIST& fit_record, Eigen::Ref<const Eigen::VectorXi> seed_chain, int nthreads
) {
	PY_LIST row_coef_record = fit_record["A_record"];
	PY_LIST row_sigma_record = fit_record["SigmaR_record"];
	PY_LIST col_coef_record = fit_record["B_record"];
	PY_LIST col_sigma_record = fit_record["SigmaC_record"];
	std::vector<std::unique_ptr<MatMniwForecaster>> forecaster(num_chains);
	for (int i = 0; i < num_chains; ++i) {
		MatMniwRecords mat_record(
			CAST<Eigen::MatrixXd>(row_coef_record[i]),
			CAST<Eigen::MatrixXd>(row_sigma_record[i]),
			CAST<Eigen::MatrixXd>(col_coef_record[i]),
			CAST<Eigen::MatrixXd>(col_sigma_record[i])
		);
		forecaster[i] = std::make_unique<MatMniwForecaster>(mat_record, step, y, num_data, lag, static_cast<unsigned int>(seed_chain[i]));
	}
	return forecaster;
}

class MatMniwForecastRun : public bvhar::McmcForecastRun<Eigen::MatrixXd, Eigen::MatrixXd> {
public:
	MatMniwForecastRun(
		int num_chains, int lag, int step, const Eigen::MatrixXd& y, int num_data,
		LIST& fit_record, const Eigen::VectorXi& seed_chain, int nthreads
	)
	: bvhar::McmcForecastRun<Eigen::MatrixXd, Eigen::MatrixXd>(num_chains, lag, step, nthreads) {
		auto temp_forecaster = initialize_matmniwforecaster(num_chains, lag, step, y, num_data, fit_record, seed_chain, nthreads);
		for (int i = 0; i < num_chains; ++i) {
			forecaster[i] = std::move(temp_forecaster[i]);
		}
	}
	virtual ~MatMniwForecastRun() = default;
};

} // namespace baymar

#endif // BAYMAR_BAYES_MNIW_FORECASTER_H