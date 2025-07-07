#ifndef BAYMAR_BAYES_MDFM_AUGMENT_H
#define BAYMAR_BAYES_MDFM_AUGMENT_H

// #include "./config.h"
#include "../misc/draw.h"
#include "../../math/design.h"

namespace baymar {

class MatAugmenter;
class MatFactorAugmenter;

class MatAugmenter {
public:
	MatAugmenter() {}
	virtual ~MatAugmenter() = default;

	virtual void appendDesign(std::vector<Eigen::SparseMatrix<double>>& x) {}

	virtual void updateResid(
		std::vector<Eigen::SparseMatrix<double>>& x, std::vector<Eigen::MatrixXd>& y,
		Eigen::Ref<const Eigen::MatrixXd> row_coef, Eigen::Ref<const Eigen::MatrixXd> col_coef
	) {}
	
	virtual void updateFactor(
		Eigen::Ref<const Eigen::MatrixXd> row_coef, Eigen::Ref<const Eigen::MatrixXd> row_sig_lower,
		Eigen::Ref<const Eigen::MatrixXd> col_coef, Eigen::Ref<const Eigen::MatrixXd> col_sig_lower,
		BHRNG& rng
	) {}

	virtual void updateRecords(int id) {}

	virtual void appendRecords(LIST& list) {}
};

class MatFactorAugmenter : public MatAugmenter {
public:
	MatFactorAugmenter(int num_iter, int num_design, int lag, int nrow_factor, int ncol_factor)
	: nrow_factor(nrow_factor), ncol_factor(ncol_factor),
		size_factor(nrow_factor * ncol_factor), lag(lag), num_design(num_design),
		resid(num_design), factor_mat(num_design),
		dfm_coef(Eigen::MatrixXd::Zero(size_factor, lag)),
		dfm_prec(Eigen::VectorXd::Ones(size_factor)),
		ig_shp(Eigen::VectorXd::Constant(size_factor, 3.0)), ig_scl(Eigen::VectorXd::Ones(size_factor)),
		prior_mean(Eigen::VectorXd::Zero(lag)), prior_prec(Eigen::VectorXd::Ones(lag)),
		factor_record(Eigen::MatrixXd::Zero(num_iter + 1, num_design * size_factor)),
		coef_record(Eigen::MatrixXd::Zero(num_iter + 1, size_factor * lag)),
		prec_record(Eigen::MatrixXd::Zero(num_iter + 1, size_factor)) {
		// use ShrinkageUpdater for prior_prec later!
	}
	virtual ~MatFactorAugmenter() = default;

	void appendDesign(std::vector<Eigen::SparseMatrix<double>>& x) override {
		// diag(Y_{t - 1}, ..., Y_{t - p}, X_t, ..., X_{t - s}, F_t)
		for (int i = 0; i < num_design; ++i) {
			append_x(x[i], factor_mat[i], nrow_factor, ncol_factor);
		}
	}

	void updateResid(
		std::vector<Eigen::SparseMatrix<double>>& x, std::vector<Eigen::MatrixXd>& y,
		Eigen::Ref<const Eigen::MatrixXd> row_coef, Eigen::Ref<const Eigen::MatrixXd> col_coef
	) override {
		for (int i = 0; i < num_design; ++i) {
			// resid[i] = y[i] - row_coef.topRows(row_coef.rows() - nrow_factor).transpose() * x[i].topLeftCorner(x[i].rows() - nrow_factor, x[i].cols() - ncol_factor) * col_coef.topRows(col_coef.rows() - ncol_factor);
			resid[i] = y[i] - row_coef.transpose() * x[i].topLeftCorner(x[i].rows() - nrow_factor, x[i].cols() - ncol_factor) * col_coef;
		}
	}
	
	void updateFactor(
		Eigen::Ref<const Eigen::MatrixXd> row_coef, Eigen::Ref<const Eigen::MatrixXd> row_sig_lower,
		Eigen::Ref<const Eigen::MatrixXd> col_coef, Eigen::Ref<const Eigen::MatrixXd> col_sig_lower,
		BHRNG& rng
	) override {
		draw_dfm_factor(
			factor_mat, lag, nrow_factor, ncol_factor, dfm_coef, dfm_prec,
			row_coef, row_sig_lower,
			col_coef, col_sig_lower,
			resid, rng
		);
		draw_dfm_prec(dfm_prec, lag, ig_shp, ig_scl, factor_mat, dfm_coef, rng);
		draw_dfm_coef(dfm_coef, dfm_prec, prior_mean, prior_prec, factor_mat, lag, rng);
	}

	void updateRecords(int id) override {
		for (int i = 0; i < num_design; ++i) {
			// f_{11, p + 1}, f_{21, p + 1}, ..., f_{p1p2, p + 1}, f_{11, p + 2}, ..., f_{p1p2, T}
			factor_record.row(id).segment(i * size_factor, size_factor) = factor_mat[i].reshaped();
		}
		coef_record.row(id) = dfm_coef.reshaped();
		prec_record.row(id) = dfm_prec;
	}

	void appendRecords(LIST& list) override {
		list["F_record"] = factor_record;
		list["Rho_record"] = coef_record;
		list["Lambda_record"] = prec_record;
	}
	
protected:
	int nrow_factor, ncol_factor, size_factor, lag, num_design;
	std::vector<Eigen::MatrixXd> resid;
	std::vector<Eigen::MatrixXd> factor_mat; // F_{p + 1}, ..., F_t
	Eigen::MatrixXd dfm_coef; // p1*p2 x s
	Eigen::VectorXd dfm_prec; // lambda_{1, 1}, ..., lambda_{p1, p2}
	Eigen::VectorXd ig_shp, ig_scl;
	Eigen::VectorXd prior_mean, prior_prec;
	Eigen::MatrixXd factor_record, coef_record, prec_record;
};

} // namespace baymar

#endif // BAYMAR_BAYES_MDFM_AUGMENT_H