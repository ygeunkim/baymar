#ifndef BAYMAR_BAYES_MDFM_MDFM_H
#define BAYMAR_BAYES_MDFM_MDFM_H

// #include <bvhar/base>
#include "./config.h"

namespace baymar {

class McmcMatAugment;
class McmcMatDfm;

class McmcMatAugment {
public:
	McmcMatAugment() {}
	virtual ~McmcMatAugment() = default;

	virtual void updateResid(
		std::vector<Eigen::SparseMatrix<double>>& x, std::vector<Eigen::MatrixXd>& y,
		Eigen::Ref<Eigen::MatrixXd> row_coef, Eigen::Ref<Eigen::MatrixXd> col_coef
	) {}
	
	virtual void updateFactor(
		Eigen::Ref<Eigen::MatrixXd> row_coef, Eigen::Ref<Eigen::MatrixXd> row_sig_lower,
		Eigen::Ref<Eigen::MatrixXd> col_coef, Eigen::Ref<Eigen::MatrixXd> col_sig_lower,
		BHRNG& rng
	) {}
};

class McmcMatDfm : public McmcMatAugment {
public:
	McmcMatDfm(int num_design, int lag, int nrow_factor, int ncol_factor)
	: nrow_factor(nrow_factor), ncol_factor(ncol_factor),
		size_factor(nrow_factor * ncol_factor), lag(lag), num_design(num_design),
		resid(num_design), factor_mat(num_design),
		dfm_coef(Eigen::MatrixXd::Zero(size_factor, lag)),
		dfm_prec(Eigen::VectorXd::Ones(size_factor)),
		ig_shp(Eigen::VectorXd::Constant(3.0, size_factor)), ig_scl(Eigen::VectorXd::Ones(size_factor)),
		prior_mean(Eigen::VectorXd::Zero(size_factor)), prior_prec(Eigen::VectorXd::Ones(size_factor)) {
		// updateResid(x, y, row_coef, col_coef);
	}
	virtual ~McmcMatDfm() = default;

	void updateResid(
		std::vector<Eigen::SparseMatrix<double>>& x, std::vector<Eigen::MatrixXd>& y,
		Eigen::Ref<Eigen::MatrixXd> row_coef, Eigen::Ref<Eigen::MatrixXd> col_coef
	) override {
		for (int i = 0; i < num_design; ++i) {
			resid[i] = y[i] - row_coef.transpose() * x[i] * col_coef;
		}
	}
	
	void updateFactor(
		Eigen::Ref<Eigen::MatrixXd> row_coef, Eigen::Ref<Eigen::MatrixXd> row_sig_lower,
		Eigen::Ref<Eigen::MatrixXd> col_coef, Eigen::Ref<Eigen::MatrixXd> col_sig_lower,
		BHRNG& rng
	) override {
		draw_dfm_factor(
			factor_mat, lag, nrow_factor, ncol_factor, dfm_coef, dfm_prec,
			row_coef, row_sig_lower, col_coef, col_sig_lower,
			resid, rng
		);
		draw_dfm_prec(dfm_prec, lag, ig_shp, ig_scl, factor_mat, dfm_coef, rng);
		draw_dfm_coef(dfm_coef, dfm_prec, prior_mean, prior_prec, factor_mat, lag, rng);
	}
	
protected:
	int nrow_factor, ncol_factor, size_factor, lag, num_design;
	std::vector<Eigen::MatrixXd> resid;
	std::vector<Eigen::MatrixXd> factor_mat; // F_{p + 1}, ..., F_t
	Eigen::MatrixXd dfm_coef; // p1*p2 x s
	Eigen::VectorXd dfm_prec; // lambda_{1, 1}, ..., lambda_{p1, p2}
	Eigen::VectorXd ig_shp, ig_scl;
	Eigen::VectorXd prior_mean, prior_prec;
};

} // namespace baymar

#endif // BAYMAR_BAYES_MDFM_MDFM_H