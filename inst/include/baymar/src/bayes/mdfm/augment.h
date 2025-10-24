#ifndef BAYMAR_BAYES_MDFM_AUGMENT_H
#define BAYMAR_BAYES_MDFM_AUGMENT_H

#include "./config.h"
// #include "../misc/draw.h"
// #include "../../math/design.h"

namespace baymar {

class MatAugmenter;
class MatFactorAugmenter;
class MatFactorVarAugmenter;

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
		BVHAR_BHRNG& rng
	) {}

	virtual void updateFactor(
		Eigen::Ref<const Eigen::MatrixXd> row_coef, Eigen::Ref<const Eigen::MatrixXd> row_sig_lower,
		Eigen::Ref<const Eigen::MatrixXd> col_coef, Eigen::Ref<const Eigen::MatrixXd> col_sig_lower,
		std::vector<Eigen::MatrixXd>& y,
		BVHAR_BHRNG& rng
	) {}

	virtual void updateRecords(int id) {}

	virtual void appendRecords(BVHAR_LIST& list) {}
};

class MatFactorAugmenter : public MatAugmenter {
public:
	MatFactorAugmenter(int num_iter, int num_design, const MatDfmParams& params)
	: need_restrict(false), num_iter(num_iter), nrow_factor(params._nrow_factor), ncol_factor(params._ncol_factor),
		size_factor(params._size_factor), lag(params._lag), num_design(num_design),
		resid(num_design), factor_mat(num_design) {
		mdfm_record = std::make_unique<MatDfmRecords>(num_iter, num_design, size_factor);
	}
	virtual ~MatFactorAugmenter() = default;

	bool NeedsRestrict() {
		return need_restrict;
	}

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
			// resid[i] = y[i] - row_coef.transpose() * x[i] * col_coef;
		}
	}

	void updateFactor(
		Eigen::Ref<const Eigen::MatrixXd> row_coef, Eigen::Ref<const Eigen::MatrixXd> row_sig_lower,
		Eigen::Ref<const Eigen::MatrixXd> col_coef, Eigen::Ref<const Eigen::MatrixXd> col_sig_lower,
		BVHAR_BHRNG& rng
	) override {
		draw_wn_factor(
			factor_mat, nrow_factor, ncol_factor,
			row_coef.transpose(), row_sig_lower,
			col_coef.transpose(), col_sig_lower,
			resid, rng
		);
	}

	void updateFactor(
		Eigen::Ref<const Eigen::MatrixXd> row_coef, Eigen::Ref<const Eigen::MatrixXd> row_sig_lower,
		Eigen::Ref<const Eigen::MatrixXd> col_coef, Eigen::Ref<const Eigen::MatrixXd> col_sig_lower,
		std::vector<Eigen::MatrixXd>& y,
		BVHAR_BHRNG& rng
	) override {
		draw_wn_factor(
			factor_mat, nrow_factor, ncol_factor,
			row_coef.transpose(), row_sig_lower,
			col_coef.transpose(), col_sig_lower,
			y, rng
		);
	}

	// template <bool isRow = true>
	// void updateCoefCov(
	// 	Eigen::Ref<Eigen::MatrixXd> coef, Eigen::Ref<Eigen::MatrixXd> sig_lower,
	// 	Eigen::Ref<const Eigen::MatrixXd> other_coef, Eigen::Ref<const Eigen::MatrixXd> other_sig_lower,
	// 	Eigen::Ref<const Eigen::MatrixXd> prior_mean, Eigen::Ref<const Eigen::VectorXd> prior_prec,
	// 	Eigen::Ref<const Eigen::MatrixXd> iw_scl,
	// 	double iw_df, int other_dim,
	// 	BVHAR_BHRNG& rng
	// ) {
	// 	int dim_factor = ncol_factor;
	// 	using is_row = std::integral_constant<bool, isRow>;
	// 	if (is_row::value) {
	// 		dim_factor = nrow_factor;
	// 	}
	// 	draw_coef_only<isRow, Eigen::MatrixXd>(
	// 		coef, sig_lower,
	// 		other_coef, other_sig_lower,
	// 		prior_mean, prior_prec, iw_scl, iw_df,
	// 		num_design, other_dim, dim_factor,
	// 		factor_mat, resid, rng
	// 	);
	// }

	template <bool isRow = true>
	void updateCoefCov(
		Eigen::Ref<Eigen::MatrixXd> coef, Eigen::Ref<Eigen::MatrixXd> sig_lower,
		Eigen::Ref<Eigen::MatrixXd> other_coef, Eigen::Ref<const Eigen::MatrixXd> other_sig_lower,
		Eigen::Ref<const Eigen::MatrixXd> prior_mean, Eigen::Ref<const Eigen::VectorXd> prior_prec,
		Eigen::Ref<const Eigen::MatrixXd> iw_scl,
		double iw_df, int other_dim,
		const std::vector<Eigen::MatrixXd>& y,
		BVHAR_BHRNG& rng
	) {
		int dim_factor = ncol_factor;
		using is_row = std::integral_constant<bool, isRow>;
		if (is_row::value) {
			dim_factor = nrow_factor;
		}
		draw_coef_sig<isRow, Eigen::MatrixXd>(
			coef, sig_lower,
			other_coef, other_sig_lower,
			prior_mean, prior_prec, iw_scl, iw_df,
			num_design, other_dim,
			0, 0, dim_factor, need_restrict,
			factor_mat, y, rng
		);
	}

	void updateRecords(int id) override {
		mdfm_record->assignRecords(
			id, factor_mat,
			num_design, size_factor
		);
	}

	void appendRecords(BVHAR_LIST& list) override {
		mdfm_record->appendRecords(list);
	}

	template <typename RecordType>
	RecordType returnStructRecords(int num_burn, int thin) const {
		return mdfm_record->returnRecords<RecordType>(num_iter, num_burn, thin);
	}

	const std::vector<Eigen::MatrixXd>& getFactor() const {
		return factor_mat;
	}
	
protected:
	bool need_restrict;
	int num_iter, nrow_factor, ncol_factor, size_factor, lag, num_design;
	std::vector<Eigen::MatrixXd> resid;
	std::vector<Eigen::MatrixXd> factor_mat; // F_{p + 1}, ..., F_t
	std::unique_ptr<MatDfmRecords> mdfm_record;
};

class MatFactorVarAugmenter : public MatFactorAugmenter {
public:
	MatFactorVarAugmenter(int num_iter, int num_design, const MatDfmVarParams& params, const MatDfmVarInits& inits)
	: MatFactorAugmenter(num_iter, num_design, params),
		dfm_coef(inits._init_factor_coef), dfm_sig(inits._init_factor_prec),
		ig_shp(params._sig_shp), ig_scl(params._sig_scl), prior_mean(params._mean), prior_prec(params._prec) {
		mdfm_record = std::make_unique<MatDfmVarRecords>(num_iter, num_design, size_factor, lag);
		need_restrict = true;
		// use ShrinkageUpdater for prior_prec later!
	}
	virtual ~MatFactorVarAugmenter() = default;
	
	void updateFactor(
		Eigen::Ref<const Eigen::MatrixXd> row_coef, Eigen::Ref<const Eigen::MatrixXd> row_sig_lower,
		Eigen::Ref<const Eigen::MatrixXd> col_coef, Eigen::Ref<const Eigen::MatrixXd> col_sig_lower,
		BVHAR_BHRNG& rng
	) override {
		draw_dfm_factor(
			factor_mat, lag, nrow_factor, ncol_factor, dfm_coef, dfm_sig,
			row_coef.transpose(), row_sig_lower,
			col_coef.transpose(), col_sig_lower,
			resid, rng
		);
		draw_dfm_sig(dfm_sig, lag, ig_shp, ig_scl, factor_mat, dfm_coef, rng);
		draw_dfm_coef(dfm_coef, dfm_sig, prior_mean, prior_prec, factor_mat, lag, rng);
	}

	void updateFactor(
		Eigen::Ref<const Eigen::MatrixXd> row_coef, Eigen::Ref<const Eigen::MatrixXd> row_sig_lower,
		Eigen::Ref<const Eigen::MatrixXd> col_coef, Eigen::Ref<const Eigen::MatrixXd> col_sig_lower,
		std::vector<Eigen::MatrixXd>& y,
		BVHAR_BHRNG& rng
	) override {
		draw_dfm_factor(
			factor_mat, lag, nrow_factor, ncol_factor, dfm_coef, dfm_sig,
			row_coef.transpose(), row_sig_lower,
			col_coef.transpose(), col_sig_lower,
			y, rng
		);
		draw_dfm_sig(dfm_sig, lag, ig_shp, ig_scl, factor_mat, dfm_coef, rng);
		draw_dfm_coef(dfm_coef, dfm_sig, prior_mean, prior_prec, factor_mat, lag, rng);
	}

	void updateRecords(int id) override {
		mdfm_record->assignRecords(
			id,
			factor_mat, dfm_coef, dfm_sig,
			num_design, size_factor
		);
	}

private:
	Eigen::MatrixXd dfm_coef; // p1*p2 x s
	Eigen::VectorXd dfm_sig; // lambda_{1, 1}, ..., lambda_{p1, p2}
	Eigen::VectorXd ig_shp, ig_scl;
	Eigen::VectorXd prior_mean, prior_prec;
};

inline std::unique_ptr<MatFactorAugmenter> initialize_factoraugmenter(
	int num_iter, int num_design,
	BVHAR_LIST& param_prior, BVHAR_LIST& param_init
) {
	std::unique_ptr<MatFactorAugmenter> augmenter_ptr;
	// MatDfmVarParams dfm_params(*factor_lag, *nrow_factor, *ncol_factor);
	// MatDfmVarInits dfm_inits((*nrow_factor) * (*ncol_factor), *factor_lag);
	int lag = BVHAR_CAST_INT(param_prior["lag"]);
	if (lag == 0) {
		MatDfmParams params(param_prior);
		augmenter_ptr = std::make_unique<MatFactorAugmenter>(num_iter, num_design, params);
	} else {
		MatDfmVarParams params(param_prior);
		MatDfmVarInits inits(param_init);
		augmenter_ptr = std::make_unique<MatFactorVarAugmenter>(num_iter, num_design, params, inits);
	}
	return augmenter_ptr;
}

} // namespace baymar

#endif // BAYMAR_BAYES_MDFM_AUGMENT_H