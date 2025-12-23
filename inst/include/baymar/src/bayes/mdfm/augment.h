#ifndef BAYMAR_BAYES_MDFM_AUGMENT_H
#define BAYMAR_BAYES_MDFM_AUGMENT_H

#include "./config.h"
// #include "../shrinkage/shrinkage.h"
// #include "../misc/draw.h"
// #include "../../math/design.h"

namespace baecon {
namespace baymar {

class MatAugmenter;
class MatFactorAugmenter;
class MatFactorVarAugmenter;
class MatFactorMarAugmenter;

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
			// resid[i] = y[i] - row_coef.transpose() * x[i].topLeftCorner(x[i].rows() - nrow_factor, x[i].cols() - ncol_factor) * col_coef;
			resid[i] = y[i] - row_coef.transpose() * x[i] * col_coef;
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

	template <bool isRow = true>
	void updateCoefCov(
		Eigen::Ref<Eigen::MatrixXd> coef, Eigen::Ref<Eigen::MatrixXd> sig_lower,
		Eigen::Ref<const Eigen::MatrixXd> other_coef, Eigen::Ref<const Eigen::MatrixXd> other_sig_lower,
		Eigen::Ref<const Eigen::MatrixXd> prior_mean, Eigen::Ref<const Eigen::VectorXd> prior_prec,
		Eigen::Ref<const Eigen::MatrixXd> iw_scl,
		double iw_df, int other_dim,
		BVHAR_BHRNG& rng
	) {
		int dim_factor = ncol_factor;
		using is_row = std::integral_constant<bool, isRow>;
		if (is_row::value) {
			dim_factor = nrow_factor;
		}
		draw_coef_only<isRow, Eigen::MatrixXd>(
			coef, sig_lower,
			other_coef, other_sig_lower,
			prior_mean, prior_prec, iw_scl, iw_df,
			num_design, other_dim, dim_factor,
			factor_mat, resid, rng
		);
	}

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

class MatFactorMarAugmenter : public MatFactorAugmenter {
public:
	MatFactorMarAugmenter(int num_iter, int num_design, const MatDfmParams& params)
	: MatFactorAugmenter(num_iter, num_design, params),
		fac_nrow_row_coef(nrow_factor * lag), fac_nrow_col_coef(ncol_factor * lag),
		fac_row_coef(Eigen::MatrixXd::Identity(fac_nrow_row_coef, nrow_factor)),
		fac_row_sig_lower(Eigen::MatrixXd::Identity(nrow_factor, nrow_factor)),
		fac_col_coef(Eigen::MatrixXd::Identity(fac_nrow_col_coef, ncol_factor)),
		fac_col_sig_lower(Eigen::MatrixXd::Identity(ncol_factor, ncol_factor)),
		fac_row_mean(Eigen::MatrixXd::Zero(fac_nrow_row_coef, nrow_factor)),
		fac_row_iw(Eigen::MatrixXd::Identity(nrow_factor, nrow_factor)),
		fac_col_mean(Eigen::MatrixXd::Zero(fac_nrow_col_coef, ncol_factor)),
		fac_col_iw(Eigen::MatrixXd::Identity(ncol_factor, ncol_factor)),
		fac_row_prec(Eigen::VectorXd::Ones(fac_nrow_row_coef)),
		fac_col_prec(Eigen::VectorXd::Ones(fac_nrow_col_coef)),
		fac_row_df(nrow_factor + 2), fac_col_df(ncol_factor + 2) {
		mdfm_record = std::make_unique<MatDfmMarRecords>(num_iter, num_design, nrow_factor, ncol_factor, lag);
		need_restrict = true;
		// Temporarily initialize Horseshoe updater inside constructor
		// -> Should move into initialize_factoraugmenter later to choose priors
		MatShrinkageParams shrinkage_param;
		MatGlInits row_shrinkage_inits(fac_nrow_row_coef);
		MatGlInits col_shrinkage_inits(fac_nrow_col_coef);
		row_updater = std::make_unique<MatHsUpdater>(num_iter, shrinkage_param, row_shrinkage_inits);
		col_updater = std::make_unique<MatHsUpdater>(num_iter, shrinkage_param, fac_nrow_col_coef);
	}

	// Should use this later
	MatFactorMarAugmenter(
		int num_iter, int num_design,
		const MatDfmMarParams& params, const MatDfmMarInits& inits,
		std::unique_ptr<MatShrinkageUpdater>& row_updater, std::unique_ptr<MatShrinkageUpdater>& col_updater
	)
	: MatFactorAugmenter(num_iter, num_design, params),
		fac_nrow_row_coef(nrow_factor * lag), fac_nrow_col_coef(ncol_factor * lag),
		fac_row_coef(inits.mniw_init._init_row_coef), fac_row_sig_lower(inits.mniw_init._init_row_lower),
		fac_col_coef(inits.mniw_init._init_col_coef), fac_col_sig_lower(inits.mniw_init._init_col_lower),
		fac_row_mean(params.mniw_params._row_mean), fac_row_iw(params.mniw_params._row_iw_scl),
		fac_col_mean(params.mniw_params._col_mean), fac_col_iw(params.mniw_params._col_iw_scl),
		fac_row_prec(params.mniw_params._row_prec), fac_col_prec(params.mniw_params._col_prec),
		fac_row_df(params.mniw_params._row_iw_df), fac_col_df(params.mniw_params._col_iw_df),
		row_updater(std::move(row_updater)), col_updater(std::move(col_updater)) {
		mdfm_record = std::make_unique<MatDfmMarRecords>(num_iter, num_design, nrow_factor, ncol_factor, lag);
		need_restrict = true;
	}
	
	virtual ~MatFactorMarAugmenter() = default;

	void updateFactor(
		Eigen::Ref<const Eigen::MatrixXd> row_coef, Eigen::Ref<const Eigen::MatrixXd> row_sig_lower,
		Eigen::Ref<const Eigen::MatrixXd> col_coef, Eigen::Ref<const Eigen::MatrixXd> col_sig_lower,
		BVHAR_BHRNG& rng
	) override {
		row_updater->updatePrec(
			fac_row_prec,
			fac_row_coef, fac_row_sig_lower,
			fac_row_mean,
			rng
		);
		col_updater->updatePrec(
			fac_col_prec,
			fac_col_coef, fac_col_sig_lower,
			fac_col_mean,
			rng
		);
		draw_mar_factor(
			factor_mat, lag, nrow_factor, ncol_factor,
			fac_row_coef, fac_row_sig_lower,
			fac_col_coef, fac_col_sig_lower,
			row_coef.transpose(), row_sig_lower,
			col_coef.transpose(), col_sig_lower,
			resid, rng
		);
		// factor_mat: p + 1, ..., T
		// F_t = A^T diag(F_{t - 1}, ..., F_{t - s}) B + V_t, t = p + 1 + s, ..., T
		std::vector<Eigen::MatrixXd> response = build_mar_response(factor_mat, lag);
		std::vector<Eigen::SparseMatrix<double>> factor_design = build_mar_design(factor_mat, lag);
		draw_coef_sig<true>(
			fac_row_coef, fac_row_sig_lower,
			fac_col_coef, fac_col_sig_lower,
			fac_row_mean, fac_row_prec, fac_row_iw, fac_row_df,
			num_design - lag, ncol_factor,
			0, 0, 0, false,
			factor_design, response, rng
		);
		draw_coef_sig<false>(
			fac_col_coef, fac_col_sig_lower,
			fac_row_coef, fac_row_sig_lower,
			fac_col_mean, fac_col_prec, fac_col_iw, fac_col_df,
			num_design - lag, nrow_factor,
			0, 0, 0, false,
			factor_design, response, rng
		);
	}

	void updateFactor(
		Eigen::Ref<const Eigen::MatrixXd> row_coef, Eigen::Ref<const Eigen::MatrixXd> row_sig_lower,
		Eigen::Ref<const Eigen::MatrixXd> col_coef, Eigen::Ref<const Eigen::MatrixXd> col_sig_lower,
		std::vector<Eigen::MatrixXd>& y,
		BVHAR_BHRNG& rng
	) override {
		row_updater->updatePrec(
			fac_row_prec,
			fac_row_coef, fac_row_sig_lower,
			fac_row_mean,
			rng
		);
		col_updater->updatePrec(
			fac_col_prec,
			fac_col_coef, fac_col_sig_lower,
			fac_col_mean,
			rng
		);
		draw_mar_factor(
			factor_mat, lag, nrow_factor, ncol_factor,
			fac_row_coef, fac_row_sig_lower,
			fac_col_coef, fac_col_sig_lower,
			row_coef.transpose(), row_sig_lower,
			col_coef.transpose(), col_sig_lower,
			y, rng
		);
		std::vector<Eigen::MatrixXd> response = build_mar_response(factor_mat, lag);
		std::vector<Eigen::SparseMatrix<double>> factor_design = build_mar_design(factor_mat, lag);
		draw_coef_sig<true>(
			fac_row_coef, fac_row_sig_lower,
			fac_col_coef, fac_col_sig_lower,
			fac_row_mean, fac_row_prec, fac_row_iw, fac_row_df,
			num_design - lag, ncol_factor,
			0, 0, 0, false,
			factor_design, response, rng
		);
		draw_coef_sig<false>(
			fac_col_coef, fac_col_sig_lower,
			fac_row_coef, fac_row_sig_lower,
			fac_col_mean, fac_col_prec, fac_col_iw, fac_col_df,
			num_design - lag, nrow_factor,
			0, 0, 0, false,
			factor_design, response, rng
		);
	}

	void updateRecords(int id) override {
		mdfm_record->assignRecords(
			id,
			factor_mat,
			fac_row_coef, fac_row_sig_lower,
			fac_col_coef, fac_col_sig_lower,
			num_design,
			fac_nrow_row_coef, nrow_factor,
			fac_nrow_col_coef, ncol_factor
		);
	}

private:
	int fac_nrow_row_coef, fac_nrow_col_coef;
	Eigen::MatrixXd fac_row_coef, fac_row_sig_lower, fac_col_coef, fac_col_sig_lower;
	Eigen::MatrixXd fac_row_mean, fac_row_iw, fac_col_mean, fac_col_iw;
	Eigen::VectorXd fac_row_prec, fac_col_prec;
	double fac_row_df, fac_col_df;
	std::unique_ptr<MatShrinkageUpdater> row_updater;
	std::unique_ptr<MatShrinkageUpdater> col_updater;
};

inline std::unique_ptr<MatFactorAugmenter> initialize_factoraugmenter(
	int num_iter, int num_design,
	BVHAR_LIST& param_prior, BVHAR_LIST& param_init,
	BVHAR_OPTIONAL<BVHAR_LIST> row_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST> row_init_spec = BVHAR_NULLOPT,
	BVHAR_OPTIONAL<BVHAR_LIST> col_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST> col_init_spec = BVHAR_NULLOPT
) {
	std::unique_ptr<MatFactorAugmenter> augmenter_ptr;
	// // MatDfmVarParams dfm_params(*factor_lag, *nrow_factor, *ncol_factor);
	// // MatDfmVarInits dfm_inits((*nrow_factor) * (*ncol_factor), *factor_lag);
	// int lag = BVHAR_CAST_INT(param_prior["lag"]);
	// if (lag == 0) {
	// 	MatDfmParams params(param_prior);
	// 	augmenter_ptr = std::make_unique<MatFactorAugmenter>(num_iter, num_design, params);
	// } else if (false) {
	// 	MatDfmParams params(param_prior);
	// 	// MatDfmMarParams params(param_prior);
	// 	// MatDfmMarInits inits(param_init);
	// 	augmenter_ptr = std::make_unique<MatFactorMarAugmenter>(num_iter, num_design, params);
	// } else {
	// 	MatDfmVarParams params(param_prior);
	// 	MatDfmVarInits inits(param_init);
	// 	augmenter_ptr = std::make_unique<MatFactorVarAugmenter>(num_iter, num_design, params, inits);
	// }
	BVHAR_STRING factor_model_nm = BVHAR_CAST<BVHAR_STRING>(param_prior["factor_type"]);
	int factor_type = 0;
	if (factor_model_nm == "wn") {
		factor_type = 1;
	} else if (factor_model_nm == "var") {
		factor_type = 2;
	} else if (factor_model_nm == "mar") {
		factor_type = 3;
	}
	switch (factor_type) {
		case 1: {
			MatDfmParams params(param_prior);
			augmenter_ptr = std::make_unique<MatFactorAugmenter>(num_iter, num_design, params);
			return augmenter_ptr;
		}
		case 2: {
			MatDfmVarParams params(param_prior);
			MatDfmVarInits inits(param_init);
			augmenter_ptr = std::make_unique<MatFactorVarAugmenter>(num_iter, num_design, params, inits);
			return augmenter_ptr;
		}
		case 3: {
			MatDfmMarParams params(param_prior);
			MatDfmMarInits inits(param_init);
			// int row_prior_type = BVHAR_CAST_INT(row_prior["factor_type"]);
			// int col_prior_type = BVHAR_CAST_INT(col_prior["factor_type"]);
			// Or *_prior_type = 0 and choose inside initialize_matshrinkageupdater
			auto row_updater = initialize_matshrinkageupdater(num_iter, *row_prior, *row_init_spec, 0, "factor_");
			auto col_updater = initialize_matshrinkageupdater(num_iter, *col_prior, *col_init_spec, 0, "factor_");
			augmenter_ptr = std::make_unique<MatFactorMarAugmenter>(
				num_iter, num_design,
				params, inits,
				row_updater, col_updater
			);
			// MatDfmParams params(param_prior);
			// augmenter_ptr = std::make_unique<MatFactorMarAugmenter>(num_iter, num_design, params);
			return augmenter_ptr;
		}
		default: {
			BVHAR_STOP("Not defined.");
		}
	}
	return augmenter_ptr;
}

} // namespace baymar
} // namespace baecon

#endif // BAYMAR_BAYES_MDFM_AUGMENT_H