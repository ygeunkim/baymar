#ifndef BAYMAR_BAYES_MDFM_MDFM_H
#define BAYMAR_BAYES_MDFM_MDFM_H

#include "./config.h"
// #include "../shrinkage/shrinkage.h"

namespace baymar {

class McmcMatDfm;
class McmcMatDfmVar;
template <typename BaseDfm> class MatDfmRun;

class McmcMatDfm : public bvhar::McmcAlgo {
public:
	McmcMatDfm(
		const MatDfmParams& params, const MatMniwInits& inits,
		std::unique_ptr<MatShrinkageUpdater>& row_updater, std::unique_ptr<MatShrinkageUpdater>& col_updater,
		unsigned int seed
	)
	: bvhar::McmcAlgo(params, seed),
		nrow_factor(params._nrow_factor), ncol_factor(params._ncol_factor), size_factor(params._size_factor),
		lag(params._lag), num_design(params._design),
		num_row(params._row), num_col(params._col),
		nrow_row_coef(params._row_row_coef), nrow_col_coef(params._row_col_coef),
		y(params._y), /*x(num_design),*/ factor_mat(num_design),
		row_updater(std::move(row_updater)), col_updater(std::move(col_updater)),
		row_coef(inits._init_row_coef), row_sig_lower(inits._init_row_lower),
		col_coef(inits._init_col_coef), col_sig_lower(inits._init_col_lower),
		// mdfm_record(std::make_unique<MatDfmRecords>(num_iter, num_row, num_col, nrow_row_coef, nrow_col_coef, num_design, size_factor)),
		row_prior_mean(params._row_mean), row_iw_scl(params._row_iw_scl),
		col_prior_mean(params._col_mean), col_iw_scl(params._col_iw_scl),
		row_prior_prec(params._row_prec), col_prior_prec(params._col_prec),
		row_iw_df(params._row_iw_df), col_iw_df(params._col_iw_df) {
		BVHAR_DEBUG_LOG(
			debug_logger,
			"McmcMatDfm Constructor: row_coef: {} x {}, row_sig_lower: {} x {}, row_prior_mean: {} x {}, row_prior_prec: {}",
			row_coef.rows(), row_coef.cols(), row_sig_lower.rows(), row_sig_lower.cols(),
			row_prior_mean.rows(), row_prior_mean.cols(), row_prior_prec.size()
		);
		BVHAR_DEBUG_LOG(
			debug_logger,
			"McmcMatDfm Constructor: col_coef: {} x {}, col_sig_lower: {} x {}, col_prior_mean: {} x {}, col_prior_prec: {}",
			col_coef.rows(), col_coef.cols(), col_sig_lower.rows(), col_sig_lower.cols(),
			col_prior_mean.rows(), col_prior_mean.cols(), col_prior_prec.size()
		);
		BVHAR_DEBUG_LOG(
			debug_logger,
			"McmcMatDfm Constructor: nrow_factor: {}, ncol_factor: {}, lag: {}, num_design: {}",
			nrow_factor, ncol_factor, lag, num_design
		);
	}
	virtual ~McmcMatDfm() = default;

	void doWarmUp() override {
		BVHAR_DEBUG_LOG(debug_logger, "doWarmUp() called");
		std::lock_guard<std::mutex> lock(mtx);
		updatePrec();
		updateFactor();
		// updateDesign();
		updateCoefCov();
	}

	void doPosteriorDraws() override {
		BVHAR_DEBUG_LOG(debug_logger, "doPosteriorDraws() called");
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updatePrec();
		updateFactor();
		// updateDesign();
		updateCoefCov();
		updateRecords();
	}

	LIST returnRecords(int num_burn, int thin) override {
		BVHAR_DEBUG_LOG(debug_logger, "returnRecords(num_burn={}, thin={}) called", num_burn, thin);
		LIST res = mdfm_record->returnListRecords(nrow_row_coef, num_row, nrow_col_coef, num_col, num_design, size_factor);
		mdfm_record->appendRecords(res);
		for (auto& record : res) {
			if (IS_MATRIX(ACCESS_LIST(record, res))) {
				ACCESS_LIST(record, res) = bvhar::thin_record(CAST<Eigen::MatrixXd>(ACCESS_LIST(record, res)), num_iter, num_burn, thin);
			} else {
				ACCESS_LIST(record, res) = bvhar::thin_record(CAST<Eigen::VectorXd>(ACCESS_LIST(record, res)), num_iter, num_burn, thin);
			}
		}
		return res;
	}

	template <typename RecordType>
	RecordType returnStructRecords(int num_burn, int thin) const {
		return mdfm_record->returnRecords<RecordType>(num_iter, num_burn, thin);
	}

protected:
	int nrow_factor, ncol_factor, size_factor, lag, num_design;
	int num_row, num_col, nrow_row_coef, nrow_col_coef;
	std::vector<Eigen::MatrixXd> y;
	// std::vector<Eigen::SparseMatrix<double>> x;
	std::vector<Eigen::MatrixXd> factor_mat;
	std::unique_ptr<MatShrinkageUpdater> row_updater;
	std::unique_ptr<MatShrinkageUpdater> col_updater;
	Eigen::MatrixXd row_coef, row_sig_lower, col_coef, col_sig_lower;
	std::unique_ptr<MatDfmRecords> mdfm_record;
	Eigen::MatrixXd row_prior_mean, row_iw_scl;
	Eigen::MatrixXd col_prior_mean, col_iw_scl;
	Eigen::VectorXd row_prior_prec, col_prior_prec;
	double row_iw_df, col_iw_df;

	virtual void updateFactor() = 0;
	
	void updatePrec() {
		BVHAR_DEBUG_LOG(debug_logger, "updatePrec() called");
		row_updater->updatePrec(
			row_prior_prec,
			row_coef,
			row_sig_lower,
			row_prior_mean,
			rng
		);
		col_updater->updatePrec(
			col_prior_prec,
			col_coef,
			col_sig_lower,
			col_prior_mean,
			rng
		);
	}

	void updateCoefCov() {
		BVHAR_DEBUG_LOG(debug_logger, "updateCoefCov() called");
		draw_coef_sig<true, Eigen::MatrixXd>(
			row_coef, row_sig_lower,
			col_coef, col_sig_lower,
			row_prior_mean, row_prior_prec, row_iw_scl, row_iw_df,
			num_design, num_col,
			factor_mat, y, rng
		);
		draw_coef_sig<false, Eigen::MatrixXd>(
			col_coef, col_sig_lower,
			row_coef, row_sig_lower,
			col_prior_mean, col_prior_prec, col_iw_scl, col_iw_df,
			num_design, num_row,
			factor_mat, y, rng
		);
	}

	virtual void updateRecords() = 0;
};

class McmcMatDfmVar : public McmcMatDfm {
public:
	McmcMatDfmVar(
		const MatDfmVarParams& params, const MatDfmVarInits& inits,
		std::unique_ptr<MatShrinkageUpdater>& row_updater, std::unique_ptr<MatShrinkageUpdater>& col_updater,
		unsigned int seed
	)
	: McmcMatDfm(params, inits, row_updater, col_updater, seed),
		ig_shp(params._sig_shp), ig_scl(params._sig_scl),
		prior_mean(params._mean), prior_prec(params._prec),
		factor_coef(inits._init_factor_coef), factor_prec(inits._init_factor_prec) {
		BVHAR_DEBUG_LOG(
			debug_logger,
			"McmcMatDfmVar Constructor: factor_coef: {} x {}, factor_prec: {}, ig_shp: {}, ig_scl: {}, prior_mean: {}, prior_prec: {}",
			factor_coef.rows(), factor_coef.cols(), factor_prec.size(),
			ig_shp.size(), ig_scl.size(),
			prior_mean.size(), prior_prec.size()
		);
		mdfm_record = std::make_unique<MatDfmVarRecords>(num_iter, num_row, num_col, nrow_row_coef, nrow_col_coef, num_design, size_factor, lag);
		// To check Gibbs sampler
		// factor_mat[0] = Eigen::MatrixXd::Zero(3, 2);
		// factor_mat[0] << -1.5510253, -1.506134,
		// 						     0.6051438, -1.813521,
		// 						     1.1589654, 3.320811;
		// factor_mat[1] = Eigen::MatrixXd::Zero(3, 2);
		// factor_mat[1] << -0.4425465, -0.351861,
		// 						     0.3299830, -2.060215,
		// 						     1.1378966, 2.881387;
		// factor_mat[2] = Eigen::MatrixXd::Zero(3, 2);
		// factor_mat[2] << -1.69158780, 1.217798,
		// 						     2.37891583, -2.927637,
		// 						     0.07124717, 1.626212;
		// factor_mat[3] = Eigen::MatrixXd::Zero(3, 2);
		// factor_mat[3] << -1.7256651, 0.2030085,
		// 						     0.5475849, -2.2477632,
		// 						     1.8368687, 1.4821128;
		// factor_mat[4] = Eigen::MatrixXd::Zero(3, 2);
		// factor_mat[4] << 0.7092791, -1.0445764,
    //                  0.4568669, -0.9046775,
    //                  0.9205983, 1.0211350;
		// factor_mat[5] = Eigen::MatrixXd::Zero(3, 2);
		// factor_mat[5] << -0.4844620, -2.01179324,
    //                  1.4164791, -0.01378793,
    //                  0.4090317, 1.26876682;
		// factor_mat[6] = Eigen::MatrixXd::Zero(3, 2);
		// factor_mat[6] << -0.4677551, -2.318547,
    //                  0.2080809, -1.018533,
    //                  2.0374795, 1.029340;
		// factor_mat[7] = Eigen::MatrixXd::Zero(3, 2);
		// factor_mat[7] << 1.2482667, -2.4525043,
    //                  0.6804626, 0.8428475,
    //                  0.5123235, -0.2734674;
		// factor_mat[8] = Eigen::MatrixXd::Zero(3, 2);
		// factor_mat[8] << 2.29187467, -1.7333012,
    //                  0.02751096, 0.9484436,
    //                  1.59251112, -0.0479162;
		// factor_mat[9] = Eigen::MatrixXd::Zero(3, 2);
		// factor_mat[9] << 0.8730462, -1.01843977,
    //                  -0.4562021, 0.97558878,
    //                  1.8641821, -0.05799166;
		// row_coef << 0.2655087, 0.9082078, 0.9446753, 0.06178627, 0.6870228, 0.4976992, 0.3800352, 0.2121425, 0.26722067, 0.3823880,
		// 						0.3721239, 0.2016819, 0.6607978, 0.20597457, 0.3841037, 0.7176185, 0.7774452, 0.6516738, 0.38611409, 0.8696908,
		// 						0.5728534, 0.8983897, 0.6291140, 0.17655675, 0.7698414, 0.9919061, 0.9347052, 0.1255551, 0.01339033, 0.3403490;
		// col_coef << 0.4820801, 0.4935413, 0.8273733, 0.7942399, 0.7237109, 0.8209463, 0.7829328, 0.5297196, 0.0233312, 0.7323137,
		// 					  0.5995658, 0.1862176, 0.6684667, 0.1079436, 0.4112744, 0.6470602, 0.5530363, 0.7893562, 0.4772301, 0.6927316;
		// row_sig_lower = Eigen::MatrixXd::Identity(num_row, num_row) * sqrt(.5);
		// col_sig_lower = Eigen::MatrixXd::Identity(num_col, num_col) * sqrt(.3);
		// factor_prec = Eigen::VectorXd::Ones(6);
		// factor_coef << 0.847762, 0.8861209, 0.8438097, 0.8244797, 0.8070679, 0.8099466;
		// To check Gibbs sampler
	}
	virtual ~McmcMatDfmVar() = default;

protected:
	void updateFactor() override {
		BVHAR_DEBUG_LOG(debug_logger, "updateFactor() called");
		draw_dfm_factor(
			factor_mat, lag, nrow_factor, ncol_factor, factor_coef, factor_prec,
			row_coef.transpose(), row_sig_lower,
			col_coef.transpose(), col_sig_lower,
			y, rng
		);
		draw_dfm_prec(factor_prec, lag, ig_shp, ig_scl, factor_mat, factor_coef, rng);
		draw_dfm_coef(factor_coef, factor_prec, prior_mean, prior_prec, factor_mat, lag, rng);
	}

	void updateRecords() override {
		BVHAR_DEBUG_LOG(debug_logger, "updateRecords() called");
		mdfm_record->assignRecords(
			mcmc_step, row_coef, row_sig_lower, col_coef, col_sig_lower,
			factor_mat, factor_coef, factor_prec,
			nrow_row_coef, num_row, nrow_col_coef, num_col,
			num_design, size_factor
		);
	}

private:
	Eigen::VectorXd ig_shp, ig_scl;
	Eigen::VectorXd prior_mean, prior_prec;
	Eigen::MatrixXd factor_coef; // p1*p2 x s
	Eigen::VectorXd factor_prec; // lambda_{1, 1}, ..., lambda_{p1, p2}
};

template <typename BaseDfm = McmcMatDfmVar>
inline std::vector<std::unique_ptr<McmcMatDfm>> initialize_matdfm(
	int num_chains, int num_iter,
	std::vector<Eigen::MatrixXd>& y, int factor_lag,
	LIST& param_dfm, LIST_OF_LIST& dfm_init,
	LIST& row_prior, LIST_OF_LIST& row_init, const int row_prior_type,
	LIST& col_prior, LIST_OF_LIST& col_init, const int col_prior_type,
  Eigen::Ref<const Eigen::VectorXi> seed_chain
) {
	using PARAMS = typename std::conditional<std::is_same<BaseDfm, McmcMatDfmVar>::value, MatDfmVarParams, MatDfmParams>::type;
	using INITS = typename std::conditional<std::is_same<BaseDfm, McmcMatDfmVar>::value, MatDfmVarInits, MatMniwInits>::type;
	PARAMS params(num_iter, y, param_dfm);
	std::vector<std::unique_ptr<McmcMatDfm>> mcmc_ptr(num_chains);
	for (int i = 0; i < num_chains; ++i) {
		LIST row_init_spec = row_init[i];
		LIST col_init_spec = col_init[i];
		auto row_updater = initialize_matshrinkageupdater(num_iter, row_prior, row_init_spec, row_prior_type);
		auto col_updater = initialize_matshrinkageupdater(num_iter, col_prior, col_init_spec, col_prior_type);
		row_updater->initPrec(params._row_prec.head(params._row_row_coef));
		col_updater->initPrec(params._col_prec.head(params._row_col_coef));
		LIST init_spec = dfm_init[i];
		INITS inits(init_spec);
		mcmc_ptr[i] = std::make_unique<BaseDfm>(params, inits, row_updater, col_updater, static_cast<unsigned int>(seed_chain[i]));
	}
	return mcmc_ptr;
}

template <typename BaseDfm = McmcMatDfmVar>
class MatDfmRun : public bvhar::McmcRun {
public:
	MatDfmRun(
		int num_chains, int num_iter, int num_burn, int thin,
		std::vector<Eigen::MatrixXd>& y, int factor_lag,
		LIST& param_dfm, LIST_OF_LIST& dfm_init,
		LIST& row_prior, LIST_OF_LIST& row_init, const int row_prior_type,
		LIST& col_prior, LIST_OF_LIST& col_init, const int col_prior_type,
		Eigen::Ref<const Eigen::VectorXi> seed_chain, bool display_progress, int nthreads
	)
	: bvhar::McmcRun(num_chains, num_iter, num_burn, thin, display_progress, nthreads) {
		auto temp_mcmc = initialize_matdfm<BaseDfm>(
			num_chains, num_iter - num_burn, y, factor_lag, param_dfm, dfm_init,
			row_prior, row_init, row_prior_type,
			col_prior, col_init, col_prior_type,
			seed_chain
		);
		for (int i = 0; i < num_chains; ++i) {
			mcmc_ptr[i] = std::move(temp_mcmc[i]);
		}
	}
	virtual ~MatDfmRun() = default;
};

} // namespace baymar

#endif // BAYMAR_BAYES_MDFM_MDFM_H