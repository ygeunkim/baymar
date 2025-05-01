#ifndef BAYMAR_BAYES_MNIW_MNIW_H
#define BAYMAR_BAYES_MNIW_MNIW_H

#include "./config.h"

namespace baymar {

class McmcMatMniw;

class McmcMatMniw {
public:
	McmcMatMniw(
		int num_iter,
		std::vector<Eigen::SparseMatrix<double>>& x, std::vector<Eigen::MatrixXd>& y,
		const Eigen::MatrixXd& row_prior_mean, const Eigen::MatrixXd& row_prior_prec, const Eigen::MatrixXd& row_iw_scl, double row_iw_df,
		const Eigen::MatrixXd& col_prior_mean, const Eigen::MatrixXd& col_prior_prec, const Eigen::MatrixXd& col_iw_scl, double col_iw_df,
		std::vector<Eigen::MatrixXd> init_row, std::vector<Eigen::MatrixXd> init_col,
		unsigned int seed
	)
	: x(x), y(y),
		num_iter(num_iter), num_row(y[0].rows()), num_col(y[0].cols()), num_design(y.size()),
		row_record(num_iter + 1, std::vector<Eigen::MatrixXd>(2)), col_record(num_iter + 1, std::vector<Eigen::MatrixXd>(2)),
		mcmc_step(0), rng(seed),
		row_params(init_row), col_params(init_col),
		row_prior_mean(row_prior_mean), row_prior_prec(row_prior_prec), row_iw_scl(row_iw_scl),
		col_prior_mean(col_prior_mean), col_prior_prec(col_prior_prec), col_iw_scl(col_iw_scl),
		row_iw_df(row_iw_df), col_iw_df(col_iw_df) {
		updateRecords();
	}
	virtual ~McmcMatMniw() = default;
	
	void doWarmUp() {
		std::lock_guard<std::mutex> lock(mtx);
		updateCoefCov();
	}

	void doPosteriorDraws() {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updateCoefCov();
		updateRecords();
	}

	LIST returnRecords() {
		LIST res = CREATE_LIST(
			// NAMED("A_record") = row_coef_record,
			// NAMED("Sigr_record") = row_sig_record,
			// NAMED("B_record") = col_coef_record,
			// NAMED("Sigc_record") = col_sig_record
			NAMED("row_record") = WRAP(row_record),
			NAMED("col_record") = WRAP(col_record)
		);
		return res;
	}

protected:
	std::mutex mtx;
	std::vector<Eigen::SparseMatrix<double>> x;
	std::vector<Eigen::MatrixXd> y;
	int num_iter;
	int num_row;
	int num_col;
	int num_design;
	std::vector<std::vector<Eigen::MatrixXd>> row_record, col_record;
	std::atomic<int> mcmc_step; // MCMC step
	BHRNG rng; // RNG instance for multi-chain
	std::vector<Eigen::MatrixXd> row_params, col_params;
	Eigen::MatrixXd row_prior_mean, row_prior_prec, row_iw_scl;
	Eigen::MatrixXd col_prior_mean, col_prior_prec, col_iw_scl;
	double row_iw_df, col_iw_df;

	/**
	 * @brief Increment the MCMC step
	 * 
	 */
	void addStep() { ++mcmc_step; }

	void updateCoefCov() {
		draw_coefsig_row(
			row_params, col_params,
			row_prior_mean, row_prior_prec, row_iw_scl, row_iw_df,
			num_design, num_col,
			x, y, rng
		);
		draw_coefsig_col(
			col_params, row_params,
			col_prior_mean, col_prior_prec, col_iw_scl, col_iw_df,
			num_design, num_row,
			x, y, rng
		);
	}

	void updateRecords() {
		row_record[mcmc_step] = row_params;
		col_record[mcmc_step] = col_params;
	}
};

} // namespace baymar

#endif // BAYMAR_BAYES_MNIW_MNIW_H