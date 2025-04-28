#ifndef BAYMAR_BAYES_MISC_COEF_HELPER_H
#define BAYMAR_BAYES_MISC_COEF_HELPER_H

#include <bvhar/utils>

namespace baymar {

inline void draw_coefsig(std::vector<Eigen::MatrixXd>& coef_sig,
												 Eigen::Ref<Eigen::MatrixXd> prior_mean, Eigen::Ref<Eigen::MatrixXd> prior_prec,
												 Eigen::Ref<Eigen::MatrixXd> other_coef, Eigen::Ref<Eigen::MatrixXd> other_prec,
												 Eigen::Ref<Eigen::MatrixXd> iw_scl,
												 double iw_df, int num_mat, int other_dim,
												 std::vector<Eigen::SparseMatrix<double>>& x, std::vector<Eigen::MatrixXd>& y) {
	Eigen::MatrixXd post_cov = std::accumulate(
		x.begin(), x.end(), prior_prec,
		[&other_coef, &other_prec](const Eigen::MatrixXd& acc, const Eigen::SparseMatrix<double>& x_t) {
			Eigen::MatrixXd temp = x_t * other_coef * other_prec * other_coef.transpose() * x_t.transpose();
			return acc + temp;
		}
	);
	int id = 0;
	Eigen::MatrixXd post_solve = std::accumulate(
		x.begin(), x.end(), (prior_prec * prior_mean).eval(),
		[&other_coef, &other_prec, &y, &id](const Eigen::MatrixXd& acc, const Eigen::SparseMatrix<double>& x_t) {
			Eigen::MatrixXd temp = x_t * other_coef * other_prec * y[id].transpose();
			++id;
			return acc + temp;
		}
	);
	Eigen::MatrixXd post_mean = post_cov.llt().solve(post_solve);
	double post_df = iw_df + num_mat * other_dim;
	Eigen::MatrixXd post_iw_scl = std::accumulate(
		y.begin(), y.end(), (iw_scl + prior_mean.transpose() * prior_prec * prior_mean - post_mean.transpose() * post_cov * post_mean).eval(),
		[&other_prec](const Eigen::MatrixXd& acc, const Eigen::MatrixXd& y_t) {
			return acc + y_t * other_prec * y_t.transpose();
		}
	);
	coef_sig = bvhar::sim_mn_iw(post_mean, post_cov, post_iw_scl, post_df, true);
}

} // namespace baymar

#endif // BAYMAR_BAYES_MISC_COEF_HELPER_H