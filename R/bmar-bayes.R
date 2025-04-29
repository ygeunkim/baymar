#' Fitting BMAR(p) with Minnesota Prior
#' 
#' @param y Matrix-valued time series data
#' @param p VAR lag (Default: 1)
#' @param num_chains Number of MCMC chains
#' @param num_iter MCMC iteration number
#' @param num_burn Number of burn-in (warm-up). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param num_thread Number of threads
#' 
#' @references
#' Chan, J. C. C. & Qi, Y. (2024). Large Bayesian Tensor VARs with Stochastic Volatility. arXiv.
#' 
#' @importFrom Matrix bdiag
#' @importFrom stats arima
#' @importFrom purrr flatten
#' @order 1
#' @export
mar_bayes <- function(y,
                      p = 1,
                      num_chains = 1,
                      num_iter = 1000, 
                      num_burn = floor(num_iter / 2),
                      thinning = 1,
                      num_thread = 1) {
  y_list <- lapply(seq_len(dim(y)[3]), function(x) y[, , x])
  response <- tail(y_list, length(y_list) - p) # Y_{p + 1}, ..., Y_T
  design <- list()
  for (i in seq_along(response)) {
    design[[i]] <- Matrix::bdiag(y_list[i:(i + p - 1)])
  }
  # Prior
  # Y_t: n (variables) x k (regions)
  # X_t = diag(Y_{t - 1}, ..., Y_{t - p})
  # Y: n x k x (T - p)
  # X: np x kp x (T - p)
  n <- dim(y)[1]
  k <- dim(y)[2]
  T <- dim(y)[3]
  # A = (A1, ..., Ap)^T
  # A(np x n) ~ MN(A0, V_A, Sigma_r)
  # Sigma_r(n x n) ~ IW(nu_r, S_r)
  S_r <- diag(n)
  # diag(S_r) <- sapply(
  #   1:n,
  #   function(i) arima(c(y[i,,]), order = c(4, 0, 0))$sigma2
  # )
  A0 <- matrix(0L, nrow = n * p, ncol = n)
  kappa_A <- .1
  V_A <- matrix(0L, nrow = n * p, ncol = n * p)
  for (l in 1:p) {
    idx <- c(1:n) + (l - 1) * n
    diag(V_A)[idx] <- kappa_A / (l^2 * diag(S_r))
  }
  nu_r <- n + 2
  # B = (B1, ..., Bp)^T
  # B(kp x k) ~ MN(B0, V_B, Sigma_c)
  # Sigma_c(k x k) ~ IW(nu_c, S_c)
  B0 <- do.call(rbind, lapply(1:p, function(x) diag(k))) # kp x k
  S_c <- diag(k)
  # diag(S_c) <- sapply(
  #   1:k,
  #   function(i) arima(c(y[, i,]), order = c(4, 0, 0))$sigma2
  # )
  kappa_B <- .1
  V_B <- matrix(0L, nrow = k * p, ncol = k * p)
  for (l in 1:p) {
    idx <- c(1:k) + (l - 1) * k
    diag(V_B)[idx] <- kappa_A / (l^2 * diag(S_c))
  }
  nu_c <- k + 2
  # Initialization
  init_row <- lapply(
    seq_len(num_chains),
    function(x) {
      init_cov <- diag(exp(runif(n, -1, 0)))
      # init_cov <- matrix(exp(runif(n^2, -1, 0)), ncol = n)
      # init_cov[lower.tri(init_cov)] <- t(init_cov)[lower.tri(init_cov)]
      list(
        init_coef = matrix(runif(n * p * k, -1, 1), ncol = n),
        init_cov = init_cov
      )
    }
  )
  init_col <- lapply(
    seq_len(num_chains),
    function(x) {
      init_cov <- diag(exp(runif(k, -1, 0)))
      # init_cov <- matrix(exp(runif(k^2, -1, 0)), ncol = k)
      # init_cov[lower.tri(init_cov)] <- t(init_cov)[lower.tri(init_cov)]
      list(
        init_coef = matrix(runif(k^2 * p, -1, 1), ncol = k),
        init_cov = init_cov
      )
    }
  )
  res <- estimate_bmar_mniw(
    num_chains = num_chains, num_iter = num_iter, num_burn = num_burn, thin = thinning,
    x = design, y = response,
    # row_prior_mean = A0, row_prior_prec = diag(1 / diag(V_A)), row_iw_scl = S_r, row_iw_df = nu_r,
    # col_prior_mean = B0, col_prior_prec = diag(1 / diag(V_B)), col_iw_scl = S_c, col_iw_df = nu_c,
    row_prior_mean = matrix(0, nrow = n * p, ncol = n), row_prior_prec = diag(n * p), row_iw_scl = diag(n), row_iw_df = n + 2,
    col_prior_mean = matrix(0, nrow = k * p, ncol = k), col_prior_prec = diag(k * p), col_iw_scl = diag(k), col_iw_df = k + 2,
    init_row = init_row, init_col = init_col,
    seed_chain = sample.int(.Machine$integer.max, size = num_chains),
    nthreads = num_thread
  )
  row_record <-
    lapply(res, function(x) tail(x$row_record, num_iter - num_burn)) |> # chain x iter x (coef_sig)
    flatten()
  row_coef <- lapply(row_record, function(x) x[[1]])
  row_sig <- lapply(row_record, function(x) x[[2]])
  col_record <-
    lapply(res, function(x) tail(x$col_record, num_iter - num_burn)) |> # chain x iter x (coef_sig)
    flatten()
  col_coef <- lapply(col_record, function(x) x[[1]])
  col_sig <- lapply(col_record, function(x) x[[2]])
  res$row_coef <- Reduce("+", row_coef) / length(row_coef)
  colnames(res$row_coef) <- dimnames(y)[[1]]
  res$row_sig <- Reduce("+", row_sig) / length(row_sig)
  rownames(res$row_sig) <- dimnames(y)[[1]]
  colnames(res$row_sig) <- dimnames(y)[[1]]
  res$col_coef <- Reduce("+", col_coef) / length(col_coef)
  colnames(res$col_coef) <- dimnames(y)[[2]]
  res$col_sig <- Reduce("+", col_sig) / length(col_sig)
  rownames(res$col_sig) <- dimnames(y)[[2]]
  colnames(res$col_sig) <- dimnames(y)[[2]]
  res[c("row_coef", "row_sig", "col_coef", "col_sig")]
}
