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
#' @importFrom stats ar.ols mean
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
  if (!is.array(y)) {
    stop("Provide array.")
  }
  if (length(dim(y)) != 3) {
    stop("Array should be 3-dim: variable x region x time")
  }
  n <- dim(y)[1]
  k <- dim(y)[2]
  T <- dim(y)[3]
  if (is.null(dimnames(y))) {
    dimnames(y) <- list(
      paste("row", seq_len(n), sep = "_"),
      paste("col", seq_len(k), sep = "_"),
      seq_len(T)
    )
  }
  var_names <- dimnames(y)
  y_list <- lapply(seq_len(T), function(x) y[, , x])
  response <- tail(y_list, length(y_list) - p) # Y_{p + 1}, ..., Y_T
  design <- list()
  for (i in seq_along(response)) {
    design[[i]] <- bdiag(y_list[i:(i + p - 1)])
  }
  S_r <- diag(n)
  diag(S_r) <- sapply(
    1:n,
    function(i) {
      sapply(
        1:k,
        function(j) {
          ar.ols(y[i, j, ], aic = FALSE, order = 4)$var.pred
        }
      ) |>
        mean()
    }
  )
  A0 <- matrix(0L, nrow = n * p, ncol = n)
  kappa_A <- .1
  V_A <- matrix(0L, nrow = n * p, ncol = n * p)
  V_A <- kronecker(diag(1 / c(1:p)^2), diag(kappa_A / diag(S_r)))
  nu_r <- n + 2
  B0 <- kronecker(rep(1, p), diag(k)) # kp x k
  S_c <- diag(k)
  diag(S_c) <- sapply(
    1:k,
    function(i) {
      sapply(
        1:n,
        function(j) {
          ar.ols(y[j, i, ], aic = FALSE, order = 4)$var.pred
        }
      ) |>
        mean()
    }
  )
  kappa_B <- .1
  V_B <- matrix(0L, nrow = k * p, ncol = k * p)
  # for (l in 1:p) {
  #   idx <- c(1:k) + (l - 1) * k
  #   diag(V_B)[idx] <- kappa_B / (l^2 * diag(S_c))
  # }
  V_B <- kronecker(diag(1 / c(1:p)^2), diag(kappa_B / diag(S_c)))
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
    row_prior_mean = A0, row_prior_prec = diag(1 / diag(V_A)), row_iw_scl = S_r, row_iw_df = nu_r,
    col_prior_mean = B0, col_prior_prec = diag(1 / diag(V_B)), col_iw_scl = S_c, col_iw_df = nu_c,
    # row_prior_mean = matrix(0, nrow = n * p, ncol = n), row_prior_prec = diag(n * p), row_iw_scl = diag(n), row_iw_df = n + 2,
    # col_prior_mean = matrix(0, nrow = k * p, ncol = k), col_prior_prec = diag(k * p), col_iw_scl = diag(k), col_iw_df = k + 2,
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
  rownames(res$row_coef) <- lapply(
    1:p,
    function(lag) paste(var_names[[1]], lag, sep = "_")
  ) |>
    unlist()
  colnames(res$row_coef) <- var_names[[1]]
  res$row_sig <- Reduce("+", row_sig) / length(row_sig)
  rownames(res$row_sig) <- var_names[[1]]
  colnames(res$row_sig) <- var_names[[1]]
  res$col_coef <- Reduce("+", col_coef) / length(col_coef)
  rownames(res$col_coef) <- lapply(
    1:p,
    function(lag) paste(var_names[[2]], lag, sep = "_")
  ) |>
    unlist()
  colnames(res$col_coef) <- var_names[[2]]
  res$col_sig <- Reduce("+", col_sig) / length(col_sig)
  rownames(res$col_sig) <- var_names[[2]]
  colnames(res$col_sig) <- var_names[[2]]
  res[c("row_coef", "row_sig", "col_coef", "col_sig")]
}
