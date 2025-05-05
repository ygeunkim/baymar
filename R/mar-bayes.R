#' Fitting BMAR(p) with Minnesota Prior
#' 
#' @param y Matrix-valued time series data
#' @param p VAR lag (Default: 1)
#' @param num_chains Number of MCMC chains
#' @param num_iter MCMC iteration number
#' @param num_burn Number of burn-in (warm-up). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param row_spec Row coefficient specification
#' @param col_spec Column coefficient specification
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
                      row_spec = set_minnesota(),
                      col_spec = set_minnesota(),
                      num_thread = 1) {
  if (!is.array(y)) {
    stop("Provide array.")
  }
  if (length(dim(y)) != 3) {
    stop("Array should be 3-dim: variable x region x time")
  }
  validate_bmar_spec(row_spec, "row")
  validate_bmar_spec(col_spec, "col")
  nrow_data <- dim(y)[1]
  ncol_data <- dim(y)[2]
  num_data <- dim(y)[3]
  nrow_row_coef <- nrow_data * p
  nrow_col_coef <- ncol_data * p
  if (is.null(dimnames(y))) {
    dimnames(y) <- list(
      paste("row", seq_len(nrow_data), sep = "_"),
      paste("col", seq_len(ncol_data), sep = "_"),
      seq_len(num_data)
    )
  }
  var_names <- dimnames(y)
  y_list <- lapply(seq_len(num_data), function(x) y[, , x])
  response <- tail(y_list, length(y_list) - p) # Y_{p + 1}, ..., Y_T
  design <- list()
  for (i in seq_along(response)) {
    design[[i]] <- bdiag(y_list[i:(i + p - 1)])
  }
  S_r <- diag(nrow_data)
  diag(S_r) <- sapply(
    1:nrow_data,
    function(i) {
      sapply(
        1:ncol_data,
        function(j) {
          ar.ols(y[i, j, ], aic = FALSE, order = 4)$var.pred
        }
      ) |>
        mean()
    }
  )
  A0 <- matrix(0L, nrow = nrow_row_coef, ncol = nrow_data)
  kappa_A <- .1
  V_A <- matrix(0L, nrow = nrow_row_coef, ncol = nrow_row_coef)
  V_A <- kronecker(diag(1 / c(1:p)^2), diag(kappa_A / diag(S_r)))
  nu_r <- nrow_data + 2
  B0 <- kronecker(rep(1, p), diag(ncol_data)) # kp x k
  S_c <- diag(ncol_data)
  diag(S_c) <- sapply(
    1:ncol_data,
    function(i) {
      sapply(
        1:nrow_data,
        function(j) {
          ar.ols(y[j, i, ], aic = FALSE, order = 4)$var.pred
        }
      ) |>
        mean()
    }
  )
  kappa_B <- .1
  V_B <- matrix(0L, nrow = nrow_col_coef, ncol = nrow_col_coef)
  V_B <- kronecker(diag(1 / c(1:p)^2), diag(kappa_B / diag(S_c)))
  nu_c <- ncol_data + 2
  param_prior <- list(
    row_prior_mean = A0,
    row_prior_prec = diag(1 / diag(V_A)),
    row_iw_scl = S_r,
    row_iw_df = nu_r,
    col_prior_mean = B0,
    col_prior_prec = diag(1 / diag(V_B)),
    col_iw_scl = S_c,
    col_iw_df = nu_c
  )
  # Initialization
  param_init <- lapply(
    seq_len(num_chains),
    function(x) {
      init_cov <- diag(exp(runif(nrow_data, -1, 0)))
      # init_cov <- matrix(exp(runif(n^2, -1, 0)), ncol = n)
      # init_cov[lower.tri(init_cov)] <- t(init_cov)[lower.tri(init_cov)]
      list(
        row_init_coef = matrix(runif(nrow_row_coef * nrow_data, -1, 1), ncol = nrow_data),
        row_init_lower = init_cov
      )
    }
  )
  param_init <- lapply(
    param_init,
    function(init) {
      init_cov <- diag(exp(runif(ncol_data, -1, 0)))
      # init_cov <- matrix(exp(runif(k^2, -1, 0)), ncol = k)
      # init_cov[lower.tri(init_cov)] <- t(init_cov)[lower.tri(init_cov)]
      append(
        init,
        list(
          col_init_coef = matrix(runif(nrow_col_coef * ncol_data, -1, 1), ncol = ncol_data),
          col_init_lower = init_cov
        )
      )
    }
  )
  row_prior <- row_spec
  col_prior <- col_spec
  row_init <- lapply(
    seq_len(num_chains),
    function(x) {
      list(
        kappa = runif(1, 0, 1)
      )
    }
  )
  col_init <- lapply(
    seq_len(num_chains),
    function(x) {
      list(
        kappa = runif(1, 0, 1)
      )
    }
  )
  res <- estimate_bmar_mniw(
    num_chains = num_chains, num_iter = num_iter, num_burn = num_burn, thin = thinning,
    x = design, y = response,
    param_coef_sig = param_prior, coef_sig_init = param_init,
    row_prior = row_prior, row_init = row_init, row_prior_type = 1,
    col_prior = col_prior, col_init = col_init, col_prior_type = 1,
    seed_chain = sample.int(.Machine$integer.max, size = num_chains),
    display_progress = TRUE, nthreads = num_thread
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
