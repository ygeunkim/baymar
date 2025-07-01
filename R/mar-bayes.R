#' Fitting Bayesian MAR
#' 
#' This function fits Bayesian Matrix Autoregressive (BMAR) model with various priors.
#' 
#' @param y Matrix-valued time series data
#' @param p VAR lag (Default: 1)
#' @param exogen Unmodeled matrices
#' @param s Lag of exogeneous matrices in MARX(p, s). By default, `s = 0`.
#' @param famar_spec Augmented factor matrix specification.
#' @param num_chains Number of MCMC chains
#' @param num_iter MCMC iteration number
#' @param num_burn Number of burn-in (warm-up). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param row_spec Row coefficient specification
#' @param col_spec Column coefficient specification
#' @param exogen_row_spec Exogenous row coefficient prior specification.
#' @param exogen_col_spec Exogenous column coefficient prior specification.
#' @param factor_row_spec Factor row coefficient prior specification.
#' @param factor_col_spec Factor column coefficient prior specification.
#' @param verbose Progress log
#' @param num_thread Number of threads
#' 
#' @references
#' Chan, J. C. C. & Qi, Y. (2025). Large Bayesian matrix autoregressions. Journal of Econometrics, 105955.
#' 
#' Zhang, W. (2025). Bayesian Dynamic Factor Models for High-Dimensional Matrix-Valued Time Series. SSRN Electronic Journal.
#' @importFrom Matrix bdiag
#' @importFrom utils tail
#' @importFrom posterior as_draws_df bind_draws summarise_draws
#' @order 1
#' @export
mar_bayes <- function(y,
                      p = 1,
                      exogen = NULL,
                      s = 0,
                      famar_spec = set_famar(),
                      num_chains = 1,
                      num_iter = 1000,
                      num_burn = floor(num_iter / 2),
                      thinning = 1,
                      row_spec = set_mar_minnesota(),
                      col_spec = row_spec,
                      exogen_row_spec = row_spec,
                      exogen_col_spec = row_spec,
                      factor_row_spec = row_spec,
                      factor_col_spec = col_spec,
                      verbose = FALSE,
                      num_thread = 1) {
  if (!is.array(y)) {
    stop("Provide array.")
  }
  if (length(dim(y)) != 3) {
    stop("Array should be 3-dim: variable x region x time")
  }
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
  for (i in (seq_along(response) + p)) {
    # design[[i]] <- bdiag(y_list[i:(i + p - 1)])
    design[[i - p]] <- bdiag(y_list[(i - 1):(i - p)]) # diag(Y_{t - 1}, ..., Y_{t - p}), t = p + 1, ..., T
  }
  name_row_lag <- lapply(
    1:p,
    function(lag) paste(var_names[[1]], lag, sep = "_")
  ) |>
    unlist()
  name_col_lag <- lapply(
    1:p,
    function(lag) paste(var_names[[2]], lag, sep = "_")
  ) |>
    unlist()
  nrow_exogen <- 0
  ncol_exogen <- 0
  nrow_exogen_row_coef <- 0
  nrow_exogen_col_coef <- 0
  row_exogen_prior_type <- 0
  col_exogen_prior_type <- 0
  row_exogen_prior <- list()
  col_exogen_prior <- list()
  row_exogen_init <- list()
  col_exogen_init <- list()
  nrow_factor <- 0
  ncol_factor <- 0
  lag_factor <- 0
  row_factor_prior_type <- 0
  col_factor_prior_type <- 0
  row_factor_prior <- list()
  col_factor_prior <- list()
  row_factor_init <- list()
  col_factor_init <- list()
  is_famar <- FALSE
  if (!is.famarspec(famar_spec)) {
    stop("Wrong 'famar_spec'")
  }
  if (famar_spec$nrow_factor > 0 && famar_spec$ncol_factor > 0) {
    nrow_factor <- famar_spec$nrow_factor
    ncol_factor <- famar_spec$ncol_factor
    lag_factor <- famar_spec$lag
    is_famar <- TRUE
  }
  if (!is.null(exogen)) {
    if (!is.array(exogen)) {
      stop("Provide array.")
    }
    if (length(dim(exogen)) != 3) {
      stop("Array should be 3-dim: variable x region x time")
    }
    if (is.null(dimnames(exogen))) {
      dimnames(exogen) <- list(
        paste("x_row", seq_len(nrow_exogen), sep = "_"),
        paste("x_col", seq_len(ncol_exogen), sep = "_"),
        seq_len(num_data)
      )
    }
    name_exogen <- dimnames(exogen)
    nrow_exogen <- dim(exogen)[1]
    ncol_exogen <- dim(exogen)[2]
    name_row_lag <- c(
      name_row_lag,
      lapply(
        0:s,
        function(lag) paste(name_exogen[[1]], lag, sep = "_")
      ) |> 
      unlist()
    )
    name_col_lag <- c(
      name_col_lag,
      lapply(
        0:s,
        function(lag) paste(name_exogen[[2]], lag, sep = "_")
      ) |>
        unlist()
    )
    row_exogen_id <- nrow_row_coef + seq_len((s + 1) * nrow_exogen)
    col_exogen_id <- nrow_col_coef + seq_len((s + 1) * ncol_exogen)
    nrow_exogen_row_coef <- length(row_exogen_id)
    nrow_exogen_col_coef <- length(col_exogen_id)
    exogen_list <- lapply(seq_len(dim(exogen)[3]), function(x) exogen[, , x])
    for (i in (seq_along(response) + p)) {
      design[[i - p]] <- bdiag(append(
        design[[i - p]],
        exogen_list[i:(i - s)]
      )) # X_t, ..., X_{t - s}
    }
    row_exogen_prior <- validate_bmar_prior(exogen_row_spec)
    col_exogen_prior <- validate_bmar_prior(exogen_col_spec)
    row_exogen_prior_type <- get_prior_id(exogen_row_spec$prior)
    col_exogen_prior_type <- get_prior_id(exogen_col_spec$prior)
  }
  param_prior <- validate_bmar_row_spec(
    y = y,
    p = p,
    bayes_spec = row_spec,
    nrow_data = nrow_data,
    ncol_data = ncol_data,
    nrow_row_coef = nrow_row_coef
  )
  param_prior <- append(
    param_prior,
    validate_bmar_col_spec(
      y = y,
      p = p,
      bayes_spec = col_spec,
      nrow_data = nrow_data,
      ncol_data = ncol_data,
      nrow_col_coef = nrow_col_coef
    )
  )
  # Initialization
  param_init <- get_bmar_coef_init(
    num_chains = num_chains,
    nrow_data = nrow_data,
    ncol_data = ncol_data,
    nrow_row_coef = nrow_row_coef + nrow_exogen_row_coef + nrow_factor,
    nrow_col_coef = nrow_col_coef + nrow_exogen_col_coef + ncol_factor
  )
  row_prior <- validate_bmar_prior(row_spec)
  col_prior <- validate_bmar_prior(col_spec)
  row_init <- get_bmar_init(row_spec, num_chains, nrow_row_coef)
  col_init <- get_bmar_init(col_spec, num_chains, nrow_col_coef)
  row_prior_type <- get_prior_id(row_spec$prior)
  col_prior_type <- get_prior_id(col_spec$prior)
  if (!is.null(exogen)) {
    param_prior <- validate_bmarx_rowspec(
      param_prior = param_prior,
      x = exogen,
      s = s,
      bayes_spec = exogen_row_spec,
      nrow_exogen = nrow_exogen,
      ncol_exogen = ncol_exogen,
      nrow_exogen_row_coef = nrow_exogen_row_coef
    )
    param_prior <- validate_bmarx_colspec(
      param_prior = param_prior,
      x = exogen,
      s = s,
      bayes_spec = exogen_row_spec,
      nrow_exogen = nrow_exogen,
      ncol_exogen = ncol_exogen,
      nrow_exogen_col_coef = nrow_exogen_col_coef
    )
    row_exogen_init <- get_bmar_init(exogen_row_spec, num_chains, nrow_exogen_row_coef)
    col_exogen_init <- get_bmar_init(exogen_col_spec, num_chains, nrow_exogen_col_coef)
  }
  if (is_famar) {
    row_factor_prior <- validate_bmar_prior(factor_row_spec)
    col_factor_prior <- validate_bmar_prior(factor_col_spec)
    row_factor_prior_type <- get_prior_id(factor_row_spec$prior)
    col_factor_prior_type <- get_prior_id(factor_col_spec$prior)
    param_prior$row_prior_mean <- rbind(
      param_prior$row_prior_mean,
      matrix(0L, nrow = nrow_factor, ncol = nrow_data)
    )
    param_prior$row_prior_prec <- c(param_prior$row_prior_prec, rep(1, nrow_factor))
    param_prior$col_prior_mean <- rbind(
      param_prior$col_prior_mean,
      matrix(0L, nrow = ncol_factor, ncol = ncol_data)
    )
    param_prior$col_prior_prec <- c(param_prior$col_prior_prec, rep(1, ncol_factor))
    row_factor_init <- get_bmar_init(factor_row_spec, num_chains, nrow_factor)
    col_factor_init <- get_bmar_init(factor_col_spec, num_chains, ncol_factor)
    for (i in (seq_along(response) + p)) {
      design[[i - p]] <- bdiag(append(
        design[[i - p]],
        list(matrix(1L, nrow = nrow_factor, ncol = ncol_factor))
      ))
    }
    name_row_lag <- c(
      name_row_lag,
      paste("factor", seq_len(nrow_factor), sep = "_")
    )
    name_col_lag <- c(
      name_col_lag,
      paste("factor", seq_len(ncol_factor), sep = "_")
    )
    res <- estimate_bmdfm_mniw(
      num_chains = num_chains, num_iter = num_iter, num_burn = num_burn, thin = thinning,
      x = design, y = response,
      param_coef_sig = param_prior, coef_sig_init = param_init,
      row_prior = row_prior, row_init = row_init, row_prior_type = row_prior_type,
      col_prior = col_prior, col_init = col_init, col_prior_type = col_prior_type,
      exogen_row_prior = row_exogen_prior, exogen_row_init = row_exogen_init, exogen_row_prior_type = row_exogen_prior_type, exogen_rows = nrow_exogen_row_coef,
      exogen_col_prior = col_exogen_prior, exogen_col_init = col_exogen_init, exogen_col_prior_type = col_exogen_prior_type, exogen_cols = nrow_exogen_col_coef,
      factor_row_prior = row_factor_prior, factor_row_init = row_factor_init, factor_row_prior_type = row_factor_prior_type, factor_rows = nrow_factor,
      factor_col_prior = col_factor_prior, factor_col_init = col_factor_init, factor_col_prior_type = col_factor_prior_type, factor_cols = ncol_factor,
      factor_lag = lag_factor,
      seed_chain = sample.int(.Machine$integer.max, size = num_chains),
      display_progress = verbose, nthreads = num_thread
    )
  } else {
    res <- estimate_bmar_mniw(
      num_chains = num_chains, num_iter = num_iter, num_burn = num_burn, thin = thinning,
      x = design, y = response,
      param_coef_sig = param_prior, coef_sig_init = param_init,
      row_prior = row_prior, row_init = row_init, row_prior_type = row_prior_type,
      col_prior = col_prior, col_init = col_init, col_prior_type = col_prior_type,
      exogen_row_prior = row_exogen_prior, exogen_row_init = row_exogen_init, exogen_row_prior_type = row_exogen_prior_type, exogen_rows = nrow_exogen_row_coef,
      exogen_col_prior = col_exogen_prior, exogen_col_init = col_exogen_init, exogen_col_prior_type = col_exogen_prior_type, exogen_cols = nrow_exogen_col_coef,
      seed_chain = sample.int(.Machine$integer.max, size = num_chains),
      display_progress = verbose, nthreads = num_thread
    )
  }
  res <- do.call(rbind, res)
  rec_names <- colnames(res)
  param_names <- gsub(pattern = "_record$", replacement = "", rec_names)
  res <- apply(
    res,
    2,
    function(x) {
      if (is.vector(x[[1]])) {
        return(as.matrix(unlist(x)))
      }
      do.call(rbind, x)
    }
  )
  names(res) <- rec_names
  row_coef <- matrix(colMeans(res$A_record), ncol = nrow_data)
  col_coef <- matrix(colMeans(res$B_record), ncol = ncol_data)
  if (!is.null(exogen)) {
    row_coef <- rbind(
      row_coef,
      matrix(colMeans(res$C_record), ncol = nrow_data)
    )
    col_coef <- rbind(
      col_coef,
      matrix(colMeans(res$D_record), ncol = ncol_data)
    )
  }
  if (is_famar) {
    row_coef <- rbind(
      row_coef,
      matrix(colMeans(res$G_record), ncol = nrow_data)
    )
    col_coef <- rbind(
      col_coef,
      matrix(colMeans(res$H_record), ncol = ncol_data)
    )
  }
  row_sig <- diag(nrow_data)
  row_sig[lower.tri(row_sig, diag = TRUE)] <- colMeans(res$SigmaR_record)
  row_sig[upper.tri(row_sig, diag = FALSE)] <- row_sig[lower.tri(row_sig, diag = FALSE)]
  col_sig <- diag(ncol_data)
  col_sig[lower.tri(col_sig, diag = TRUE)] <- colMeans(res$SigmaC_record)
  col_sig[upper.tri(col_sig, diag = FALSE)] <- col_sig[lower.tri(col_sig, diag = FALSE)]
  is_symm <- grepl(pattern = "^Sigma", x = param_names)
  num_col <- c(nrow_data, nrow_data, ncol_data, ncol_data)
  num_row <- c(nrow_row_coef, nrow_data, nrow_col_coef, ncol_data)
  num_matrix <- rep(0, 4)
  if (!is.null(exogen)) {
    num_col <- c(num_col, nrow_data, ncol_data)
    num_row <- c(num_row, nrow_exogen_row_coef, nrow_exogen_col_coef)
    num_matrix <- c(num_matrix, rep(0, 3))
  }
  if (is_famar) {
    num_col <- c(num_col, nrow_data, ncol_data, ncol_factor)
    num_row <- c(num_row, nrow_factor, ncol_factor, nrow_factor)
    num_matrix <- c(num_matrix, rep(0, 2), length(y_list) - p)
  }
  # num_row <- c(nrow_row_coef + nrow_exogen_row_coef, nrow_data, nrow_col_coef + nrow_exogen_col_coef, ncol_data)
  res[rec_names] <- lapply(
    seq_along(res[rec_names]),
    function(id) {
      split_matrix_chain(
        res[rec_names][[id]],
        chain = num_chains, varname = param_names[id],
        num_row = num_row[id], num_col = num_col[id], is_symm = is_symm[id],
        num_design = num_matrix[id]
      )
    }
  )
  res[rec_names] <- lapply(res[rec_names], as_draws_df)
  res$param <- bind_draws(
    res$A_record,
    res$SigmaR_record,
    res$B_record,
    res$SigmaC_record
  )
  if (!is.null(exogen)) {
    res$param <- bind_draws(
      res$param,
      res$C_record,
      res$D_record
    )
  }
  res[rec_names] <- NULL
  res$param_names <- param_names
  rownames(row_coef) <- name_row_lag
  colnames(row_coef) <- var_names[[1]]
  rownames(row_sig) <- var_names[[1]]
  colnames(row_sig) <- var_names[[1]]
  rownames(col_coef) <- name_col_lag
  colnames(col_coef) <- var_names[[2]]
  rownames(col_sig) <- var_names[[2]]
  colnames(col_sig) <- var_names[[2]]
  res$coefficients <- list(
    row = row_coef,
    col = col_coef
  )
  res$covmat <- list(
    row = row_sig,
    col = col_sig
  )
  res$spec <- list(
    row = row_spec,
    col = col_spec
  )
  res$init <- list(
    param = param_init,
    row = row_init,
    col = col_init
  )
  if (!is.null(exogen)) {
    res$spec <- append(
      res$spec,
      list(exogen_row = exogen_row_spec, exogen_col = exogen_col_spec)
    )
    res$init <- append(
      res$init,
      list(exogen_row = row_exogen_init, exogen_col = col_exogen_init)
    )
    res$exogen_data <- exogen
    res$s <- s
    res$exogen_row_id <- row_exogen_id
    res$exogen_col_id <- col_exogen_id
  }
  res$call <- match.call()
  res$y <- y
  res$chain <- num_chains
  res$iter <- num_iter
  res$burn <- num_burn
  res$thin <- thinning
  res$p <- p
  class(res) <- "marbayes"
  res
}
