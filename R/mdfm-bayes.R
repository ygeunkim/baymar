#' Fitting Bayesian Matrix DFM
#' 
#' This function fits Bayesian Matrix Dynamic Factor Model (MDFM) with various priors.
#' 
#' @param y Matrix-valued time series data
#' @param dfm_spec Factor matrix specification.
#' @param num_chains Number of MCMC chains
#' @param num_iter MCMC iteration number
#' @param num_burn Number of burn-in (warm-up). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param row_spec Row coefficient specification
#' @param col_spec Column coefficient specification
#' @param verbose Progress log
#' @param num_thread Number of threads
#' 
#' @order 1
#' @export
mdfm_bayes <- function(y,
                       dfm_spec = set_dfm(),
                       num_chains = 1,
                       num_iter = 1000,
                       num_burn = floor(num_iter / 2),
                       thinning = 1,
                       row_spec = set_mar_minnesota(),
                       col_spec = row_spec,
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
  p <- 1
  # nrow_row_coef <- nrow_data * p
  # nrow_col_coef <- ncol_data * p
  if (is.null(dimnames(y))) {
    dimnames(y) <- list(
      paste("row", seq_len(nrow_data), sep = "_"),
      paste("col", seq_len(ncol_data), sep = "_"),
      seq_len(num_data)
    )
  }
  var_names <- dimnames(y)
  y_list <- lapply(seq_len(num_data), function(x) y[, , x])
  # response <- tail(y_list, length(y_list) - p) # Y_{p + 1}, ..., Y_T
  response <- y_list
  # name_row_lag <- lapply(
  #   1:p,
  #   function(lag) paste(var_names[[1]], lag, sep = "_")
  # ) |>
  #   unlist()
  # name_col_lag <- lapply(
  #   1:p,
  #   function(lag) paste(var_names[[2]], lag, sep = "_")
  # ) |>
  #   unlist()
  dfm_spec <- validate_bmdfm_spec(dfm_spec)
  nrow_factor <- dfm_spec$nrow_factor
  ncol_factor <- dfm_spec$ncol_factor
  size_factor <- nrow_factor * ncol_factor
  lag_factor <- dfm_spec$lag
  name_row_lag <- paste("factor_row", seq_len(nrow_factor), sep = "_")
  name_col_lag <- paste("factor_col", seq_len(ncol_factor), sep = "_")
  if (dfm_spec$nrow_factor == 0 || dfm_spec$ncol_factor == 0) {
    stop("Wrong 'dfm_spec'")
  }
  nrow_row_coef <- nrow_factor
  nrow_col_coef <- ncol_factor
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
  param_prior <- append(
    param_prior,
    list(
      nrow_factor = nrow_factor,
      ncol_factor = ncol_factor,
      size_factor = size_factor,
      lag = lag_factor,
      shape = dfm_spec$shape,
      scale = dfm_spec$scale
    )
  )
  param_prior$row_prior_prec <- rep(1, nrow_factor)
  param_prior$col_prior_prec <- rep(1, ncol_factor)
  param_init <- get_bmdfm_coef_init(
    num_chains = num_chains,
    nrow_data = nrow_data,
    ncol_data = ncol_data,
    nrow_row_coef = nrow_row_coef,
    nrow_col_coef = nrow_col_coef,
    size_factor = size_factor,
    factor_lag = lag_factor
  )
  row_prior <- validate_bmar_prior(row_spec)
  col_prior <- validate_bmar_prior(col_spec)
  row_init <- get_bmar_init(row_spec, num_chains, nrow_row_coef)
  col_init <- get_bmar_init(col_spec, num_chains, nrow_col_coef)
  row_prior_type <- get_prior_id(row_spec$prior)
  col_prior_type <- get_prior_id(col_spec$prior)
  res <- estimate_bmdfm(
    num_chains = num_chains, num_iter = num_iter, num_burn = num_burn, thin = thinning,
    y = response,
    factor_lag = lag_factor,
    param_dfm = param_prior, dfm_init = param_init,
    row_prior = row_prior, row_init = row_init, row_prior_type = row_prior_type,
    col_prior = col_prior, col_init = col_init, col_prior_type = col_prior_type,
    seed_chain = sample.int(.Machine$integer.max, size = num_chains),
    display_progress = verbose, nthreads = num_thread
  )
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
  row_sig <- diag(nrow_data)
  row_sig[lower.tri(row_sig, diag = TRUE)] <- colMeans(res$SigmaR_record)
  row_sig[upper.tri(row_sig, diag = FALSE)] <- row_sig[lower.tri(row_sig, diag = FALSE)]
  col_sig <- diag(ncol_data)
  col_sig[lower.tri(col_sig, diag = TRUE)] <- colMeans(res$SigmaC_record)
  col_sig[upper.tri(col_sig, diag = FALSE)] <- col_sig[lower.tri(col_sig, diag = FALSE)]
  fac_series <- array(colMeans(res$F_record), dim = c(nrow_factor, ncol_factor, length(y_list)))
  # Should compute posterior mean of F_t: will be 3d array
  is_symm <- grepl(pattern = "^Sigma", x = param_names)
  num_col <- c(nrow_data, nrow_data, ncol_data, ncol_data, ncol_factor, ncol_factor, ncol_factor)
  num_row <- c(nrow_row_coef, nrow_data, nrow_col_coef, ncol_data, nrow_factor, nrow_factor, nrow_factor)
  num_matrix <- c(rep(0, 4), length(y_list), lag_factor, 0)
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
  res$param <- Reduce(bind_draws, res[rec_names])
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
  dimnames(fac_series) <- list(name_row_lag, name_col_lag, seq_along(y_list))
  res$coefficients <- list(
    row = row_coef,
    col = col_coef
  )
  res$covmat <- list(
    row = row_sig,
    col = col_sig
  )
  res$factor <- fac_series
  res$spec <- list(
    row = row_spec,
    col = col_spec,
    factor = dfm_spec
  )
  res$init <- list(
    param = param_init,
    row = row_init,
    col = col_init
  )
  res$call <- match.call()
  res$y <- y
  res$chain <- num_chains
  res$iter <- num_iter
  res$burn <- num_burn
  res$thin <- thinning
  res$p <- p
  class(res) <- "mdfmbayes"
  res
}
