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
#' @param verbose Progress log
#' @param num_thread Number of threads
#' 
#' @references
#' Chan, J. C. C. & Qi, Y. (2024). Large Bayesian Tensor VARs with Stochastic Volatility. arXiv.
#' 
#' @importFrom Matrix bdiag
#' @importFrom posterior as_draws_df bind_draws summarise_draws
#' @order 1
#' @export
mar_bayes <- function(y,
                      p = 1,
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
  param_init <- get_bmar_init(
    num_chains = num_chains,
    nrow_data = nrow_data,
    ncol_data = ncol_data,
    nrow_row_coef = nrow_row_coef,
    nrow_col_coef = nrow_col_coef
  )
  row_prior <- validate_bmar_prior(row_spec)
  col_prior <- validate_bmar_prior(col_spec)
  row_init <- switch(
    row_spec$prior,
    "Minnesota" = { get_mat_minn_init(num_chains) },
    "Horseshoe" = {
      get_mat_hs_init(
        num_chains = num_chains,
        nrow_coef = nrow_row_coef
      )
    },
    stop("Wrong row prior")
  )
  col_init <- switch(
    col_spec$prior,
    "Minnesota" = { get_mat_minn_init(num_chains) },
    "Horseshoe" = {
      get_mat_hs_init(
        num_chains = num_chains,
        nrow_coef = nrow_col_coef
      )
    },
    stop("Wrong column prior")
  )
  row_prior_type <- get_prior_id(row_spec$prior)
  col_prior_type <- get_prior_id(col_spec$prior)
  res <- estimate_bmar_mniw(
    num_chains = num_chains, num_iter = num_iter, num_burn = num_burn, thin = thinning,
    x = design, y = response,
    param_coef_sig = param_prior, coef_sig_init = param_init,
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
  is_symm <- grepl(pattern = "^Sigma", x = param_names)
  num_col <- c(nrow_data, nrow_data, ncol_data, ncol_data)
  res[rec_names] <- lapply(
    seq_along(res[rec_names]),
    function(id) {
      split_matrix_chain(res[rec_names][[id]], chain = num_chains, varname = param_names[id], lag = ifelse(is_symm[id], 1, p), num_col = num_col[id], is_symm = is_symm[id])
    }
  )
  res[rec_names] <- lapply(res[rec_names], as_draws_df)
  res$param <- bind_draws(
    res$A_record,
    res$SigmaR_record,
    res$B_record,
    res$SigmaC_record
  )
  res[rec_names] <- NULL
  res$param_names <- param_names
  rownames(row_coef) <- lapply(
    1:p,
    function(lag) paste(var_names[[1]], lag, sep = "_")
  ) |>
    unlist()
  colnames(row_coef) <- var_names[[1]]
  rownames(row_sig) <- var_names[[1]]
  colnames(row_sig) <- var_names[[1]]
  rownames(col_coef) <- lapply(
    1:p,
    function(lag) paste(var_names[[2]], lag, sep = "_")
  ) |>
    unlist()
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
  class(res) <- "marbayes"
  res
}
