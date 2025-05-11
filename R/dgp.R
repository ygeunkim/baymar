#' Generate Matrix Time Series following MAR(p)
#'
#' This function generates MAR(p) time series array.
#'
#' @param num_sim Number to generated process
#' @param num_burn Number of burn-in
#' @param p MAR lag
#' @param row_coef Row MAR coefficient (np x n).
#' @param col_coef Column MAR coefficient (kp x k).
#' @param row_sig Row covariance of innovation matrix. By default, identity matrix.
#' @param col_sig Column covariance of innovation matrix. By default, identity matrix.
#' @param init Initial \eqn{Y_1, \ldots, Y_p} array or list to simulate MAR model. By default, zero.
#'
#' @return n x k x (num_sim - num_burn) array.
#'
#' @importFrom Matrix bdiag
#' @export
sim_mar <- function(num_sim,
                    num_burn,
                    p = 1,
                    row_coef,
                    col_coef,
                    row_sig = diag(ncol(row_coef)),
                    col_sig = diag(ncol(col_coef)),
                    init = array(0L, dim = c(ncol(row_coef), ncol(col_coef), p))) {
  validate_coef_sig(coef = row_coef, sig = row_sig)
  validate_coef_sig(coef = col_coef, sig = col_sig)
  if (is.array(init)) {
    if (length(dim(init)) != 3) {
      stop("Array 'init' should have three margins.")
    }
    if (dim(init)[3] != p) {
      stop("Array 'init' should be nrow x ncol x p.")
    }
    init_list <- lapply(seq_len(p), function(id) init[, , id])
    init_mat <- bdiag(init_list[1:p])
  } else if (is.list(init)) {
    if (length(init) != p) {
      stop("Length of list 'init' should be p.")
    }
    num_row <- nrow(init[[1]])
    num_col <- ncol(init[[1]])
    if (any(sapply(init, nrow) != num_row)) {
      stop(sprintf("Every matrix of 'init' should be %d", num_row))
    }
    if (any(sapply(init, ncol) != num_col)) {
      stop(sprintf("Every matrix of 'init' should be %d", num_col))
    }
    init_mat <- bdiag(init[1:p])
  } else {
    stop("'init' should be list or 3d array.")
  }
  sim_mar_export(
    num_sim = num_sim,
    num_burn = num_burn,
    init = init_mat,
    row_coef = row_coef,
    col_coef = col_coef,
    row_sig = row_sig,
    col_sig = col_sig
  ) |> 
  simplify2array()
}
