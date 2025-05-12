#' Forecasting MAR
#' 
#' Forecasts matrix-valued time series.
#' 
#' @param object Model object
#' @param n_ahead step to forecast
#' @param num_thread Number of threads
#' @param med `r lifecycle::badge("experimental")` If `TRUE`, use median of forecast draws instead of mean (default).
#' @param ... not used
#' @importFrom posterior subset_draws as_draws_matrix
#' @importFrom stats median
#' @order 1
#' @export
predict.marbayes <- function(object, n_ahead, num_thread = 1, med = FALSE, ...) {
  fit_record <- get_bmar_records(object, TRUE)
  nrow_data <- dim(object$y)[1]
  ncol_data <- dim(object$y)[2]
  num_data <- dim(object$y)[3]
  y_list <- lapply(seq_len(num_data), function(x) object$y[, , x])
  pred_res <- forecast_bmar_mniw(
    num_chains = object$chain,
    lag = object$p,
    step = n_ahead,
    response_mat = do.call(rbind, y_list),
    num_data = length(y_list),
    fit_record = fit_record,
    seed_chain = sample.int(.Machine$integer.max, size = object$chain),
    nthreads = num_thread
  )
  return(pred_res)
  y_distn <- do.call(cbind, pred_res)
  # num_draw <- nrow(object$param)
  # y_distn <-
  #   pred_res |>
  #   unlist() |>
  #   array(dim = c(nrow_data, ncol_data, n_ahead, num_draw))
}