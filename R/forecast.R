#' Forecasting MAR
#'
#' Forecasts matrix-valued time series.
#'
#' @param object Model object
#' @param n_ahead step to forecast
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param num_thread Number of threads
#' @param med `r lifecycle::badge("experimental")` If `TRUE`, use median of forecast draws instead of mean (default).
#' @param ... not used
#' @importFrom posterior subset_draws as_draws_matrix
#' @importFrom stats median
#' @order 1
#' @export
predict.marbayes <- function(object, n_ahead, level = .05, num_thread = 1, med = FALSE, ...) {
  fit_record <- get_bmar_records(object, TRUE)
  nrow_data <- dim(object$y)[1]
  ncol_data <- dim(object$y)[2]
  num_data <- dim(object$y)[3]
  var_names <- dimnames(object$y)
  var_names[[3]] <- 1:n_ahead
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
  num_draw <- nrow(object$param)
  y_distn <-
    do.call(cbind, pred_res) |> # (n * h) x (k * num_draw)
    split.data.frame(gl(n_ahead, nrow_data)) |>
    lapply(
      function(x) {
        split.data.frame(t(x), gl(num_draw, ncol_data)) |>
          lapply(t)
      }
    ) |>
    lapply(simplify2array)
  if (med) {
    pred_mean <-
      lapply(y_distn, function(x) apply(x, c(1, 2), median)) |>
      simplify2array()
  } else {
    pred_mean <-
      lapply(y_distn, function(x) apply(x, c(1, 2), mean)) |>
      simplify2array()
  }
  lower_quantile <-
    lapply(y_distn, function(x) apply(x, c(1, 2), quantile, probs = level / 2)) |> 
    simplify2array()
  upper_quantile <-
    lapply(y_distn, function(x) apply(x, c(1, 2), quantile, probs = 1 - level / 2)) |>
    simplify2array()
  est_se <-
    lapply(y_distn, function(x) apply(x, c(1, 2), sd)) |> 
    simplify2array()
  dimnames(pred_mean) <- var_names
  dimnames(lower_quantile) <- var_names
  dimnames(upper_quantile) <- var_names
  dimnames(est_se) <- var_names
  res <- list(
    forecast = pred_mean,
    se = est_se,
    lower = lower_quantile,
    upper = upper_quantile,
    lower_joint = lower_quantile,
    upper_joint = upper_quantile,
    y = object$y
  )
  class(res) <- c("predmarbayes", "predmar")
  res
}