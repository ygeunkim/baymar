#' Forecasting MAR
#'
#' Forecasts matrix-valued time series.
#'
#' @param object Model object
#' @param n_ahead step to forecast
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param newxreg New values for exogenous matrices.
#' @param num_thread Number of threads
#' @param med `r lifecycle::badge("experimental")` If `TRUE`, use median of forecast draws instead of mean (default).
#' @param ... not used
#' @importFrom posterior subset_draws as_draws_matrix
#' @importFrom stats median sd quantile
#' @order 1
#' @export
predict.marbayes <- function(object, n_ahead, level = .05, newxreg, num_thread = 1, med = FALSE, ...) {
  fit_record <- get_bmar_records(object, TRUE)
  nrow_data <- dim(object$y)[1]
  ncol_data <- dim(object$y)[2]
  num_data <- dim(object$y)[3]
  var_names <- dimnames(object$y)
  y_list <- lapply(seq_len(num_data), function(x) object$y[, , x])
  is_insample <- missing(n_ahead) || is.null(n_ahead)
  if (is_insample) {
    n_ahead <- length(y_list) - object$p
  }
  var_names[[3]] <- 1:n_ahead
  nrow_factor <- 0
  ncol_factor <- 0
  factor_lag <- 0
  if ("factor" %in% names(object$spec)) {
    nrow_factor <- object$spec$factor$nrow_factor
    ncol_factor <- object$spec$factor$ncol_factor
    factor_lag <- ifelse(is_insample, 0, object$spec$factor$lag)
  }
  if (!is.null(eval.parent(object$call$exogen))) {
    exogen_list <- lapply(seq_len(dim(object$exogen_data)[3]), function(x) object$exogen_data[, , x])
    if (is_insample) {
      exogen_list <- do.call(rbind, exogen_list)
    } else {
      newxreg_list <- validate_newxmat(newxreg = newxreg, n_ahead = n_ahead)
      exogen_list <- rbind(
        do.call(rbind, tail(exogen_list, object$s)),
        do.call(rbind, newxreg_list)
      )
    }
    pred_res <- forecast_bmarx_mniw(
      num_chains = object$chain,
      lag = object$p,
      step = n_ahead,
      response_mat = do.call(rbind, y_list),
      num_data = length(y_list),
      nrow_factor = nrow_factor,
      ncol_factor = ncol_factor,
      factor_lag = factor_lag,
      fit_record = fit_record,
      seed_chain = sample.int(.Machine$integer.max, size = object$chain),
      exogen = exogen_list,
      exogen_lag = object$s,
      nthreads = num_thread,
      insample = is_insample
    )
  } else {
    pred_res <- forecast_bmar_mniw(
      num_chains = object$chain,
      lag = object$p,
      step = n_ahead,
      response_mat = do.call(rbind, y_list),
      num_data = length(y_list),
      nrow_factor = nrow_factor,
      ncol_factor = ncol_factor,
      factor_lag = factor_lag,
      fit_record = fit_record,
      seed_chain = sample.int(.Machine$integer.max, size = object$chain),
      nthreads = num_thread,
      insample = is_insample
    )
  }
  num_draw <- nrow(object$param)
  # y_distn <-
  #   do.call(cbind, pred_res) |> # (n * h) x (k * num_draw)
  #   split.data.frame(gl(n_ahead, nrow_data)) |>
  #   lapply(
  #     function(x) {
  #       split.data.frame(t(x), gl(num_draw, ncol_data)) |>
  #         lapply(t)
  #     }
  #   ) |>
  #   lapply(simplify2array)
  y_distn <- process_mar_forecast_draws(
    x = pred_res,
    n_ahead = n_ahead,
    nrow_data = nrow_data,
    ncol_data = ncol_data,
    num_draw = num_draw
  )
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

#' Forecasting MDFM
#'
#' Forecasts matrix dynamic factor model.
#'
#' @param object Model object
#' @param n_ahead step to forecast
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param num_thread Number of threads
#' @param med `r lifecycle::badge("experimental")` If `TRUE`, use median of forecast draws instead of mean (default).
#' @param ... not used
#' @order 1
#' @export
predict.mdfmbayes <- function(object, n_ahead, level = .05, num_thread = 1, med = FALSE, ...) {
  fit_record <- get_bmar_records(object, TRUE)
  nrow_data <- dim(object$y)[1]
  ncol_data <- dim(object$y)[2]
  num_data <- dim(object$y)[3]
  y_list <- lapply(seq_len(num_data), function(x) object$y[, , x])
  is_insample <- missing(n_ahead) || is.null(n_ahead)
  if (is_insample) {
    # n_ahead <- length(y_list) - object$spec$factor$lag
    n_ahead <- length(y_list)
  }
  var_names <- dimnames(object$y)
  var_names[[3]] <- 1:n_ahead
  nrow_factor <- object$spec$factor$nrow_factor
  ncol_factor <- object$spec$factor$ncol_factor
  factor_lag <- object$spec$factor$lag
  pred_res <- forecast_bdfm_mniw(
    num_chains = object$chain,
    step = n_ahead,
    nrow_factor = nrow_factor,
    ncol_factor = ncol_factor,
    factor_lag = factor_lag,
    fit_record = fit_record,
    seed_chain = sample.int(.Machine$integer.max, size = object$chain),
    nthreads = num_thread,
    insample = is_insample
  )
  num_draw <- nrow(object$param)
  y_distn <- process_mar_forecast_draws(
    x = pred_res,
    n_ahead = n_ahead,
    nrow_data = nrow_data,
    ncol_data = ncol_data,
    num_draw = num_draw
  )
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

#' Pseudo out-of-sample Forecasting based on Rolling Window
#' 
#' @param object Model object
#' @param n_ahead Step to forecast in rolling window scheme
#' @param y_test Test data to be compared.
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param newxreg New values for exogenous variables.
#' @param num_thread `r lifecycle::badge("experimental")` Number of threads
#' @param med `r lifecycle::badge("experimental")` If `TRUE`, use median of forecast draws instead of mean (default).
#' @param lpl `r lifecycle::badge("experimental")` Compute log-predictive likelihood (LPL). By default, `FALSE`.
#' @param mcmc `r lifecycle::badge("experimental")` If `TRUE`, run new MCMC in new windows. By default, `TRUE`.
#' @param use_fit `r lifecycle::badge("experimental")` Use `object` result for the first window. By default, `TRUE`.
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @param ... Additional arguments
#' @exportS3Method bvhar::forecast_roll
forecast_roll.marbayes <- function(object, n_ahead, y_test,
                                   level = .05,
                                   newxreg,
                                   num_thread = 1,
                                   med = FALSE,
                                   lpl = FALSE,
                                   mcmc = TRUE,
                                   use_fit = TRUE,
                                   verbose = FALSE, ...) {
  fit_record <- list()
  if (use_fit) {
    fit_record <- get_bmar_records(object, TRUE)
  }
  nrow_data <- dim(object$y)[1]
  ncol_data <- dim(object$y)[2]
  num_data <- dim(object$y)[3]
  nrow_row_coef <- nrow_data * object$p
  nrow_col_coef <- ncol_data * object$p
  is_full <- is.numeric(y_test) && length(y_test) == 1
  if (is_full) {
    num_test <- y_test
    num_data <- num_data - num_test
    y_test <- object$y[, , (num_data + 1):(num_data + num_test)]
    object$y <- object$y[, , seq_len(num_data)]
    use_fit <- FALSE
  }
  num_test <- dim(y_test)[3]
  y_list <- lapply(seq_len(num_data), function(x) object$y[, , x])
  y_test_list <- lapply(seq_len(num_test), function(x) y_test[, , x])
  var_names <- dimnames(object$y)
  var_names[[3]] <- n_ahead:length(y_test_list)
  param_prior <- validate_bmar_row_spec(
    y = object$y,
    p = object$p,
    bayes_spec = object$spec$row,
    nrow_data = nrow_data,
    ncol_data = ncol_data,
    nrow_row_coef = nrow_row_coef
  )
  param_prior <- append(
    param_prior,
    validate_bmar_col_spec(
      y = object$y,
      p = object$p,
      bayes_spec = object$spec$col,
      nrow_data = nrow_data,
      ncol_data = ncol_data,
      nrow_col_coef = nrow_col_coef
    )
  )
  row_prior <- validate_bmar_prior(object$spec$row)
  col_prior <- validate_bmar_prior(object$spec$col)
  num_horizon <- length(y_test_list) - n_ahead + 1
  nrow_factor <- 0
  ncol_factor <- 0
  factor_lag <- 0
  row_factor_prior <- list()
  col_factor_prior <- list()
  row_factor_prior_type <- 0
  col_factor_prior_type <- 0
  row_factor_init <- list()
  col_factor_init <- list()
  if ("factor" %in% names(object$spec)) {
    nrow_factor <- object$spec$factor$nrow_factor
    ncol_factor <- object$spec$factor$ncol_factor
    factor_lag <- object$spec$factor$lag
    row_factor_prior <- validate_bmar_prior(object$spec$factor_row)
    col_factor_prior <- validate_bmar_prior(object$spec$factor_col)
    row_factor_prior_type <- get_prior_id(object$spec$factor_row$prior)
    col_factor_prior_type <- get_prior_id(object$spec$factor_col$prior)
    row_factor_init <- object$init$factor_row
    col_factor_init <- object$init$factor_col
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
    param_prior <- append(
      param_prior,
      list(
        nrow_factor = nrow_factor,
        ncol_factor = ncol_factor,
        size_factor = nrow_factor * ncol_factor,
        lag = factor_lag,
        shape = object$spec$factor$arsig$shape,
        scale = object$spec$factor$arsig$scale
      )
    )
  }
  is_exogen <- !is.null(eval.parent(object$call$exogen))
  if (is_exogen) {
    if (is_full) {
      newxreg <- object$exogen_data[, , (num_data + 1):(num_data + num_test)]
      object$exogen_data <- object$exogen_data[, , seq_len(num_data)]
    }
    newxreg_list <- validate_newxmat(newxreg = newxreg, n_ahead = num_test)
    exogen_row_prior <- validate_bmar_prior(object$spec$exogen_row)
    exogen_col_prior <- validate_bmar_prior(object$spec$exogen_col)
    param_prior <- validate_bmarx_rowspec(
      param_prior = param_prior,
      x = object$exogen_data,
      s = object$s,
      bayes_spec = object$spec$exogen_row,
      nrow_exogen = dim(object$exogen_data)[1],
      ncol_exogen = dim(object$exogen_data)[2],
      nrow_exogen_row_coef = (object$s + 1) * dim(object$exogen_data)[1]
    )
    param_prior <- validate_bmarx_colspec(
      param_prior = param_prior,
      x = object$exogen_data,
      s = object$s,
      bayes_spec = object$spec$exogen_col,
      nrow_exogen = dim(object$exogen_data)[1],
      ncol_exogen = dim(object$exogen_data)[2],
      nrow_exogen_col_coef = (object$s + 1) * dim(object$exogen_data)[2]
    )
    exogen_list <- lapply(seq_len(dim(object$exogen_data)[3]), function(x) object$exogen_data[, , x])
    pred_res <- roll_bmarx_mniw(
      y = do.call(rbind, y_list),
      lag = object$p,
      num_data = length(y_list),
      num_chains = object$chain,
      num_iter = object$iter,
      num_burn = object$burn,
      thin = object$thin,
      fit_record = fit_record,
      run_mcmc = mcmc,
      param_coef_sig = param_prior, coef_sig_init = object$init$param,
      row_prior = row_prior, row_init = object$init$row, row_prior_type = get_prior_id(object$spec$row$prior),
      col_prior = col_prior, col_init = object$init$col, col_prior_type = get_prior_id(object$spec$col$prior),
      factor_row_prior = row_factor_prior, factor_row_init = row_factor_init, factor_row_prior_type = row_factor_prior_type, factor_rows = nrow_factor,
      factor_col_prior = col_factor_prior, factor_col_init = col_factor_init, factor_col_prior_type = col_factor_prior_type, factor_cols = ncol_factor,
      factor_lag = factor_lag,
      step = n_ahead, y_test = do.call(rbind, y_test_list), get_lpl = lpl, use_fit = use_fit,
      seed_chain = sample.int(.Machine$integer.max, size = object$chain * num_horizon) |> matrix(ncol = object$chain),
      seed_forecast = sample.int(.Machine$integer.max, size = object$chain),
      display_progress = verbose,
      nthreads = num_thread,
      exogen = rbind(
        do.call(rbind, exogen_list),
        do.call(rbind, newxreg_list)
      ),
      exogen_lag = object$s,
      exogen_row_prior = exogen_row_prior, exogen_row_init = object$init$exogen_row, exogen_row_prior_type = get_prior_id(object$spec$exogen_row$prior),
      exogen_col_prior = exogen_col_prior, exogen_col_init = object$init$exogen_col, exogen_col_prior_type = get_prior_id(object$spec$exogen_col$prior)
    )
  } else {
    pred_res <- roll_bmar_mniw(
      y = do.call(rbind, y_list),
      lag = object$p,
      num_data = length(y_list),
      num_chains = object$chain,
      num_iter = object$iter,
      num_burn = object$burn,
      thin = object$thin,
      fit_record = fit_record,
      run_mcmc = mcmc,
      param_coef_sig = param_prior, coef_sig_init = object$init$param,
      row_prior = row_prior, row_init = object$init$row, row_prior_type = get_prior_id(object$spec$row$prior),
      col_prior = col_prior, col_init = object$init$col, col_prior_type = get_prior_id(object$spec$col$prior),
      factor_row_prior = row_factor_prior, factor_row_init = row_factor_init, factor_row_prior_type = row_factor_prior_type, factor_rows = nrow_factor,
      factor_col_prior = col_factor_prior, factor_col_init = col_factor_init, factor_col_prior_type = col_factor_prior_type, factor_cols = ncol_factor,
      factor_lag = factor_lag,
      step = n_ahead, y_test = do.call(rbind, y_test_list), get_lpl = lpl, use_fit = use_fit,
      seed_chain = sample.int(.Machine$integer.max, size = object$chain * num_horizon) |> matrix(ncol = object$chain),
      seed_forecast = sample.int(.Machine$integer.max, size = object$chain),
      display_progress = verbose,
      nthreads = num_thread
    )
  }
  num_draw <- nrow(object$param)
  if (lpl) {
    if (med) {
      lpl_val <- apply(pred_res$lpl, 1, median)
    } else {
      lpl_val <- rowMeans(pred_res$lpl)
    }
    pred_res$lpl <- NULL
  }
  y_distn <-
    lapply(
      pred_res$forecast,
      process_mar_ourforecast_draws,
      n_ahead = n_ahead,
      ncol_data = ncol_data,
      num_draw = num_draw
    )
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
    eval_id = n_ahead:length(y_test_list),
    y = object$y
  )
  if (lpl) {
    res$lpl <- lpl_val
  }
  class(res) <- c("predmarbayes_roll", "predmarcv")
  res
}

#' Pseudo out-of-sample Forecasting for MDFM based on Rolling Window
#'
#' @param object Model object
#' @param n_ahead Step to forecast in rolling window scheme
#' @param y_test Test data to be compared.
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param newxreg Not used.
#' @param num_thread `r lifecycle::badge("experimental")` Number of threads
#' @param med `r lifecycle::badge("experimental")` If `TRUE`, use median of forecast draws instead of mean (default).
#' @param lpl `r lifecycle::badge("experimental")` Compute log-predictive likelihood (LPL). By default, `FALSE`.
#' @param mcmc `r lifecycle::badge("experimental")` If `TRUE`, run new MCMC in new windows. By default, `TRUE`.
#' @param use_fit `r lifecycle::badge("experimental")` Use `object` result for the first window. By default, `TRUE`.
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @param ... Additional arguments
#' @exportS3Method bvhar::forecast_roll
forecast_roll.mdfmbayes <- function(object, n_ahead, y_test,
                                    level = .05,
                                    newxreg = NULL,
                                    num_thread = 1,
                                    med = FALSE,
                                    lpl = FALSE,
                                    mcmc = TRUE,
                                    use_fit = TRUE,
                                    verbose = FALSE, ...) {
  fit_record <- list()
  if (use_fit) {
    fit_record <- get_bmar_records(object, TRUE)
  }
  nrow_data <- dim(object$y)[1]
  ncol_data <- dim(object$y)[2]
  num_data <- dim(object$y)[3]
  nrow_factor <- object$spec$factor$nrow_factor
  ncol_factor <- object$spec$factor$ncol_factor
  factor_lag <- object$spec$factor$lag
  # nrow_row_coef <- nrow_data * object$p
  # nrow_col_coef <- ncol_data * object$p
  is_full <- is.numeric(y_test) && length(y_test) == 1
  if (is_full) {
    num_test <- y_test
    num_data <- num_data - num_test
    y_test <- object$y[, , (num_data + 1):(num_data + num_test)]
    object$y <- object$y[, , seq_len(num_data)]
    use_fit <- FALSE
  }
  num_test <- dim(y_test)[3]
  y_list <- lapply(seq_len(num_data), function(x) object$y[, , x])
  y_test_list <- lapply(seq_len(num_test), function(x) y_test[, , x])
  var_names <- dimnames(object$y)
  var_names[[3]] <- n_ahead:length(y_test_list)
  param_prior <- validate_bmar_row_spec(
    y = object$y,
    p = object$p,
    bayes_spec = object$spec$row,
    nrow_data = nrow_data,
    ncol_data = ncol_data,
    nrow_row_coef = nrow_factor
  )
  param_prior <- append(
    param_prior,
    validate_bmar_col_spec(
      y = object$y,
      p = object$p,
      bayes_spec = object$spec$col,
      nrow_data = nrow_data,
      ncol_data = ncol_data,
      nrow_col_coef = ncol_factor
    )
  )
  param_prior <- append(
    param_prior,
    list(
      nrow_factor = nrow_factor,
      ncol_factor = ncol_factor,
      size_factor = nrow_factor * ncol_factor,
      lag = factor_lag,
      shape = object$spec$factor$arsig$shape,
      scale = object$spec$factor$arsig$scale
    )
  )
  param_prior$row_prior_prec <- rep(1, nrow_factor)
  param_prior$col_prior_prec <- rep(1, ncol_factor)
  row_prior <- validate_bmar_prior(object$spec$row)
  col_prior <- validate_bmar_prior(object$spec$col)
  num_horizon <- length(y_test_list) - n_ahead + 1
  pred_res <- roll_bdfm_mniw(
    y = do.call(rbind, y_list),
    num_data = length(y_list),
    num_chains = object$chain,
    num_iter = object$iter,
    num_burn = object$burn,
    thin = object$thin,
    fit_record = fit_record,
    run_mcmc = mcmc,
    param_coef_sig = param_prior, coef_sig_init = object$init$param,
    row_prior = row_prior, row_init = object$init$row, row_prior_type = get_prior_id(object$spec$row$prior),
    col_prior = col_prior, col_init = object$init$col, col_prior_type = get_prior_id(object$spec$col$prior),
    factor_rows = nrow_factor, factor_cols = ncol_factor, factor_lag = factor_lag,
    step = n_ahead, y_test = do.call(rbind, y_test_list), get_lpl = lpl, use_fit = use_fit,
    seed_chain = sample.int(.Machine$integer.max, size = object$chain * num_horizon) |> matrix(ncol = object$chain),
    seed_forecast = sample.int(.Machine$integer.max, size = object$chain),
    display_progress = verbose,
    nthreads = num_thread
  )
  num_draw <- nrow(object$param)
  if (lpl) {
    if (med) {
      lpl_val <- apply(pred_res$lpl, 1, median)
    } else {
      lpl_val <- rowMeans(pred_res$lpl)
    }
    pred_res$lpl <- NULL
  }
  y_distn <-
    lapply(
      pred_res$forecast,
      process_mar_ourforecast_draws,
      n_ahead = n_ahead,
      ncol_data = ncol_data,
      num_draw = num_draw
    )
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
    eval_id = n_ahead:length(y_test_list),
    y = object$y
  )
  if (lpl) {
    res$lpl <- lpl_val
  }
  class(res) <- c("predmarbayes_roll", "predmarcv")
  res
}

#' Pseudo out-of-sample Forecasting based on Expanding Window
#' 
#' @param object Model object
#' @param n_ahead Step to forecast in rolling window scheme
#' @param y_test Test data to be compared.
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param newxreg New values for exogenous variables.
#' @param num_thread `r lifecycle::badge("experimental")` Number of threads
#' @param med `r lifecycle::badge("experimental")` If `TRUE`, use median of forecast draws instead of mean (default).
#' @param lpl `r lifecycle::badge("experimental")` Compute log-predictive likelihood (LPL). By default, `FALSE`.
#' @param mcmc `r lifecycle::badge("experimental")` If `TRUE`, run new MCMC in new windows. By default, `TRUE`.
#' @param use_fit `r lifecycle::badge("experimental")` Use `object` result for the first window. By default, `TRUE`.
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @param ... Additional arguments
#' @exportS3Method bvhar::forecast_expand
forecast_expand.marbayes <- function(object, n_ahead, y_test,
                                     level = .05,
                                     newxreg,
                                     num_thread = 1,
                                     med = FALSE,
                                     lpl = FALSE,
                                     mcmc = TRUE,
                                     use_fit = TRUE,
                                     verbose = FALSE, ...) {
  fit_record <- list()
  if (use_fit) {
    fit_record <- get_bmar_records(object, TRUE)
  }
  nrow_data <- dim(object$y)[1]
  ncol_data <- dim(object$y)[2]
  num_data <- dim(object$y)[3]
  nrow_row_coef <- nrow_data * object$p
  nrow_col_coef <- ncol_data * object$p
  is_full <- is.numeric(y_test) && length(y_test) == 1
  if (is_full) {
    num_test <- y_test
    num_data <- num_data - num_test
    y_test <- object$y[, , (num_data + 1):(num_data + num_test)]
    object$y <- object$y[, , seq_len(num_data)]
    use_fit <- FALSE
  }
  num_test <- dim(y_test)[3]
  y_list <- lapply(seq_len(num_data), function(x) object$y[, , x])
  y_test_list <- lapply(seq_len(num_test), function(x) y_test[, , x])
  var_names <- dimnames(object$y)
  var_names[[3]] <- n_ahead:length(y_test_list)
  param_prior <- validate_bmar_row_spec(
    y = object$y,
    p = object$p,
    bayes_spec = object$spec$row,
    nrow_data = nrow_data,
    ncol_data = ncol_data,
    nrow_row_coef = nrow_row_coef
  )
  param_prior <- append(
    param_prior,
    validate_bmar_col_spec(
      y = object$y,
      p = object$p,
      bayes_spec = object$spec$col,
      nrow_data = nrow_data,
      ncol_data = ncol_data,
      nrow_col_coef = nrow_col_coef
    )
  )
  row_prior <- validate_bmar_prior(object$spec$row)
  col_prior <- validate_bmar_prior(object$spec$col)
  num_horizon <- length(y_test_list) - n_ahead + 1
  nrow_factor <- 0
  ncol_factor <- 0
  factor_lag <- 0
  row_factor_prior <- list()
  col_factor_prior <- list()
  row_factor_prior_type <- 0
  col_factor_prior_type <- 0
  row_factor_init <- list()
  col_factor_init <- list()
  if ("factor" %in% names(object$spec)) {
    nrow_factor <- object$spec$factor$nrow_factor
    ncol_factor <- object$spec$factor$ncol_factor
    factor_lag <- object$spec$factor$lag
    row_factor_prior <- validate_bmar_prior(object$spec$factor_row)
    col_factor_prior <- validate_bmar_prior(object$spec$factor_col)
    row_factor_prior_type <- get_prior_id(object$spec$factor_row$prior)
    col_factor_prior_type <- get_prior_id(object$spec$factor_col$prior)
    row_factor_init <- object$init$factor_row
    col_factor_init <- object$init$factor_col
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
    param_prior <- append(
      param_prior,
      list(
        nrow_factor = nrow_factor,
        ncol_factor = ncol_factor,
        size_factor = nrow_factor * ncol_factor,
        lag = factor_lag,
        shape = object$spec$factor$arsig$shape,
        scale = object$spec$factor$arsig$scale
      )
    )
  }
  is_exogen <- !is.null(eval.parent(object$call$exogen))
  if (is_exogen) {
    if (is_full) {
      newxreg <- object$exogen_data[, , (num_data + 1):(num_data + num_test)]
      object$exogen_data <- object$exogen_data[, , seq_len(num_data)]
    }
    newxreg_list <- validate_newxmat(newxreg = newxreg, n_ahead = num_test)
    exogen_row_prior <- validate_bmar_prior(object$spec$exogen_row)
    exogen_col_prior <- validate_bmar_prior(object$spec$exogen_col)
    param_prior <- validate_bmarx_rowspec(
      param_prior = param_prior,
      x = object$exogen_data,
      s = object$s,
      bayes_spec = object$spec$exogen_row,
      nrow_exogen = dim(object$exogen_data)[1],
      ncol_exogen = dim(object$exogen_data)[2],
      nrow_exogen_row_coef = (object$s + 1) * dim(object$exogen_data)[1]
    )
    param_prior <- validate_bmarx_colspec(
      param_prior = param_prior,
      x = object$exogen_data,
      s = object$s,
      bayes_spec = object$spec$exogen_col,
      nrow_exogen = dim(object$exogen_data)[1],
      ncol_exogen = dim(object$exogen_data)[2],
      nrow_exogen_col_coef = (object$s + 1) * dim(object$exogen_data)[2]
    )
    exogen_list <- lapply(seq_len(dim(object$exogen_data)[3]), function(x) object$exogen_data[, , x])
    pred_res <- expand_bmarx_mniw(
      y = do.call(rbind, y_list),
      lag = object$p,
      num_data = length(y_list),
      num_chains = object$chain,
      num_iter = object$iter,
      num_burn = object$burn,
      thin = object$thin,
      fit_record = fit_record,
      run_mcmc = mcmc,
      param_coef_sig = param_prior, coef_sig_init = object$init$param,
      row_prior = row_prior, row_init = object$init$row, row_prior_type = get_prior_id(object$spec$row$prior),
      col_prior = col_prior, col_init = object$init$col, col_prior_type = get_prior_id(object$spec$col$prior),
      factor_row_prior = row_factor_prior, factor_row_init = row_factor_init, factor_row_prior_type = row_factor_prior_type, factor_rows = nrow_factor,
      factor_col_prior = col_factor_prior, factor_col_init = col_factor_init, factor_col_prior_type = col_factor_prior_type, factor_cols = ncol_factor,
      factor_lag = factor_lag,
      step = n_ahead, y_test = do.call(rbind, y_test_list), get_lpl = lpl, use_fit = use_fit,
      seed_chain = sample.int(.Machine$integer.max, size = object$chain * num_horizon) |> matrix(ncol = object$chain),
      seed_forecast = sample.int(.Machine$integer.max, size = object$chain),
      display_progress = verbose,
      nthreads = num_thread,
      exogen = rbind(
        do.call(rbind, exogen_list),
        do.call(rbind, newxreg_list)
      ),
      exogen_lag = object$s,
      exogen_row_prior = exogen_row_prior, exogen_row_init = object$init$exogen_row, exogen_row_prior_type = get_prior_id(object$spec$exogen_row$prior),
      exogen_col_prior = exogen_col_prior, exogen_col_init = object$init$exogen_col, exogen_col_prior_type = get_prior_id(object$spec$exogen_col$prior)
    )
  } else {
    pred_res <- expand_bmar_mniw(
      y = do.call(rbind, y_list),
      lag = object$p,
      num_data = length(y_list),
      num_chains = object$chain,
      num_iter = object$iter,
      num_burn = object$burn,
      thin = object$thin,
      fit_record = fit_record,
      run_mcmc = mcmc,
      param_coef_sig = param_prior, coef_sig_init = object$init$param,
      row_prior = row_prior, row_init = object$init$row, row_prior_type = get_prior_id(object$spec$row$prior),
      col_prior = col_prior, col_init = object$init$col, col_prior_type = get_prior_id(object$spec$col$prior),
      factor_row_prior = row_factor_prior, factor_row_init = row_factor_init, factor_row_prior_type = row_factor_prior_type, factor_rows = nrow_factor,
      factor_col_prior = col_factor_prior, factor_col_init = col_factor_init, factor_col_prior_type = col_factor_prior_type, factor_cols = ncol_factor,
      factor_lag = factor_lag,
      step = n_ahead, y_test = do.call(rbind, y_test_list), get_lpl = lpl, use_fit = use_fit,
      seed_chain = sample.int(.Machine$integer.max, size = object$chain * num_horizon) |> matrix(ncol = object$chain),
      seed_forecast = sample.int(.Machine$integer.max, size = object$chain),
      display_progress = verbose,
      nthreads = num_thread
    )
  }
  num_draw <- nrow(object$param)
  if (lpl) {
    if (med) {
      lpl_val <- apply(pred_res$lpl, 1, median)
    } else {
      lpl_val <- rowMeans(pred_res$lpl)
    }
    pred_res$lpl <- NULL
  }
  y_distn <-
    lapply(
      pred_res$forecast,
      process_mar_ourforecast_draws,
      n_ahead = n_ahead,
      ncol_data = ncol_data,
      num_draw = num_draw
    )
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
    eval_id = n_ahead:length(y_test_list),
    y = object$y
  )
  if (lpl) {
    res$lpl <- lpl_val
  }
  class(res) <- c("predmarbayes_expand", "predmarcv")
  res
}

#' Pseudo out-of-sample Forecasting for MDFM based on Expanding Window
#'
#' @param object Model object
#' @param n_ahead Step to forecast in rolling window scheme
#' @param y_test Test data to be compared.
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param newxreg Not used.
#' @param num_thread `r lifecycle::badge("experimental")` Number of threads
#' @param med `r lifecycle::badge("experimental")` If `TRUE`, use median of forecast draws instead of mean (default).
#' @param lpl `r lifecycle::badge("experimental")` Compute log-predictive likelihood (LPL). By default, `FALSE`.
#' @param mcmc `r lifecycle::badge("experimental")` If `TRUE`, run new MCMC in new windows. By default, `TRUE`.
#' @param use_fit `r lifecycle::badge("experimental")` Use `object` result for the first window. By default, `TRUE`.
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @param ... Additional arguments
#' @exportS3Method bvhar::forecast_expand
forecast_expand.mdfmbayes <- function(object, n_ahead, y_test,
                                      level = .05,
                                      newxreg = NULL,
                                      num_thread = 1,
                                      med = FALSE,
                                      lpl = FALSE,
                                      mcmc = TRUE,
                                      use_fit = TRUE,
                                      verbose = FALSE, ...) {
  fit_record <- list()
  if (use_fit) {
    fit_record <- get_bmar_records(object, TRUE)
  }
  nrow_data <- dim(object$y)[1]
  ncol_data <- dim(object$y)[2]
  num_data <- dim(object$y)[3]
  nrow_factor <- object$spec$factor$nrow_factor
  ncol_factor <- object$spec$factor$ncol_factor
  factor_lag <- object$spec$factor$lag
  # nrow_row_coef <- nrow_data * object$p
  # nrow_col_coef <- ncol_data * object$p
  is_full <- is.numeric(y_test) && length(y_test) == 1
  if (is_full) {
    num_test <- y_test
    num_data <- num_data - num_test
    y_test <- object$y[, , (num_data + 1):(num_data + num_test)]
    object$y <- object$y[, , seq_len(num_data)]
    use_fit <- FALSE
  }
  num_test <- dim(y_test)[3]
  y_list <- lapply(seq_len(num_data), function(x) object$y[, , x])
  y_test_list <- lapply(seq_len(num_test), function(x) y_test[, , x])
  var_names <- dimnames(object$y)
  var_names[[3]] <- n_ahead:length(y_test_list)
  param_prior <- validate_bmar_row_spec(
    y = object$y,
    p = object$p,
    bayes_spec = object$spec$row,
    nrow_data = nrow_data,
    ncol_data = ncol_data,
    nrow_row_coef = nrow_factor
  )
  param_prior <- append(
    param_prior,
    validate_bmar_col_spec(
      y = object$y,
      p = object$p,
      bayes_spec = object$spec$col,
      nrow_data = nrow_data,
      ncol_data = ncol_data,
      nrow_col_coef = ncol_factor
    )
  )
  param_prior <- append(
    param_prior,
    list(
      nrow_factor = nrow_factor,
      ncol_factor = ncol_factor,
      size_factor = nrow_factor * ncol_factor,
      lag = factor_lag,
      shape = object$spec$factor$arsig$shape,
      scale = object$spec$factor$arsig$scale
    )
  )
  param_prior$row_prior_prec <- rep(1, nrow_factor)
  param_prior$col_prior_prec <- rep(1, ncol_factor)
  row_prior <- validate_bmar_prior(object$spec$row)
  col_prior <- validate_bmar_prior(object$spec$col)
  num_horizon <- length(y_test_list) - n_ahead + 1
  pred_res <- expand_bdfm_mniw(
    y = do.call(rbind, y_list),
    num_data = length(y_list),
    num_chains = object$chain,
    num_iter = object$iter,
    num_burn = object$burn,
    thin = object$thin,
    fit_record = fit_record,
    run_mcmc = mcmc,
    param_coef_sig = param_prior, coef_sig_init = object$init$param,
    row_prior = row_prior, row_init = object$init$row, row_prior_type = get_prior_id(object$spec$row$prior),
    col_prior = col_prior, col_init = object$init$col, col_prior_type = get_prior_id(object$spec$col$prior),
    factor_rows = nrow_factor, factor_cols = ncol_factor, factor_lag = factor_lag,
    step = n_ahead, y_test = do.call(rbind, y_test_list), get_lpl = lpl, use_fit = use_fit,
    seed_chain = sample.int(.Machine$integer.max, size = object$chain * num_horizon) |> matrix(ncol = object$chain),
    seed_forecast = sample.int(.Machine$integer.max, size = object$chain),
    display_progress = verbose,
    nthreads = num_thread
  )
  num_draw <- nrow(object$param)
  if (lpl) {
    if (med) {
      lpl_val <- apply(pred_res$lpl, 1, median)
    } else {
      lpl_val <- rowMeans(pred_res$lpl)
    }
    pred_res$lpl <- NULL
  }
  y_distn <-
    lapply(
      pred_res$forecast,
      process_mar_ourforecast_draws,
      n_ahead = n_ahead,
      ncol_data = ncol_data,
      num_draw = num_draw
    )
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
    eval_id = n_ahead:length(y_test_list),
    y = object$y
  )
  if (lpl) {
    res$lpl <- lpl_val
  }
  class(res) <- c("predmarbayes_expand", "predmarcv")
  res
}