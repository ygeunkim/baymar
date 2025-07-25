help_bmar_pred <- function(row_spec, col_spec, exogen_row_spec = NULL, exogen_col_spec = NULL, factor_row_spec = NULL, factor_col_spec = NULL) {
  toy_data <- chanqi2025[1:2, 1:3, 1:10]
  exogen <- NULL
  newxreg <- NULL
  if (!is.null(exogen_row_spec)) {
    exogen <- chanqi2025[3:4, 4:5, 1:10]
    newxreg <- chanqi2025[3:4, 4:5, 11:13]
  }
  famar_spec <- set_famar(nrow_factor = 0, ncol_factor = 0, factor_lag = 1)
  if (!is.null(factor_row_spec)) {
    famar_spec <- set_famar(nrow_factor = 2, ncol_factor = 2, factor_lag = 2)
  }
  set.seed(1)
  fit_test <- mar_bayes(
    toy_data,
    p = 2,
    exogen = exogen,
    s = 0,
    famar_spec = famar_spec,
    num_chains = 2,
    num_iter = 5,
    num_burn = 2,
    thinning = 1,
    row_spec = row_spec,
    col_spec = col_spec,
    exogen_row_spec = exogen_row_spec,
    exogen_col_spec = exogen_col_spec,
    num_thread = 1
  )
  set.seed(1)
  predict(fit_test, n_ahead = 3, newxreg = newxreg)
}

test_that("Minnesota Prior", {
  expect_no_error(
    pred_test <- help_bmar_pred(set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1))
  )
  expect_no_error(
    pred_x_test <- help_bmar_pred(set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1))
  )
  expect_no_error(
    pred_x_test <- help_bmar_pred(set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1))
  )
})

test_that("Horseshoe Prior", {
  expect_no_error(
    pred_test <- help_bmar_pred(set_mar_horseshoe(), set_mar_horseshoe())
  )
  expect_no_error(
    pred_x_test <- help_bmar_pred(set_mar_horseshoe(), set_mar_horseshoe(), set_mar_horseshoe(), set_mar_horseshoe())
  )
  expect_no_error(
    pred_x_test <- help_bmar_pred(set_mar_horseshoe(), set_mar_horseshoe(), set_mar_horseshoe(), set_mar_horseshoe(), set_mar_horseshoe(), set_mar_horseshoe())
  )
})

test_that("Hierarchical Minnesota Prior", {
  expect_no_error(
    pred_test <- help_bmar_pred(set_mar_minnesota(), set_mar_minnesota())
  )
  expect_no_error(
    pred_x_test <- help_bmar_pred(set_mar_minnesota(), set_mar_minnesota(), set_mar_minnesota(), set_mar_minnesota())
  )
  expect_no_error(
    pred_x_test <- help_bmar_pred(set_mar_minnesota(), set_mar_minnesota(), set_mar_minnesota(), set_mar_minnesota(), set_mar_minnesota(), set_mar_minnesota())
  )
})

help_bmar_roll <- function(row_spec, col_spec, exogen_row_spec = NULL, exogen_col_spec = NULL, factor_row_spec = NULL, factor_col_spec = NULL) {
  toy_data <- chanqi2025[1:2, 1:3, 1:10]
  eval_data <- chanqi2025[1:2, 1:3, 11:12]
  exogen <- NULL
  newxreg <- NULL
  if (!(is.null(exogen_row_spec) && is.null(exogen_col_spec))) {
    exogen <- chanqi2025[3:4, 4:5, 1:10]
    newxreg <- chanqi2025[3:4, 4:5, 11:12]
  }
  famar_spec <- set_famar(nrow_factor = 0, ncol_factor = 0, factor_lag = 1)
  if (!is.null(factor_row_spec)) {
    famar_spec <- set_famar(nrow_factor = 2, ncol_factor = 2, factor_lag = 2)
  }
  set.seed(1)
  fit_test <- mar_bayes(
    toy_data,
    p = 1,
    exogen = exogen,
    s = 0,
    famar_spec = famar_spec,
    num_chains = 2,
    num_iter = 5,
    num_burn = 2,
    thinning = 1,
    row_spec = row_spec,
    col_spec = col_spec,
    exogen_row_spec = exogen_row_spec,
    exogen_col_spec = exogen_col_spec,
    num_thread = 1
  )
  set.seed(1)
  forecast_roll(fit_test, 1, y_test = eval_data, newxreg = newxreg)
}

test_that("Minnesota Prior - Rolling", {
  expect_no_error(
    pred_test <- help_bmar_roll(set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1))
  )
  expect_no_error(
    pred_test <- help_bmar_roll(set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1))
  )
})

test_that("Horseshoe Prior - Rolling", {
  expect_no_error(
    pred_test <- help_bmar_roll(set_mar_horseshoe(), set_mar_horseshoe())
  )
  expect_no_error(
    pred_test <- help_bmar_roll(set_mar_horseshoe(), set_mar_horseshoe(), set_mar_horseshoe(), set_mar_horseshoe())
  )
})

test_that("Hierarchical Minnesota Prior - Rolling", {
  expect_no_error(
    pred_test <- help_bmar_roll(set_mar_minnesota(), set_mar_minnesota())
  )
  expect_no_error(
    pred_test <- help_bmar_roll(set_mar_minnesota(), set_mar_minnesota(), set_mar_minnesota(), set_mar_minnesota())
  )
})

help_bmar_expand <- function(row_spec, col_spec, exogen_row_spec = NULL, exogen_col_spec = NULL, factor_row_spec = NULL, factor_col_spec = NULL) {
  toy_data <- chanqi2025[1:2, 1:3, 1:10]
  eval_data <- chanqi2025[1:2, 1:3, 11:12]
  exogen <- NULL
  newxreg <- NULL
  if (!(is.null(exogen_row_spec) && is.null(exogen_col_spec))) {
    exogen <- chanqi2025[3:4, 4:5, 1:10]
    newxreg <- chanqi2025[3:4, 4:5, 11:12]
  }
  famar_spec <- set_famar(nrow_factor = 0, ncol_factor = 0, factor_lag = 1)
  if (!is.null(factor_row_spec)) {
    famar_spec <- set_famar(nrow_factor = 2, ncol_factor = 2, factor_lag = 2)
  }
  set.seed(1)
  fit_test <- mar_bayes(
    toy_data,
    p = 1,
    exogen = exogen,
    s = 0,
    famar_spec = famar_spec,
    num_chains = 2,
    num_iter = 5,
    num_burn = 2,
    thinning = 1,
    row_spec = row_spec,
    col_spec = col_spec,
    exogen_row_spec = exogen_row_spec,
    exogen_col_spec = exogen_col_spec,
    num_thread = 1
  )
  set.seed(1)
  forecast_expand(fit_test, 1, y_test = eval_data, newxreg = newxreg)
}

test_that("Minnesota Prior - Expanding", {
  expect_no_error(
    pred_test <- help_bmar_expand(set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1))
  )
  expect_no_error(
    pred_test <- help_bmar_expand(set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1))
  )
})

test_that("Horseshoe Prior - Expanding", {
  skip_on_ci()
  expect_no_error(
    pred_test <- help_bmar_expand(set_mar_horseshoe(), set_mar_horseshoe())
  )
  expect_no_error(
    pred_test <- help_bmar_expand(set_mar_horseshoe(), set_mar_horseshoe(), set_mar_horseshoe(), set_mar_horseshoe())
  )
})

test_that("Hierarchical Minnesota Prior - Expanding", {
  expect_no_error(
    pred_test <- help_bmar_expand(set_mar_minnesota(), set_mar_minnesota())
  )
  expect_no_error(
    pred_test <- help_bmar_expand(set_mar_minnesota(), set_mar_minnesota(), set_mar_minnesota(), set_mar_minnesota())
  )
})