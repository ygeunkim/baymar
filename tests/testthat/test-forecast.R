help_bmar_pred <- function(row_spec, col_spec) {
  toy_data <- chanqi2025[1:3, 1:5, 1:10]
  set.seed(2)
  fit_test <- mar_bayes(
    toy_data,
    p = 2,
    num_chains = 2,
    num_iter = 5,
    num_burn = 2,
    thinning = 1,
    row_spec = row_spec,
    col_spec = col_spec,
    num_thread = 1
  )
  set.seed(1)
  predict(fit_test, n_ahead = 3)
}

test_that("Minnesota Prior", {
  pred_test <- help_bmar_pred(set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1))
})

test_that("Horseshoe Prior", {
  pred_test <- help_bmar_pred(set_mar_horseshoe(), set_mar_horseshoe())
})

test_that("Hierarchical Minnesota Prior", {
  pred_test <- help_bmar_pred(set_mar_minnesota(), set_mar_minnesota())
})

help_bmar_roll <- function(row_spec, col_spec) {
  toy_data <- chanqi2025[1:3, 1:5, 1:10]
  eval_data <- chanqi2025[1:3, 1:5, 11:12]
  set.seed(1)
  fit_test <- mar_bayes(
    toy_data,
    p = 1,
    num_chains = 2,
    num_iter = 5,
    num_burn = 2,
    thinning = 1,
    row_spec = row_spec,
    col_spec = col_spec,
    num_thread = 1
  )
  set.seed(1)
  forecast_roll(fit_test, 1, eval_data)
}

test_that("Minnesota Prior - Rolling", {
  pred_test <- help_bmar_roll(set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1))
})

test_that("Horseshoe Prior - Rolling", {
  pred_test <- help_bmar_roll(set_mar_horseshoe(), set_mar_horseshoe())
})

test_that("Hierarchical Minnesota Prior - Rolling", {
  pred_test <- help_bmar_roll(set_mar_minnesota(), set_mar_minnesota())
})

help_bmar_expand <- function(row_spec, col_spec) {
  toy_data <- chanqi2025[1:3, 1:5, 1:10]
  eval_data <- chanqi2025[1:3, 1:5, 11:12]
  set.seed(1)
  fit_test <- mar_bayes(
    toy_data,
    p = 1,
    num_chains = 2,
    num_iter = 5,
    num_burn = 2,
    thinning = 1,
    row_spec = row_spec,
    col_spec = col_spec,
    num_thread = 1
  )
  set.seed(1)
  forecast_expand(fit_test, 1, eval_data)
}

test_that("Minnesota Prior - Expanding", {
  pred_test <- help_bmar_expand(set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1))
})

test_that("Horseshoe Prior - Expanding", {
  pred_test <- help_bmar_expand(set_mar_horseshoe(), set_mar_horseshoe())
})

test_that("Hierarchical Minnesota Prior - Expanding", {
  pred_test <- help_bmar_expand(set_mar_minnesota(), set_mar_minnesota())
})