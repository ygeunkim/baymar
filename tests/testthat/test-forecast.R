help_bmar_pred <- function(row_spec, col_spec) {
  toy_data <- chanqi2025[1:3, 1:5, 1:10]
  fit_test <- mar_bayes(
    toy_data,
    p = 2,
    num_chains = 1,
    num_iter = 5,
    num_burn = 2,
    thinning = 1,
    row_spec = row_spec,
    col_spec = col_spec,
    num_thread = 1
  )
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
