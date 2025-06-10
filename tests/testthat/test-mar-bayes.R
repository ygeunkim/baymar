help_bmar_fit <- function(row_spec, col_spec, exogen_row_spec = NULL, exogen_col_spec = NULL) {
  toy_data <- chanqi2025[1:3, 1:5, 1:10]
  exogen <- NULL
  if (!is.null(exogen_row_spec)) {
    exogen <- chanqi2025[4:5, 6:10, 1:10]
  }
  set.seed(1)
  mar_bayes(
    toy_data,
    p = 2,
    exogen = exogen,
    s = 0,
    num_chains = 1,
    num_iter = 5,
    num_burn = 2,
    thinning = 1,
    row_spec = row_spec,
    col_spec = col_spec,
    exogen_row_spec = exogen_row_spec,
    exogen_col_spec = exogen_col_spec,
    num_thread = 1
  )
}

test_that("Minnesota Prior", {
  expect_no_error(
    fit_test <- help_bmar_fit(set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1))
  )
  expect_no_error(
    fit_x_test <- help_bmar_fit(set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1), set_mar_minnesota(kappa = .1))
  )
})

test_that("Horseshoe Prior", {
  expect_no_error(
    fit_test <- help_bmar_fit(set_mar_horseshoe(), set_mar_horseshoe())
  )
  expect_no_error(
    fit_x_test <- help_bmar_fit(set_mar_horseshoe(), set_mar_horseshoe(), set_mar_horseshoe(), set_mar_horseshoe())
  )
})

test_that("Hierarchical Minnesota Prior", {
  expect_no_error(
    fit_test <- help_bmar_fit(set_mar_minnesota(), set_mar_minnesota())
  )
  expect_no_error(
    fit_x_test <- help_bmar_fit(set_mar_minnesota(), set_mar_minnesota(), set_mar_minnesota(), set_mar_minnesota())
  )
})
