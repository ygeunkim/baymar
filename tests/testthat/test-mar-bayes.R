help_bmar_fit <- function(row_spec, col_spec) {
  toy_data <- chanqi2025[1:3, 1:5, 1:10]
  mar_bayes(
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
}

test_that("Minnesota Prior", {
  fit_test <- help_bmar_fit(set_mar_minnesota(), set_mar_minnesota())
})

test_that("Horseshoe Prior", {
  fit_test <- help_bmar_fit(set_mar_horseshoe(), set_mar_horseshoe())
})
