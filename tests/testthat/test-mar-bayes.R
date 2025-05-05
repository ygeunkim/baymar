test_that("multiplication works", {
  toy_data <- chanqi2025[1:3, 1:5, 1:10]
  fit_test <- mar_bayes(
    toy_data,
    p = 2,
    num_chains = 1,
    num_iter = 5,
    num_burn = 2,
    thinning = 1,
    num_thread = 1
  )
})
