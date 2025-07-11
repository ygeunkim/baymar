test_that("MAR process generation", {
  num_row <- 3
  num_col <- 5
  lag <- 2
  set.seed(1)
  row_coef <- rbind(
    diag(runif(num_row, -1, 1)),
    diag(runif(num_row, -1, 1))
  )
  col_coef <- rbind(
    diag(runif(num_col, -1, 1)),
    diag(runif(num_col, -1, 1))
  )
  num_sim <- 5
  num_burn <- 2
  set.seed(1)
  test_y <- sim_mar(
    num_sim = num_sim,
    num_burn = num_burn,
    p = 1,
    row_coef = row_coef,
    col_coef = col_coef
  )
  expect_equal(
    dim(test_y),
    c(num_row, num_col, num_sim)
  )
})
