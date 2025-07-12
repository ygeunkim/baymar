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
    p = lag,
    row_coef = row_coef,
    col_coef = col_coef
  )
  expect_equal(
    dim(test_y),
    c(num_row, num_col, num_sim)
  )
})

test_that("MDFM-MAR process generation", {
  num_row <- 3
  num_col <- 5
  nrow_factor <- 2
  ncol_factor <- 3
  lag <- 2
  set.seed(1)
  row_coef <- matrix(runif(num_row * nrow_factor, -1, 1), ncol = num_row)
  col_coef <- matrix(runif(num_col * ncol_factor, -1, 1), ncol = num_col)
  factor_row_coef <- rbind(
    diag(runif(nrow_factor, -1, 1)),
    diag(runif(nrow_factor, -1, 1))
  )
  factor_col_coef <- rbind(
    diag(runif(ncol_factor, -1, 1)),
    diag(runif(ncol_factor, -1, 1))
  )
  num_sim <- 5
  num_burn <- 2
  expect_no_error({
    set.seed(1)
    test_y <- sim_mdfm(
      num_sim = num_sim,
      num_burn = num_burn,
      s = lag,
      row_coef = row_coef,
      col_coef = col_coef,
      factor_row_coef = factor_row_coef,
      factor_col_coef = factor_col_coef
    )
  })
})
