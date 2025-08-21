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

test_that("MDFM-vec process generation", {
  num_row <- 3
  num_col <- 5
  nrow_factor <- 2
  ncol_factor <- 3
  size_factor <- nrow_factor * ncol_factor
  lag <- 2
  set.seed(1)
  row_coef <- matrix(runif(num_row * nrow_factor, -1, 1), ncol = num_row)
  col_coef <- matrix(runif(num_col * ncol_factor, -1, 1), ncol = num_col)
  row_sig <- diag(num_row)
  col_sig <- diag(num_col)
  factor_coef <- rbind(
    diag(runif(size_factor, -1, 1)),
    diag(runif(size_factor, -1, 1))
  )
  factor_sig <- diag(size_factor)
  num_sim <- 5
  num_burn <- 2
  expect_no_error({
    set.seed(1)
    test_y <- sim_mdfm_vec_process(
      num_sim = num_sim,
      num_burn = num_burn,
      lag = lag,
      row_coef = row_coef,
      col_coef = col_coef,
      row_sig = row_sig,
      col_sig = col_sig,
      factor_init = matrix(0L, nrow = lag, ncol = size_factor),
      factor_coef = factor_coef, factor_sig = factor_sig,
      seed = sample.int(.Machine$integer.max, size = 1)
    ) |>
      lapply(simplify2array)
  })
})

test_that("FAMAR-vec process generation", {
  num_row <- 3
  num_col <- 5
  mar_lag <- 2
  nrow_factor <- 2
  ncol_factor <- 3
  size_factor <- nrow_factor * ncol_factor
  factor_lag <- 2
  set.seed(1)
  row_coef <- matrix(runif(num_row^2 * mar_lag, -1, 1), ncol = num_row)
  col_coef <- matrix(runif(num_col^2 * mar_lag, -1, 1), ncol = num_col)
  # mar_init <- array(0L, dim = c(num_row, num_col, mar_lag))
  # init_list <- lapply(seq_len(mar_lag), function(id) mar_init[, , id])
  # init_mat <- do.call(rbind, init_list)
  row_sig <- diag(num_row)
  col_sig <- diag(num_col)
  factor_row_coef <- matrix(runif(num_row * nrow_factor, -1, 1), ncol = num_row)
  factor_col_coef <- matrix(runif(num_col * ncol_factor, -1, 1), ncol = num_col)
  factor_coef <- rbind(
    diag(runif(size_factor, -1, 1)),
    diag(runif(size_factor, -1, 1))
  )
  factor_sig <- diag(size_factor)
  num_sim <- 5
  num_burn <- 2
  expect_no_error({
    set.seed(1)
    test_y <- sim_famar_vec_process(
      num_sim = num_sim,
      num_burn = num_burn,
      lag = mar_lag,
      init = matrix(0L, nrow = num_row * mar_lag, ncol = num_col),
      row_coef = row_coef,
      col_coef = col_coef,
      row_sig = row_sig,
      col_sig = col_sig,
      factor_lag = factor_lag,
      factor_init = matrix(0L, nrow = factor_lag, ncol = size_factor),
      factor_row_coef = factor_row_coef,
      factor_col_coef = factor_col_coef,
      factor_coef = factor_coef, factor_sig = factor_sig,
      seed = sample.int(.Machine$integer.max, size = 1)
    ) |>
      lapply(simplify2array)
  })
})
