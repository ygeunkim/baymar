#' Validate prior specification
#' @importFrom stats ar.ols
#' @noRd
validate_bmar_row_spec <- function(y, p, bayes_spec, nrow_data, ncol_data, nrow_row_coef) {
  if (!is.bmarspec(bayes_spec)) {
    stop("Wrong 'row_spec'")
  }
  prior_nm <- bayes_spec$prior
  A0 <- matrix(0L, nrow = nrow_row_coef, ncol = nrow_data)
  if (prior_nm == "Minnesota") {
    S_r <- diag(nrow_data)
    diag(S_r) <- sapply(
      1:nrow_data,
      function(i) {
        sapply(
          1:ncol_data,
          function(j) {
            ar.ols(y[i, j, ], aic = FALSE, order = 4)$var.pred
          }
        ) |>
          mean()
      }
    )
    kappa_A <- 1
    V_A <- kronecker(diag(1 / c(1:p)^2), diag(kappa_A / diag(S_r)))
    nu_r <- nrow_data + 2
  }
  list(
    row_prior_mean = A0,
    row_prior_prec = diag(1 / diag(V_A)),
    row_iw_scl = S_r,
    row_iw_df = nu_r
  )
}

#' @noRd
validate_bmar_col_spec <- function(y, p, bayes_spec, nrow_data, ncol_data, nrow_col_coef) {
  if (!is.bmarspec(bayes_spec)) {
    stop("Wrong 'col_spec'")
  }
  prior_nm <- bayes_spec$prior
  # B0 <- kronecker(rep(1, lag), diag(ncol_data)) # kp x k
  B0 <- matrix(0L, nrow = nrow_col_coef, ncol = ncol_data)
  if (prior_nm == "Minnesota") {
    S_c <- diag(ncol_data)
    diag(S_c) <- sapply(
      1:ncol_data,
      function(i) {
        sapply(
          1:nrow_data,
          function(j) {
            ar.ols(y[j, i, ], aic = FALSE, order = 4)$var.pred
          }
        ) |>
          mean()
      }
    )
    kappa_B <- 1
    V_B <- kronecker(diag(1 / c(1:p)^2), diag(kappa_B / diag(S_c)))
    nu_c <- ncol_data + 2
  }
  list(
    col_prior_mean = B0,
    col_prior_prec = diag(1 / diag(V_B)),
    col_iw_scl = S_c,
    col_iw_df = nu_c
  )
}

#' @noRd 
validate_bmar_prior <- function(bayes_spec) {
  prior_nm <- bayes_spec$prior
  switch(
    prior_nm,
    "Minnesota" = bayes_spec,
    stop("Wrong prior")
  )
}

#' @noRd
get_prior_id <- function(prior_nm) {
  switch(prior_nm,
    "Minnesota" = 1,
    1 # MN_VAR, MN_VHAR
  )
}

#' Set initial values for MNIW
#' @importFrom stats runif
#' @noRd
get_bmar_init <- function(num_chains, nrow_data, ncol_data, nrow_row_coef, nrow_col_coef) {
  lapply(
    seq_len(num_chains),
    function(x) {
      list(
        row_init_coef = matrix(runif(nrow_row_coef * nrow_data, -1, 1), ncol = nrow_data),
        row_init_lower = diag(exp(runif(nrow_data, -1, 0))),
        col_init_coef = matrix(runif(nrow_col_coef * ncol_data, -1, 1), ncol = ncol_data),
        col_init_lower = diag(exp(runif(ncol_data, -1, 0)))
      )
    }
  )
}

#' @noRd 
get_mat_minn_init <- function(num_chains) {
  lapply(
    seq_len(num_chains),
    function(init) {
      append(
        init,
        list(
          kappa = runif(1, 0, 1)
        )
      )
    }
  )
}
