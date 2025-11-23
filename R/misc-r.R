#' @noRd
validate_newxmat <- function(newxreg, n_ahead) {
  if (missing(newxreg) || is.null(newxreg)) {
    stop("'newxreg' should be supplied when using MARX model.")
  }
  if (!is.array(newxreg)) {
    stop("Provide array for 'newxreg'.")
  }
  if (length(dim(newxreg)) != 3) {
    stop("Array should be 3-dim: variable x region x time")
  }
  if (dim(newxreg)[3] != n_ahead) {
    stop("The length of 'newxreg' should be the same as 'n_ahead'")
  }
  lapply(seq_len(n_ahead), function(x) newxreg[, , x])
}

#' Validate prior specification
#' @noRd
validate_bmar_row_spec <- function(y, p, bayes_spec, nrow_data, ncol_data, nrow_row_coef) {
  if (!is.bmarspec(bayes_spec)) {
    stop("Wrong 'row_spec'")
  }
  prior_nm <- bayes_spec$prior
  A0 <- matrix(0L, nrow = nrow_row_coef, ncol = nrow_data)
  S_r <- diag(nrow_data)
  diag(S_r) <- sapply(
    1:nrow_data,
    function(i) {
      sapply(
        1:ncol_data,
        function(j) {
          ar_ols_sd(as.matrix(y[i, j, ]), p = 4, include_mean = FALSE, penalty = .1)
        }
      ) |>
        mean()
    }
  )
  nu_r <- nrow_data + 2
  if (prior_nm == "Minnesota" || prior_nm == "MN_Hierarchical") {
    V_A <- kronecker(diag(1 / c(1:p)^2), diag(1 / diag(S_r)))
  } else if (prior_nm == "Horseshoe") {
    V_A <- diag(nrow_row_coef)
  }
  list(
    row_prior_mean = A0,
    row_prior_prec = 1 / diag(V_A),
    row_iw_scl = S_r,
    row_iw_df = nu_r
  )
}

#' @noRd
validate_bmarx_rowspec <- function(param_prior, x, s, bayes_spec, nrow_exogen, ncol_exogen, nrow_exogen_row_coef) {
  exogen_prior <- validate_bmar_row_spec(x, s + 1, bayes_spec, nrow_exogen, ncol_exogen, nrow_exogen_row_coef)
  exogen_prior$row_prior_mean <- matrix(0L, nrow = nrow_exogen_row_coef, ncol = ncol(param_prior$row_prior_mean))
  param_prior$row_prior_mean <- rbind(param_prior$row_prior_mean, exogen_prior$row_prior_mean)
  param_prior$row_prior_prec <- c(param_prior$row_prior_prec, exogen_prior$row_prior_prec)
  param_prior
}

#' @noRd
validate_factor_row_spec <- function(factor_spec) {
  if (!is.matfactorspec(factor_spec)) {
    stop("Wrong 'factor_spec'")
  }
  bayes_spec <- factor_spec$row_spec
  nrow_factor <- factor_spec$nrow_factor
  factor_lag <- factor_spec$lag
  nrow_row_coef <- nrow_factor * factor_lag
  # if (!is.bmarspec(bayes_spec)) {
  #   stop("Wrong 'row_spec'")
  # }
  prior_nm <- bayes_spec$prior
  A0 <- matrix(0L, nrow = nrow_row_coef, ncol = nrow_factor)
  S_r <- diag(nrow_factor)
  nu_r <- nrow_factor + 2
  if (prior_nm == "Minnesota" || prior_nm == "MN_Hierarchical") {
    V_A <- kronecker(diag(1 / c(1:factor_lag)^2), S_r)
  } else if (prior_nm == "Horseshoe") {
    V_A <- diag(nrow_row_coef)
  }
  list(
    factor_row_prior_mean = A0,
    factor_row_prior_prec = 1 / diag(V_A),
    factor_row_iw_scl = S_r,
    factor_row_iw_df = nu_r
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
  S_c <- diag(ncol_data)
  diag(S_c) <- sapply(
    1:ncol_data,
    function(i) {
      sapply(
        1:nrow_data,
        function(j) {
          ar_ols_sd(as.matrix(y[j, i, ]), p = 4, include_mean = FALSE, penalty = .1)
        }
      ) |>
        mean()
    }
  )
  nu_c <- ncol_data + 2
  if (prior_nm == "Minnesota" || prior_nm == "MN_Hierarchical") {
    V_B <- kronecker(diag(1 / c(1:p)^2), diag(1 / diag(S_c)))
  } else if (prior_nm == "Horseshoe") {
    V_B <- diag(nrow_col_coef)
  }
  list(
    col_prior_mean = B0,
    col_prior_prec = 1 / diag(V_B),
    col_iw_scl = S_c,
    col_iw_df = nu_c
  )
}

#' @noRd
validate_factor_col_spec <- function(factor_spec) {
  if (!is.matfactorspec(factor_spec)) {
    stop("Wrong 'factor_spec'")
  }
  bayes_spec <- factor_spec$col_spec
  ncol_data <- factor_spec$ncol_factor
  factor_lag <- factor_spec$lag
  nrow_col_coef <- ncol_data * factor_lag
  # if (!is.bmarspec(bayes_spec)) {
  #   stop("Wrong 'col_spec'")
  # }
  prior_nm <- bayes_spec$prior
  # B0 <- kronecker(rep(1, lag), diag(ncol_data)) # kp x k
  B0 <- matrix(0L, nrow = nrow_col_coef, ncol = ncol_data)
  S_c <- diag(ncol_data)
  nu_c <- ncol_data + 2
  if (prior_nm == "Minnesota" || prior_nm == "MN_Hierarchical") {
    V_B <- kronecker(diag(1 / c(1:factor_lag)^2), S_c)
  } else if (prior_nm == "Horseshoe") {
    V_B <- diag(nrow_col_coef)
  }
  list(
    factor_col_prior_mean = B0,
    factor_col_prior_prec = 1 / diag(V_B),
    factor_col_iw_scl = S_c,
    factor_col_iw_df = nu_c
  )
}

#' @noRd
validate_bmarx_colspec <- function(param_prior, x, s, bayes_spec, nrow_exogen, ncol_exogen, nrow_exogen_col_coef) {
  exogen_prior <- validate_bmar_col_spec(x, s + 1, bayes_spec, nrow_exogen, ncol_exogen, nrow_exogen_col_coef)
  exogen_prior$col_prior_mean <- matrix(0L, nrow = nrow_exogen_col_coef, ncol = ncol(param_prior$col_prior_mean))
  param_prior$col_prior_mean <- rbind(param_prior$col_prior_mean, exogen_prior$col_prior_mean)
  param_prior$col_prior_prec <- c(param_prior$col_prior_prec, exogen_prior$col_prior_prec)
  param_prior
}

#' @noRd
validate_factor_spec <- function(bayes_spec) {
  if (!is.matfactorspec(bayes_spec)) {
    stop("Wrong 'bayes_spec'")
  }
  # if (bayes_spec$nrow_factor == 0 || bayes_spec$ncol_factor == 0) {
  #   stop("Wrong 'factor_spec'")
  # }
  size_factor <- bayes_spec$nrow_factor * bayes_spec$ncol_factor
  # if (length(bayes_spec$arsig$shape) == 1) {
  #   bayes_spec$arsig$shape <- rep(bayes_spec$arsig$shape, size_factor)
  # }
  # if (length(bayes_spec$arsig$scale) == 1) {
  #   bayes_spec$arsig$scale <- rep(bayes_spec$arsig$scale, size_factor)
  # }
  if (length(bayes_spec$shape) == 1) {
    bayes_spec$shape <- rep(bayes_spec$shape, size_factor)
  }
  if (length(bayes_spec$scale) == 1) {
    bayes_spec$scale <- rep(bayes_spec$scale, size_factor)
  }
  bayes_spec
}

#' @noRd 
validate_bmar_prior <- function(bayes_spec) {
  prior_nm <- bayes_spec$prior
  switch(
    prior_nm,
    "Minnesota" = bayes_spec,
    "Horseshoe" = list(prior = prior_nm),
    "MN_Hierarchical" = {
      bayes_spec$shape <- bayes_spec$kappa$shape
      bayes_spec$rate <- bayes_spec$kappa$rate
      bayes_spec
    },
    stop("Wrong prior")
  )
}

#' @noRd
get_prior_id <- function(prior_nm) {
  switch(
    prior_nm,
    "Minnesota" = 1,
    "Horseshoe" = 3,
    "MN_Hierarchical" = 4,
    1
  )
}

#' Set initial values for MNIW
#' @importFrom stats runif
#' @noRd
get_bmar_coef_init <- function(num_chains, nrow_data, ncol_data, nrow_row_coef, nrow_col_coef) {
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
get_empty_init <- function(num_chains) {
  lapply(
    seq_len(num_chains),
    function(init) {
      # append(init, list())
      list()
    }
  )
}

#' @noRd 
get_mat_minn_init <- function(num_chains) {
  lapply(
    seq_len(num_chains),
    function(init) {
      # append(
      #   init,
      #   list(
      #     kappa = runif(1, 0, 1)
      #   )
      # )
      list(kappa = runif(1, 0, 1))
    }
  )
}

#' @noRd
get_mat_hs_init <- function(num_chains, nrow_coef) {
  lapply(
    seq_len(num_chains),
    function(init) {
      # append(
      #   init,
      #   list(
      #     local_sparsity = exp(runif(nrow_coef, -1, 1)),
      #     global_sparsity = runif(1, 0, 1)
      #   )
      # )
      list(
        local_sparsity = exp(runif(nrow_coef, -1, 1)),
        global_sparsity = runif(1, 0, 1)
      )
    }
  )
}

#' @noRd
get_bmar_init <- function(bayes_spec, num_chains, nrow_coef) {
  switch(
    bayes_spec$prior,
    "Minnesota" = get_empty_init(num_chains),
    "Horseshoe" = {
      get_mat_hs_init(
        num_chains = num_chains,
        nrow_coef = nrow_coef
      )
    },
    "MN_Hierarchical" = get_mat_minn_init(num_chains),
    stop(sprintf("Wrong %s prior", deparse(substitute(bayes_spec))))
  )
}

#' @noRd
get_fac_var_coef_init <- function(param_init, size_factor, factor_lag) {
  lapply(
    param_init,
    function(init) {
      append(
        init,
        list(
          factor_arcoef_init = matrix(runif(size_factor * factor_lag, -1, 1), ncol = factor_lag),
          factor_arprec_init = exp(runif(size_factor, -1, 0))
        )
      )
    }
  )
}
# get_bmdfm_coef_init <- function(num_chains, nrow_data, ncol_data, nrow_row_coef, nrow_col_coef, size_factor, factor_lag) {
#   lapply(
#     seq_len(num_chains),
#     function(init) {
#       list(
#         row_init_coef = matrix(runif(nrow_row_coef * nrow_data, -1, 1), ncol = nrow_data),
#         row_init_lower = diag(exp(runif(nrow_data, -1, 0))),
#         col_init_coef = matrix(runif(nrow_col_coef * ncol_data, -1, 1), ncol = ncol_data),
#         col_init_lower = diag(exp(runif(ncol_data, -1, 0))),
#         factor_arcoef_init = matrix(runif(size_factor * factor_lag, -1, 1), ncol = factor_lag),
#         factor_arprec_init = exp(runif(size_factor, -1, 0))
#       )
#     }
#   )
# }

#' @noRd
get_fac_mar_coef_init <- function(param_init, nrow_factor, ncol_factor, factor_lag) {
  lapply(
    param_init,
    function(init) {
      append(
        init,
        list(
          factor_row_init_coef = matrix(runif(factor_lag * nrow_factor^2, -1, 1), ncol = nrow_factor),
          factor_row_init_lower = diag(exp(runif(nrow_factor, -1, 0))),
          factor_col_init_coef = matrix(runif(factor_lag * ncol_factor^2, -1, 1), ncol = ncol_factor),
          factor_col_init_lower = diag(exp(runif(ncol_factor, -1, 0)))
        )
      )
    }
  )
}

#' Validate coefficient and covariance
#'
#' @noRd
validate_coef_sig <- function(coef, sig) {
  coef_name <- deparse(substitute(coef))
  sig_name <- deparse(substitute(sig))
  if (!is.matrix(coef)) {
    stop(sprintf("'%s' should be a matrix.", coef_name))
  }
  if (!is.matrix(sig)) {
    stop(sprintf("'%s' should be a matrix.", sig_name))
  }
  if (nrow(sig) != ncol(sig)) {
    stop(sprintf("'%s' should be square matrix.", sig_name))
  }
  if (ncol(coef) != ncol(sig)) {
    stop(sprintf("'%s' and '%s' should have the same number of columns", coef_name, sig_name))
  }
}

#' Split matrix draw
#' 
#' @noRd 
split_matrix_chain <- function(x, chain = 1, varname = "A", num_row, num_col, is_symm = FALSE, num_design = 0) {
  # index <- expand.grid(seq_len(lag * num_col), seq_len(num_col))
  index <- expand.grid(seq_len(num_row), seq_len(num_col))
  if (num_row == 0) {
    index <- seq_len(num_col)
    if (num_col == 1) {
      index <- NULL
    }
    index <- sapply(index, function(x) sprintf("[%s]", paste(x, collapse = ",")))
  } else {
    if (num_design > 0) {
      index <- expand.grid(seq_len(num_row), seq_len(num_col), seq_len(num_design))
    }
    if (is_symm) {
      index <- index[apply(index, 1, function(x) x[1] >= x[2]), ]
    }
    index <- apply(index, 1, function(x) sprintf("[%s]", paste(x, collapse = ",")))
  }
  # if (lag > 0) {
  #   index <- paste0(rep(1:lag, each = length(index)), index)
  # }
  if (chain == 1) {
    colnames(x) <- paste0(varname, index)
    return(x)
  } else {
    # rbind(chain1, chain2, ...)
    num_row <- nrow(x) / chain
    res <-
      t(x) |>
      array(dim = c(ncol(x), num_row, chain)) |>
      aperm(c(2, 3, 1))
    dimnames(res) <- list(
      iteration = seq_len(num_row),
      chain = seq_len(chain),
      variable = paste0(varname, index)
    )
  }
  res
}

#' @importFrom stats setNames
#' @noRd 
get_bmar_records <- function(object, split_chain = FALSE) {
  num_chains <- 1
  if (split_chain) {
    num_chains <- object$chain
  }
  lapply(
    object$param_names,
    function(x) {
      subset_draws(object$param, variable = x) |>
        as_draws_matrix() |>
        split.data.frame(gl(num_chains, nrow(object$param) / num_chains))
    }
  ) |>
    setNames(paste(object$param_names, "record", sep = "_"))
}

#' @noRd
process_mar_forecast_draws <- function(x, n_ahead, nrow_data, ncol_data, num_draw) {
  do.call(cbind, x) |> # (n * h) x (k * num_draw)
    split.data.frame(gl(n_ahead, nrow_data)) |>
    lapply(
      function(x) {
        split.data.frame(t(x), gl(num_draw, ncol_data)) |>
          lapply(t)
      }
    ) |>
    lapply(simplify2array)
}

#' @noRd
process_mar_outforecast_draws <- function(x, n_ahead, ncol_data, num_draw) {
  do.call(cbind, x) |>
    t() |>
    split.data.frame(gl(num_draw, ncol_data)) |>
    lapply(t) |>
    simplify2array()
}

#' @noRd
process_mar_pathforecast_draws <- function(draws, n_ahead, nrow_data, ncol_data, num_draw,
                                           var_names, level = .05, med = FALSE) {
  y_distn <-
    lapply(
      draws,
      process_mar_forecast_draws,
      n_ahead = n_ahead,
      nrow_data = nrow_data,
      ncol_data = ncol_data,
      num_draw = num_draw
    )
  if (med) {
    pred_mean <-
      lapply(
        y_distn,
        function(y_draws) {
          lapply(y_draws, function(x) apply(x, c(1, 2), median)) |>
            simplify2array()
        }
      ) |>
      simplify2array()
  } else {
    pred_mean <-
      lapply(
        y_distn,
        function(y_draws) {
          lapply(y_draws, function(x) apply(x, c(1, 2), mean)) |>
            simplify2array()
        }
      ) |>
        simplify2array()
  }
  lower_quantile <-
    lapply(
      y_distn,
      function(y_draws) {
        lapply(y_draws, function(x) apply(x, c(1, 2), quantile, probs = level / 2)) |>
          simplify2array()
      }
    ) |>
      simplify2array()
  upper_quantile <-
    lapply(
      y_distn,
      function(y_draws) {
        lapply(y_draws, function(x) apply(x, c(1, 2), quantile, probs = 1 - level / 2)) |>
          simplify2array()
      }
    ) |>
    simplify2array()
  est_se <-
    lapply(
      y_distn,
      function(y_draws) {
        lapply(y_draws, function(x) apply(x, c(1, 2), sd)) |>
          simplify2array()
      }
    ) |>
    simplify2array()
  dimnames(pred_mean) <- var_names
  dimnames(lower_quantile) <- var_names
  dimnames(upper_quantile) <- var_names
  dimnames(est_se) <- var_names
  list(
    mean = pred_mean,
    sd = est_se,
    lower = lower_quantile,
    upper = upper_quantile
  )
}
