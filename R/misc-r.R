#' Validate prior specification
#' @importFrom stats ar.ols
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
          ar.ols(y[i, j, ], aic = FALSE, order.max = 4)$var.pred
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
          ar.ols(y[j, i, ], aic = FALSE, order.max = 4)$var.pred
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
validate_bmarx_colspec <- function(param_prior, x, s, bayes_spec, nrow_exogen, ncol_exogen, nrow_exogen_col_coef) {
  exogen_prior <- validate_bmar_col_spec(x, s + 1, bayes_spec, nrow_exogen, ncol_exogen, nrow_exogen_col_coef)
  exogen_prior$col_prior_mean <- matrix(0L, nrow = nrow_exogen_col_coef, ncol = ncol(param_prior$col_prior_mean))
  param_prior$col_prior_mean <- rbind(param_prior$col_prior_mean, exogen_prior$col_prior_mean)
  param_prior$col_prior_prec <- c(param_prior$col_prior_prec, exogen_prior$col_prior_prec)
  param_prior
}

#' @noRd 
validate_bmar_prior <- function(bayes_spec) {
  prior_nm <- bayes_spec$prior
  switch(
    prior_nm,
    "Minnesota" = bayes_spec,
    "Horseshoe" = list(),
    "MN_Hierarchical" = {
      bayes_spec$shape <- bayes_spec$kapp$shape
      bayes_spec$rate <- bayes_spec$kapp$rate
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
      append(init, list())
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

#' @noRd
get_mat_hs_init <- function(num_chains, nrow_coef) {
  lapply(
    seq_len(num_chains),
    function(init) {
      append(
        init,
        list(
          local_sparsity = exp(runif(nrow_coef, -1, 1)),
          global_sparsity = runif(1, 0, 1)
        )
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
split_matrix_chain <- function(x, chain = 1, varname = "A", num_row, num_col, is_symm = FALSE) {
  # index <- expand.grid(seq_len(lag * num_col), seq_len(num_col))
  index <- expand.grid(seq_len(num_row), seq_len(num_col))
  if (is_symm) {
    index <- index[apply(index, 1, function(x) x[1] >= x[2]), ]
  }
  index <- apply(index, 1, function(x) sprintf("[%s]", paste(x, collapse = ",")))
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
process_mar_ourforecast_draws <- function(x, n_ahead, ncol_data, num_draw) {
  do.call(cbind, x) |>
    t() |>
    split.data.frame(gl(num_draw, ncol_data)) |>
    lapply(t) |>
    simplify2array()
}
