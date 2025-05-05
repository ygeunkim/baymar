#' Validate prior specification
#' @noRd
validate_bmar_spec <- function(bayes_spec, nm = c("row", "col")) {
  spec_type <- match.arg(nm)
  if (!is.bmarspec(bayes_spec)) {
    stop(sprintf("Wrong '%s_spec'", spec_type))
  }
}

#' Set initial values for MNIW
#' @importFrom stats runif
#' @noRd
get_bmar_init <- function(bayes_spec, num_chains, nrow_coef, ncol_coef) {
  param_init <- lapply(
    seq_len(num_chains),
    function(x) {
      init_cov <- diag(exp(runif(ncol_coef, -1, 0)))
      list(
        init_coef = matrix(runif(nrow_coef^2 * ncol_coef, -1, 1), ncol = ncol_coef),
        init_cov = init_cov
      )
    }
  )
  # prior_nm <- bayes_spec$prior
  # if (prior_nm == "Minnesota") {
  #   param_init <- lapply(
  #     param_init,
  #     function(init) {
  #       append(
  #         init,
  #         list(
  #           kappa = runif(1, 0, 1)
  #         )
  #       )
  #     }
  #   )
  # }
  param_init
}
