#' Minnesota Prior Specification
#'
#' `r lifecycle::badge("experimental")` Set Minnesota prior.
#'
#' @param kappa Shrinkage hyperparameter of MN scale
#'
#' @order 1
#' @export
set_mar_minnesota <- function(kappa = set_kappa()) {
  if (!(
    is.kappaspec(kappa) ||
      (is.numeric(kappa) && length(kappa) == 1)
  )) {
    stop("'kappa' should be length-one numeric or kappaspec.")
  }
  res <- list(
    prior = ifelse(is.kappaspec(kappa), "MN_Hierarchical", "Minnesota"),
    kappa = kappa
  )
  class(res) <- c("matmnspec", "bmarspec")
  res
}

#' Hyperprior for kappa of Minnesota prior
#' 
#' Set Gamma prior for kappa of Minnesota prior
#' 
#' @param shape Shape for Gamma prior
#' @param rate Rate for Gamma prior
#' 
#' @order 1
#' @export 
set_kappa <- function(shape = 3, rate = 2) {
  res <- list(
    shape = shape,
    rate = rate
  )
  class(res) <- "kappaspec"
  res
}

#' @rdname set_mar_minnesota
#' @param x Any object
#' @export
is.matmnspec <- function(x) {
  inherits(x, "matmnspec")
}

#' @rdname set_mar_minnesota
#' @param x Any object
#' @export
is.bmarspec <- function(x) {
  inherits(x, "bmarspec")
}

#' @rdname set_mar_minnesota
#' @param x Any object
#' @export
is.kappaspec <- function(x) {
  inherits(x, "kappaspec")
}

#' Horseshoe Prior Specification
#' 
#' @order 1
#' @export
set_mar_horseshoe <- function() {
  res <- list(
    prior = "Horseshoe"
  )
  class(res) <- c("mathsspec", "bmarspec")
  res
}

#' @rdname set_mar_horseshoe
#' @param x Any object
#' @export
is.mathsspec <- function(x) {
  inherits(x, "mathsspec")
}

#' Factor prior specification
#' 
#' @param nrow_factor Number of rows of factor matrix.
#' By default, `0` which will does not use factor term.
#' @param ncol_factor Number of columns of factor matrix.
#' By default, `0` which will does not use factor term.
#' @param factor_lag Lag of factor autoregressions.
#'
#' @order 1
#' @export
set_factor <- function(nrow_factor = 0, ncol_factor = 0, factor_lag = 1) {
  if (factor_lag <= 0 || factor_lag %% 1 != 0) {
    stop("'factor_lag' positive integer.")
  }
  res <- list(
    nrow_factor = nrow_factor,
    ncol_factor = ncol_factor,
    lag = factor_lag
  )
  class(res) <- "factorspec"
  res
}

#' @rdname set_factor
#' @param x Any object
#' @export
is.factorspec <- function(x) {
  inherits(x, "factorspec")
}

#' Vectorzied factor prior specification
#'
#' @param nrow_factor Number of rows of factor matrix.
#' By default, `0` which will does not use factor term.
#' @param ncol_factor Number of columns of factor matrix.
#' By default, `0` which will does not use factor term.
#' @param factor_lag Lag of factor autoregressions.
#' @param ig_shape Inverse Gamma shape for precision
#' @param ig_scale Inverse Gamma scale for precision
#'
#' @order 1
#' @export
set_dfm <- function(nrow_factor = 2, ncol_factor = 2, factor_lag = 2, ig_shape = 3, ig_scale = 1) {
  if (factor_lag <= 0 || factor_lag %% 1 != 0) {
    stop("'factor_lag' positive integer.")
  }
  res <- list(
    nrow_factor = nrow_factor,
    ncol_factor = ncol_factor,
    lag = factor_lag,
    shape = ig_shape,
    scale = ig_scale
  )
  class(res) <- "dfmspec"
  res
}

#' @rdname set_dfm
#' @param x Any object
#' @export
is.dfmspec <- function(x) {
  inherits(x, "dfmspec")
}
