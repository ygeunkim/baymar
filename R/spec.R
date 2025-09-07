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
#' @param factor_arsig Inverse-Gamma prior for factor AR covariance.
#' [bvhar::set_ldlt()] can specify the IG shape and scale.
#'
#' @order 1
#' @export
set_factor <- function(nrow_factor = 0, ncol_factor = 0, factor_lag = 1, factor_arsig = set_ldlt()) {
  if (factor_lag <= 0 || factor_lag %% 1 != 0) {
    stop("'factor_lag' positive integer.")
  }
  if (!inherits(factor_arsig, "ldltspec")) {
    stop("Use 'set_ldlt()' for 'factor_arsig'.")
  }
  res <- list(
    nrow_factor = nrow_factor,
    ncol_factor = ncol_factor,
    lag = factor_lag,
    arsig = factor_arsig
    # shape = factor_arsig$shape,
    # scale = factor_arsig$scale
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
