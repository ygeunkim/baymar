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

#' SSVS Prior Specification
#' 
#' @param spike_grid Griddy gibbs grid size for scaling factor (between 0 and 1) of spike sd which is Spike sd = c * slab sd
#' @param slab_shape Inverse gamma shape for slab sd
#' @param slab_scl Inverse gamma scale for slab sd
#' @param s1 First shape of coefficients prior beta distribution
#' @param s2 Second shape of coefficients prior beta distribution
#'
#' @order 1
#' @export
set_mar_ssvs <- function(spike_grid = 100L,
                         slab_shape = .01,
                         slab_scl = .01,
                         s1 = 1, s2 = 1) {
  res <- list(
    grid_size = spike_grid,
    slab_shape = slab_shape,
    slab_scl = slab_scl,
    s1 = s1,
    s2 = s2,
    prior = "SSVS"
  )
  class(res) <- c("matssvsspec", "bmarspec")
  res
}

#' Factor prior specification
#' 
#' @param nrow_factor Number of rows of factor matrix.
#' By default, `0` which will does not use factor term.
#' @param ncol_factor Number of columns of factor matrix.
#' By default, `0` which will does not use factor term.
#' @param is_rw If `TRUE`, factor follows random walk process.
#' @param factor_lag Lag of factor autoregressions.
#' @param row_spec Row coefficient specification in factor MAR
#' @param col_spec Column coefficient specification in factor MAR
#' @param factor_arsig Inverse-Gamma prior for factor AR covariance.
#' [bvhar::set_ldlt()] can specify the IG shape and scale.
#'
#' @order 1
#' @export
set_matfactor <- function(nrow_factor = 0, ncol_factor = 0,
                          is_rw = FALSE,
                          factor_lag = 1,
                          row_spec = NULL,
                          col_spec = row_spec,
                          factor_arsig = set_ldlt()) {
  if (factor_lag < 0 || factor_lag %% 1 != 0) {
    stop("'factor_lag' should be non-negative integer.")
  }
  res <- list(
    nrow_factor = nrow_factor,
    ncol_factor = ncol_factor,
    lag = factor_lag,
    factor_type = ifelse(is_rw, "rw", "wn")
  )
  if (is_rw && factor_lag == 0) {
    if (!inherits(factor_arsig, "ldltspec")) {
      stop("Use 'set_ldlt()' for 'factor_arsig'.")
    }
    res$shape <- factor_arsig$shape
    res$scale <- factor_arsig$scale
  }
  if (factor_lag > 0) {
    if (!is.null(row_spec) && !is.null(col_spec)) {
      if (!is.bmarspec(row_spec)) {
        stop("Wrong 'row_spec'")
      }
      if (!is.bmarspec(col_spec)) {
        stop("Wrong 'col_spec'")
      }
      res$row_spec <- validate_bmar_prior(row_spec)
      res$col_spec <- validate_bmar_prior(col_spec)
      res$factor_type <- "mar"
    } else if (!is.null(factor_arsig)) {
      if (!inherits(factor_arsig, "ldltspec")) {
        stop("Use 'set_ldlt()' for 'factor_arsig'.")
      }
      # res$arsig <- factor_arsig
      res$shape <- factor_arsig$shape
      res$scale <- factor_arsig$scale
      res$factor_type <- "var"
    }
  }
  if (is_rw) {
    res$factor_type <- "rw"
    res$lag <- 1
  }
  # res <- list(
  #   nrow_factor = nrow_factor,
  #   ncol_factor = ncol_factor,
  #   lag = factor_lag,
  #   arsig = factor_arsig
  #   # shape = factor_arsig$shape,
  #   # scale = factor_arsig$scale
  # )
  class(res) <- "matfactorspec"
  res
}

#' @rdname set_matfactor
#' @param x Any object
#' @export
is.matfactorspec <- function(x) {
  inherits(x, "matfactorspec")
}
