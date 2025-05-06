#' Minnesota Prior Specification
#'
#' `r lifecycle::badge("experimental")` Set Minnesota prior.
#'
#' @param shape Shape for Gamma prior
#' @param rate Rate for Gamma prior
#'
#' @order 1
#' @export
set_minnesota <- function(shape = 3, rate = 2) {
  res <- list(
    prior = "Minnesota",
    shape = shape,
    rate = rate
  )
  class(res) <- c("matmnspec", "bmarspec")
  res
}

#' @rdname set_minnesota
#' @param x Any object
#' @export
is.matmnspec <- function(x) {
  inherits(x, "matmnspec")
}

#' @rdname set_minnesota
#' @param x Any object
#' @export
is.bmarspec <- function(x) {
  inherits(x, "bmarspec")
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
