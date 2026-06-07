# Fitting Bayesian MAR

This function fits Bayesian Matrix Autoregressive (BMAR) model with
various priors.

## Usage

``` r
mar_bayes(
  y,
  p = 1,
  exogen = NULL,
  s = 0,
  factor_spec = set_matfactor(),
  num_chains = 1,
  num_iter = 1000,
  num_burn = floor(num_iter/2),
  thinning = 1,
  row_spec = set_mar_minnesota(),
  col_spec = row_spec,
  exogen_row_spec = row_spec,
  exogen_col_spec = row_spec,
  factor_row_spec = row_spec,
  factor_col_spec = col_spec,
  verbose = FALSE,
  num_thread = 1
)
```

## Arguments

- y:

  Matrix-valued time series data

- p:

  VAR lag (Default: 1)

- exogen:

  Unmodeled matrices

- s:

  Lag of exogeneous matrices in MARX(p, s). By default, `s = 0`.

- factor_spec:

  Augmented factor matrix specification.

- num_chains:

  Number of MCMC chains

- num_iter:

  MCMC iteration number

- num_burn:

  Number of burn-in (warm-up). Half of the iteration is the default
  choice.

- thinning:

  Thinning every thinning-th iteration

- row_spec:

  Row coefficient specification

- col_spec:

  Column coefficient specification

- exogen_row_spec:

  Exogenous row coefficient prior specification.

- exogen_col_spec:

  Exogenous column coefficient prior specification.

- factor_row_spec:

  Factor row coefficient prior specification.

- factor_col_spec:

  Factor column coefficient prior specification.

- verbose:

  Progress log

- num_thread:

  Number of threads

## References

Chan, J. C. C. & Qi, Y. (2025). Large Bayesian matrix autoregressions.
Journal of Econometrics, 105955.

Zhang, W. (2025). Bayesian Dynamic Factor Models for High-Dimensional
Matrix-Valued Time Series. SSRN Electronic Journal.
