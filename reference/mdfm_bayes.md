# Fitting Bayesian Matrix DFM

This function fits Bayesian Matrix Dynamic Factor Model (MDFM) with
various priors.

## Usage

``` r
mdfm_bayes(
  y,
  factor_spec = set_matfactor(),
  num_chains = 1,
  num_iter = 1000,
  num_burn = floor(num_iter/2),
  thinning = 1,
  row_spec = set_mar_minnesota(),
  col_spec = row_spec,
  verbose = FALSE,
  num_thread = 1
)
```

## Arguments

- y:

  Matrix-valued time series data

- factor_spec:

  Factor matrix specification.

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

- verbose:

  Progress log

- num_thread:

  Number of threads
