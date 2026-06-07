# Forecasting MAR

Forecasts matrix-valued time series.

## Usage

``` r
# S3 method for class 'marbayes'
predict(
  object,
  n_ahead,
  level = 0.05,
  newxreg,
  num_thread = 1,
  med = FALSE,
  ...
)
```

## Arguments

- object:

  Model object

- n_ahead:

  step to forecast

- level:

  Specify alpha of confidence interval level 100(1 - alpha) percentage.
  By default, .05.

- newxreg:

  New values for exogenous matrices.

- num_thread:

  Number of threads

- med:

  **\[experimental\]** If `TRUE`, use median of forecast draws instead
  of mean (default).

- ...:

  not used
