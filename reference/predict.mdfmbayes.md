# Forecasting MDFM

Forecasts matrix dynamic factor model.

## Usage

``` r
# S3 method for class 'mdfmbayes'
predict(object, n_ahead, level = 0.05, num_thread = 1, med = FALSE, ...)
```

## Arguments

- object:

  Model object

- n_ahead:

  step to forecast

- level:

  Specify alpha of confidence interval level 100(1 - alpha) percentage.
  By default, .05.

- num_thread:

  Number of threads

- med:

  **\[experimental\]** If `TRUE`, use median of forecast draws instead
  of mean (default).

- ...:

  not used
