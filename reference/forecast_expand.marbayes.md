# Pseudo out-of-sample Forecasting based on Expanding Window

Pseudo out-of-sample Forecasting based on Expanding Window

## Usage

``` r
# S3 method for class 'marbayes'
forecast_expand(
  object,
  n_ahead,
  y_test,
  level = 0.05,
  newxreg,
  num_thread = 1,
  med = FALSE,
  lpl = FALSE,
  mcmc = TRUE,
  use_fit = TRUE,
  verbose = FALSE,
  ...
)
```

## Arguments

- object:

  Model object

- n_ahead:

  Step to forecast in rolling window scheme

- y_test:

  Test data to be compared.

- level:

  Specify alpha of confidence interval level 100(1 - alpha) percentage.
  By default, .05.

- newxreg:

  New values for exogenous variables.

- num_thread:

  **\[experimental\]** Number of threads

- med:

  **\[experimental\]** If `TRUE`, use median of forecast draws instead
  of mean (default).

- lpl:

  **\[experimental\]** Compute log-predictive likelihood (LPL). By
  default, `FALSE`.

- mcmc:

  **\[experimental\]** If `TRUE`, run new MCMC in new windows. By
  default, `TRUE`.

- use_fit:

  **\[experimental\]** Use `object` result for the first window. By
  default, `TRUE`.

- verbose:

  Print the progress bar in the console. By default, `FALSE`.

- ...:

  Additional arguments
