# Generate Matrix Time Series following MAR(p)

This function generates MAR(p) time series array.

## Usage

``` r
sim_mar(
  num_sim,
  num_burn = floor(num_sim/2),
  p = 1,
  row_coef,
  col_coef,
  row_sig = diag(ncol(row_coef)),
  col_sig = diag(ncol(col_coef)),
  init = array(0L, dim = c(ncol(row_coef), ncol(col_coef), p))
)
```

## Arguments

- num_sim:

  Number to generated process

- num_burn:

  Number of burn-in

- p:

  MAR lag

- row_coef:

  Row MAR coefficient (np x n).

- col_coef:

  Column MAR coefficient (kp x k).

- row_sig:

  Row covariance of innovation matrix. By default, identity matrix.

- col_sig:

  Column covariance of innovation matrix. By default, identity matrix.

- init:

  Initial \\Y_1, \ldots, Y_p\\ array or list to simulate MAR model. By
  default, zero.

## Value

n x k x (num_sim - num_burn) array.
