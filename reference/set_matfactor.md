# Factor prior specification

Factor prior specification

## Usage

``` r
set_matfactor(
  nrow_factor = 0,
  ncol_factor = 0,
  is_rw = FALSE,
  factor_lag = 1,
  row_spec = NULL,
  col_spec = row_spec,
  factor_arsig = set_ldlt()
)

is.matfactorspec(x)
```

## Arguments

- nrow_factor:

  Number of rows of factor matrix. By default, `0` which will does not
  use factor term.

- ncol_factor:

  Number of columns of factor matrix. By default, `0` which will does
  not use factor term.

- is_rw:

  If `TRUE`, factor follows random walk process.

- factor_lag:

  Lag of factor autoregressions.

- row_spec:

  Row coefficient specification in factor MAR

- col_spec:

  Column coefficient specification in factor MAR

- factor_arsig:

  Inverse-Gamma prior for factor AR covariance.
  [`bvhar::set_ldlt()`](https://bvhar.baeconverse.org/reference/set_ldlt.html)
  can specify the IG shape and scale.

- x:

  Any object
