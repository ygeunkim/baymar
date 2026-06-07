# SSVS Prior Specification

SSVS Prior Specification

## Usage

``` r
set_mar_ssvs(
  spike_grid = 100L,
  slab_shape = 0.01,
  slab_scl = 0.01,
  s1 = 1,
  s2 = 1
)
```

## Arguments

- spike_grid:

  Griddy gibbs grid size for scaling factor (between 0 and 1) of spike
  sd which is Spike sd = c \* slab sd

- slab_shape:

  Inverse gamma shape for slab sd

- slab_scl:

  Inverse gamma scale for slab sd

- s1:

  First shape of coefficients prior beta distribution

- s2:

  Second shape of coefficients prior beta distribution
