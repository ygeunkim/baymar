## code to prepare `chanqi2025` dataset goes here
# library(httr)
library(dplyr)
library(tidyr)
library(stringr)
library(readxl)

temp_file <- tempfile()
temp_xlsx <- tempfile()
download.file("https://joshuachan.org/code/BMAR_code.zip", temp_file)
unzip(zipfile = temp_file, exdir = temp_xlsx)
file_path <- file.path(temp_xlsx, "Data_state_level_6q.xlsx")
num_series <- 6 # 6 variables
ts_mat <- list()
for (id in 1:num_series) {
  sheet <- read_xlsx(
    file_path,
    sheet = 1,
    skip = 53 * (id - 1),
    n_max = 52
  )
  series_name <- names(sheet)[1]
  ts_mat[[id]] <-
    sheet |>
    rename("state" = series_name, "code" = `Region Code`) |>
    pivot_longer(
      -c(state, code),
      names_to = "date",
      values_to = tolower(series_name) |>
        str_replace_all(" ", replacement = "_") |>
        str_replace_all("-", replacement = "_") |>
        str_remove_all("_\\(.*?\\)") |>
        str_remove_all("_by_state.*")
    ) |>
    mutate(
      date = as.Date(as.numeric(date), origin = "1899-12-30")
    )
}
unlink(c(temp_file, temp_xlsx))
ts_wide <-
  purrr::reduce(ts_mat, left_join, by = c("state", "code", "date")) |>
  # filter(date < "2019-01-01") |> 
  arrange(date)
state_code <-
  ts_wide |>
  select(state, code) |>
  unique()
ts_wide <- select(ts_wide, -code)
ts_long <-
  ts_wide |>
  pivot_longer(-c(state, date), names_to = "series", values_to = "values")
# Transformation
# Follow the matlab code choice instead of the paper
# In matlab code of Chan: only log transformation vs no transformation were used
# year-over-year growth rate produces -Inf in this dataset
ts_transform <-
  ts_long |>
  mutate(
    # values = case_when(
    #   series %in% c("initial_claims", "continued_claims", "new_private_housing_units_authorized_by_building_permits") ~ log(values),
    #   series %in% c("total_nonfarm", "all_transactions_house_price_index") ~ values / lag(values, 4),
    #   series == "unemployment_rate" ~ values
    # )
    values = ifelse(
      series %in% c("initial_claims", "continued_claims", "new_private_housing_units_authorized_by_building_permits"),
      log(values),
      values
    ),
    .by = c(state, series)
  ) |>
  filter(
    state != "Washington",
    date >= "1991-01-01"
  )
# 3d array: series x state x date
chanqi2025 <-
  xtabs(values ~ series + state + date, data = ts_transform) |>
  unclass()
attr(chanqi2025, "call") <- NULL
names(dimnames(chanqi2025)) <- NULL
usethis::use_data(chanqi2025, overwrite = TRUE)
