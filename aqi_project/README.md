# Predicting and Forecasting Air Quality (AQI) – 23XWA8 Predictive Analytics

One dataset (daily AQI of one Indian city) covers the whole syllabus.

## Folder contents
| File | Purpose |
|---|---|
| `aqi_analysis.py` | Main program – runs every step and saves results |
| `generate_sample_data.py` | Makes a SAMPLE `data/city_day.csv` (only needed if you have no real data) |
| `data/city_day.csv` | Dataset (sample now; replace with the real Kaggle file) |
| `requirements.txt` | Libraries to install |
| `REPORT_TEMPLATE.md` | Skeleton for your written report / viva answers |
| `outputs/` | Charts (`figures/`), tables (`tables/`) and `summary.txt` |

## How to run (VS Code)
1. Install Python 3.9+ from python.org (tick "Add Python to PATH").
2. Open VS Code -> File -> Open Folder -> choose `aqi_project`.
3. Install the **Python** extension (Extensions icon, search "Python").
4. Open a terminal: Terminal -> New Terminal.
5. Create an environment and install libraries:
   - Windows:  `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
   - then: `pip install -r requirements.txt`
6. Run: `python aqi_analysis.py`
7. Open the `outputs` folder to see the 17 charts, the tables and `summary.txt`.

## Using the REAL dataset
1. Download `city_day.csv` from Kaggle: "Air Quality Data in India (2015-2020)".
2. Replace `data/city_day.csv` with it.
3. In `aqi_analysis.py` set `CITY = "Delhi"` (or any city in the file) and run again.

## Syllabus coverage
| Syllabus unit | Section in code |
|---|---|
| Data wrangling | 1 (load, duplicates, missing values, outliers, dates) |
| EDA: univariate, bivariate, multivariate | 2 |
| Skewness, kurtosis, central tendency, dispersion | 2 |
| Point and interval estimates | 3 |
| Linear regression: R², significance, residuals, confidence and prediction intervals | 4 |
| Multiple regression: multicollinearity (VIF), heteroscedasticity, outliers, dummy variables, model building | 5 |
| Polynomial regression | 5 |
| Logistic regression: Wald test, Hosmer-Lemeshow, classification table, Gini | 6 |
| Decision trees (CART, Gini) | 7 |
| Moving average, exponential smoothing | 8 |
| ARIMA, forecasting, RMSE and MAPE | 8 |

Not included: CHAID (no standard Python library) and multinomial logistic regression;
mention CART as the tree method used.
