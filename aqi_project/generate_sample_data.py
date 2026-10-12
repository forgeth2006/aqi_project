"""
Creates a SAMPLE dataset (data/city_day.csv) shaped like the Kaggle
"Air Quality Data in India" file, so the project runs out of the box.

>>> For your final submission, download the REAL city_day.csv from Kaggle
>>> and replace data/city_day.csv with it. aqi_analysis.py works with both.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
dates = pd.date_range("2015-01-01", "2020-06-30", freq="D")
n = len(dates)
doy = dates.dayofyear.values

# winter (Nov-Jan) is worst -> cosine seasonality peaking around day 355
season = 1 + 0.75 * np.cos(2 * np.pi * (doy - 355) / 365)
pm25 = 85 * season * rng.lognormal(0, 0.35, n)
pm10 = pm25 * rng.normal(1.9, 0.18, n) + rng.normal(0, 12, n)
no2 = 30 + 0.12 * pm25 + rng.normal(0, 8, n)
so2 = 12 + 0.03 * pm25 + rng.normal(0, 3, n)
co = 0.8 + 0.012 * pm25 + rng.normal(0, 0.35, n)
o3 = 38 + 14 * (1 - season) + rng.normal(0, 11, n)
no = 0.4 * no2 + rng.normal(0, 4, n)
nh3 = 35 + rng.normal(0, 8, n)

def sub_pm25(x):
    """Simplified CPCB PM2.5 sub-index (piecewise linear)."""
    xp = [0, 30, 60, 90, 120, 250, 500]
    fp = [0, 50, 100, 200, 300, 400, 500]
    return np.interp(x, xp, fp)

aqi = np.maximum.reduce([sub_pm25(pm25), pm10 * 0.65, no2 * 1.1, co * 40]) \
      + rng.normal(0, 30, n)
aqi = np.clip(aqi, 15, 500)

df = pd.DataFrame({
    "City": "Delhi", "Date": dates.strftime("%Y-%m-%d"),
    "PM2.5": pm25, "PM10": pm10, "NO": no, "NO2": no2, "NH3": nh3,
    "CO": co, "SO2": so2, "O3": o3, "AQI": aqi,
}).round(2)

# add realistic messiness so the cleaning step has work to do
for col in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]:
    df.loc[rng.choice(n, int(0.04 * n), replace=False), col] = np.nan
df = pd.concat([df, df.sample(15, random_state=1)]).sample(frac=1, random_state=2)
df.loc[df.sample(6, random_state=3).index, "PM2.5"] *= 4      # a few extreme outliers

df.to_csv("data/city_day.csv", index=False)
print("Sample data written:", df.shape)
