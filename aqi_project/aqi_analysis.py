"""
Predicting and Forecasting Air Quality (AQI) Using Statistical and Predictive Models
Course: 23XWA8 Predictive Analytics

Run:  python aqi_analysis.py
All tables go to outputs/tables, all charts to outputs/figures,
and a plain-text summary to outputs/summary.txt
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # save figures, don't open windows
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score,
                             mean_squared_error, classification_report)
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------------ settings
DATA_FILE = "data/city_day.csv"
CITY = "Delhi"                      # change to any city present in the file
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
FIG = "outputs/figures"
TAB = "outputs/tables"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

summary = []                        # lines collected for outputs/summary.txt


def log(text=""):
    print(text)
    summary.append(str(text))


def heading(text):
    log("\n" + "=" * 70)
    log(text)
    log("=" * 70)


def savefig(name):
    plt.tight_layout()
    plt.savefig(f"{FIG}/{name}.png", dpi=130)
    plt.close()


def rmse(y, yhat):
    return float(np.sqrt(mean_squared_error(y, yhat)))


def mape(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


def aqi_bucket(v):
    """Standard Indian CPCB AQI categories."""
    if v <= 50:
        return "Good"
    if v <= 100:
        return "Satisfactory"
    if v <= 200:
        return "Moderate"
    if v <= 300:
        return "Poor"
    if v <= 400:
        return "Very Poor"
    return "Severe"


BUCKET_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

# =========================================================================
# 1. DATA WRANGLING  (Unit: Data ingest, cleaning)
# =========================================================================
heading("1. DATA WRANGLING")
raw = pd.read_csv(DATA_FILE)
log(f"Raw file shape: {raw.shape}")

df = raw[raw["City"] == CITY].copy()
log(f"Rows for {CITY}: {len(df)}")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])

n_dup = int(df.duplicated(subset=["Date"]).sum())
df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
log(f"Duplicate dates removed: {n_dup}")

df = df[["Date"] + POLLUTANTS + ["AQI"]]
log("\nMissing values before cleaning:")
log(df.isna().sum().to_string())

df = df.dropna(subset=["AQI"])                                   # target must exist
df = df.set_index("Date")
df[POLLUTANTS] = df[POLLUTANTS].interpolate(method="time").bfill().ffill()
df = df.reset_index()

# outlier treatment: cap values outside 1.5*IQR fences (winsorising) for pollutants
n_capped = 0
for c in POLLUTANTS:
    q1, q3 = df[c].quantile([0.25, 0.75])
    lo, hi = q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)             # 3*IQR = extreme outliers
    n_capped += int(((df[c] < lo) | (df[c] > hi)).sum())
    df[c] = df[c].clip(lo, hi)
log(f"\nExtreme outlier values capped (3*IQR rule): {n_capped}")

df["AQI_Bucket"] = df["AQI"].apply(aqi_bucket)
df["Month"] = df["Date"].dt.month
df["Season"] = df["Month"].map(lambda m: "Winter" if m in (11, 12, 1, 2)
                               else "Summer" if m in (3, 4, 5, 6)
                               else "Monsoon" if m in (7, 8, 9) else "PostMonsoon")
df.to_csv(f"{TAB}/cleaned_data.csv", index=False)
log(f"Clean dataset: {df.shape[0]} rows, {df['Date'].min().date()} to {df['Date'].max().date()}")

# =========================================================================
# 2. CLASSIFICATION, TABULATION, GRAPHS, DESCRIPTIVE STATISTICS  (EDA)
# =========================================================================
heading("2. EXPLORATORY DATA ANALYSIS")

# 2a. frequency table of AQI categories
freq = df["AQI_Bucket"].value_counts().reindex(BUCKET_ORDER).fillna(0).astype(int)
freq_table = pd.DataFrame({"Days": freq, "Percent": (freq / freq.sum() * 100).round(2)})
freq_table["Cumulative %"] = freq_table["Percent"].cumsum().round(2)
freq_table.to_csv(f"{TAB}/aqi_frequency_table.csv")
log("Frequency table of AQI categories:")
log(freq_table.to_string())

colors = ["#2e9e4f", "#8bc34a", "#fdd835", "#fb8c00", "#e53935", "#7b1fa2"]
ax = freq.plot(kind="bar", color=colors, edgecolor="black")
ax.set_title(f"{CITY}: Number of days per AQI category")
ax.set_xlabel("")
ax.set_ylabel("Days")
plt.xticks(rotation=30)
savefig("01_bar_aqi_categories")

freq_nz = freq[freq > 0]
plt.figure(figsize=(6, 6))
plt.pie(freq_nz, labels=freq_nz.index, autopct="%1.1f%%",
        colors=[colors[BUCKET_ORDER.index(b)] for b in freq_nz.index])
plt.title(f"{CITY}: share of AQI categories")
savefig("02_pie_aqi_categories")

plt.figure(figsize=(8, 5))
sns.histplot(df["AQI"], bins=30, kde=True, color="steelblue")
plt.title("Histogram of AQI")
savefig("03_hist_aqi")

plt.figure(figsize=(12, 4))
plt.plot(df["Date"], df["AQI"], lw=0.7, color="firebrick")
plt.title(f"{CITY}: daily AQI over time")
plt.ylabel("AQI")
savefig("04_line_aqi_time")

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Season", y="AQI",
            order=["Winter", "PostMonsoon", "Summer", "Monsoon"])
plt.title("AQI by season")
savefig("05_box_aqi_season")

# 2b. descriptive statistics incl. skewness and kurtosis
rows = {}
for c in ["PM2.5", "AQI"]:
    x = df[c]
    rows[c] = {
        "Mean": x.mean(), "Median": x.median(), "Mode": x.round(0).mode().iloc[0],
        "Std Dev": x.std(), "Variance": x.var(), "Min": x.min(), "Max": x.max(),
        "Skewness": stats.skew(x), "Kurtosis (excess)": stats.kurtosis(x),
    }
desc = pd.DataFrame(rows).round(3)
desc.to_csv(f"{TAB}/descriptive_statistics.csv")
log("\nDescriptive statistics:")
log(desc.to_string())
for c in ["PM2.5", "AQI"]:
    sk, ku = desc.loc["Skewness", c], desc.loc["Kurtosis (excess)", c]
    shape = "right-skewed" if sk > 0.5 else "left-skewed" if sk < -0.5 else "roughly symmetric"
    tail = "heavy-tailed (leptokurtic)" if ku > 0.5 else "light-tailed (platykurtic)" if ku < -0.5 else "close to normal tails"
    log(f"  -> {c}: {shape} (skew={sk:.2f}), {tail} (kurtosis={ku:.2f})")

# 2c. bivariate / multivariate
plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x="PM2.5", y="AQI", alpha=0.4)
plt.title("AQI vs PM2.5")
savefig("06_scatter_pm25_aqi")

plt.figure(figsize=(8, 6))
corr = df[POLLUTANTS + ["AQI"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation heatmap")
savefig("07_correlation_heatmap")
corr.round(3).to_csv(f"{TAB}/correlation_matrix.csv")
r, p = stats.pearsonr(df["PM2.5"], df["AQI"])
log(f"\nPearson correlation PM2.5 vs AQI: r = {r:.3f} (p = {p:.2e})")
log("Correlation of every pollutant with AQI:")
log(corr["AQI"].drop("AQI").sort_values(ascending=False).round(3).to_string())

# =========================================================================
# 3. POINT AND INTERVAL ESTIMATES
# =========================================================================
heading("3. POINT AND INTERVAL ESTIMATION")
n = len(df)
m, se = df["AQI"].mean(), stats.sem(df["AQI"])
ci = stats.t.interval(0.95, n - 1, loc=m, scale=se)
log(f"Point estimate of mean AQI : {m:.2f}")
log(f"Standard error             : {se:.3f}")
log(f"95% confidence interval    : ({ci[0]:.2f}, {ci[1]:.2f})")
p_unhealthy = (df["AQI"] > 200).mean()
se_p = np.sqrt(p_unhealthy * (1 - p_unhealthy) / n)
log(f"Proportion of days AQI > 200: {p_unhealthy:.3f}, 95% CI "
    f"({p_unhealthy - 1.96 * se_p:.3f}, {p_unhealthy + 1.96 * se_p:.3f})")

# =========================================================================
# 4. LINEAR REGRESSION: AQI ~ PM2.5
# =========================================================================
heading("4. SIMPLE LINEAR REGRESSION (AQI ~ PM2.5)")
slr = smf.ols("AQI ~ Q('PM2.5')", data=df).fit()
log(slr.summary().as_text())
log(f"R-squared = {slr.rsquared:.3f}; slope p-value = {slr.pvalues.iloc[1]:.2e}")
log("95% CI for coefficients:\n" + slr.conf_int().round(3).to_string())

grid = pd.DataFrame({"PM2.5": np.linspace(df["PM2.5"].min(), df["PM2.5"].max(), 100)})
pred = slr.get_prediction(grid).summary_frame(alpha=0.05)
plt.figure(figsize=(8, 5.5))
plt.scatter(df["PM2.5"], df["AQI"], s=8, alpha=0.3, label="Observed")
plt.plot(grid["PM2.5"], pred["mean"], color="red", label="Fitted line")
plt.fill_between(grid["PM2.5"], pred["mean_ci_lower"], pred["mean_ci_upper"],
                 color="red", alpha=0.35, label="95% confidence interval")
plt.fill_between(grid["PM2.5"], pred["obs_ci_lower"], pred["obs_ci_upper"],
                 color="orange", alpha=0.18, label="95% prediction interval")
plt.xlabel("PM2.5")
plt.ylabel("AQI")
plt.legend()
plt.title("Simple linear regression with confidence and prediction intervals")
savefig("08_linear_regression_intervals")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
ax[0].scatter(slr.fittedvalues, slr.resid, s=8, alpha=0.4)
ax[0].axhline(0, color="red")
ax[0].set(title="Residuals vs fitted", xlabel="Fitted", ylabel="Residual")
sm.qqplot(slr.resid, line="s", ax=ax[1])
ax[1].set_title("Normal Q-Q of residuals")
sns.histplot(slr.resid, kde=True, ax=ax[2])
ax[2].set_title("Residual distribution")
savefig("09_linear_residual_analysis")

# =========================================================================
# 5. MULTIPLE + POLYNOMIAL REGRESSION
# =========================================================================
heading("5. MULTIPLE AND POLYNOMIAL REGRESSION")
train, test = train_test_split(df, test_size=0.2, random_state=42)

formulas = {
    "Linear (PM2.5)": "AQI ~ Q('PM2.5')",
    "Polynomial (PM2.5 + PM2.5^2)": "AQI ~ Q('PM2.5') + I(Q('PM2.5')**2)",
    "Multiple (all pollutants)": "AQI ~ Q('PM2.5') + Q('PM10') + NO2 + SO2 + CO + O3",
    "Multiple + Season dummies": "AQI ~ Q('PM2.5') + Q('PM10') + NO2 + SO2 + CO + O3 + C(Season)",
}
res_rows, fitted_models = [], {}
for name, f in formulas.items():
    mod = smf.ols(f, data=train).fit()
    fitted_models[name] = mod
    res_rows.append({
        "Model": name, "R2": mod.rsquared, "Adj R2": mod.rsquared_adj,
        "AIC": mod.aic, "Test RMSE": rmse(test["AQI"], mod.predict(test)),
    })
comp = pd.DataFrame(res_rows).round(3)
comp.to_csv(f"{TAB}/regression_model_comparison.csv", index=False)
log("Model comparison (fit on 80% train, RMSE on 20% test):")
log(comp.to_string(index=False))
best_reg = comp.sort_values("Test RMSE").iloc[0]["Model"]
log(f"Best regression model by test RMSE: {best_reg}")

mlr = fitted_models["Multiple + Season dummies"]
log("\n" + mlr.summary().as_text())

# multicollinearity: VIF
X = df[POLLUTANTS]
Xc = sm.add_constant(X)
vif = pd.DataFrame({"Variable": POLLUTANTS,
                    "VIF": [variance_inflation_factor(Xc.values, i + 1) for i in range(len(POLLUTANTS))]}).round(2)
vif.to_csv(f"{TAB}/vif.csv", index=False)
log("\nMulticollinearity check (VIF > 5 suggests a problem, > 10 is severe):")
log(vif.to_string(index=False))

# heteroscedasticity, autocorrelation
bp = het_breuschpagan(mlr.resid, mlr.model.exog)
log(f"\nBreusch-Pagan test: LM = {bp[0]:.2f}, p = {bp[1]:.4f} -> "
    + ("heteroscedasticity present" if bp[1] < 0.05 else "no evidence of heteroscedasticity"))
log(f"Durbin-Watson statistic: {durbin_watson(mlr.resid):.3f} (about 2 = no autocorrelation)")

# influential outliers: Cook's distance
cooks = mlr.get_influence().cooks_distance[0]
thr = 4 / len(train)
log(f"Influential observations (Cook's D > 4/n): {(cooks > thr).sum()} of {len(train)}")
plt.figure(figsize=(9, 4))
plt.stem(cooks, markerfmt=",", basefmt=" ")
plt.axhline(thr, color="red", ls="--")
plt.title("Cook's distance (outlier influence)")
savefig("10_cooks_distance")

# coefficient interpretation
log("\nInterpretation of key coefficients (multiple regression):")
for v in ["Q('PM2.5')", "Q('PM10')", "NO2", "SO2", "CO", "O3"]:
    log(f"  A 1-unit rise in {v.replace(chr(81)+'(', '').replace(')', '').replace(chr(39), '')} changes AQI by "
        f"{mlr.params[v]:+.3f} (p = {mlr.pvalues[v]:.3g}), other variables held constant")

plt.figure(figsize=(6, 6))
plt.scatter(test["AQI"], fitted_models[best_reg].predict(test), s=10, alpha=0.5)
lim = [test["AQI"].min(), test["AQI"].max()]
plt.plot(lim, lim, "r--")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title(f"Actual vs predicted ({best_reg})")
savefig("11_actual_vs_predicted")

# =========================================================================
# 6. LOGISTIC REGRESSION: Unhealthy day (AQI > 200)
# =========================================================================
heading("6. LOGISTIC REGRESSION (Unhealthy day: AQI > 200)")
df["Unhealthy"] = (df["AQI"] > 200).astype(int)
log(f"Unhealthy days: {df['Unhealthy'].sum()} of {len(df)} ({df['Unhealthy'].mean() * 100:.1f}%)")

feats = ["PM2.5", "NO2", "SO2", "CO", "O3"]
Xs = (df[feats] - df[feats].mean()) / df[feats].std()          # standardised for stable fitting
ytr_idx, yte_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=df["Unhealthy"])
Xtr, Xte = sm.add_constant(Xs.loc[ytr_idx]), sm.add_constant(Xs.loc[yte_idx])
ytr, yte = df.loc[ytr_idx, "Unhealthy"], df.loc[yte_idx, "Unhealthy"]

try:
    logit = sm.Logit(ytr, Xtr).fit(disp=0, maxiter=200)
    log(logit.summary().as_text())
    odds = pd.DataFrame({"Odds ratio (per 1 SD)": np.exp(logit.params),
                         "p-value (Wald z-test)": logit.pvalues}).round(4)
    log("\nOdds ratios and Wald test p-values:")
    log(odds.to_string())
    prob_te = logit.predict(Xte)

    # Hosmer-Lemeshow goodness of fit (10 groups)
    d = pd.DataFrame({"y": ytr, "p": logit.predict(Xtr)})
    d["g"] = pd.qcut(d["p"], 10, duplicates="drop")
    hl = d.groupby("g", observed=True).agg(obs=("y", "sum"), exp=("p", "sum"), n=("y", "size"))
    hl_stat = (((hl["obs"] - hl["exp"]) ** 2) / (hl["exp"] * (1 - hl["exp"] / hl["n"]))).sum()
    hl_p = 1 - stats.chi2.cdf(hl_stat, len(hl) - 2)
    log(f"\nHosmer-Lemeshow test: chi2 = {hl_stat:.3f}, p = {hl_p:.3f} -> "
        + ("poor fit" if hl_p < 0.05 else "model fits adequately (p > 0.05)"))

    pred_cls = (prob_te >= 0.5).astype(int)
    cm = confusion_matrix(yte, pred_cls)
    acc_log = accuracy_score(yte, pred_cls)
    auc = roc_auc_score(yte, prob_te)
    gini = 2 * auc - 1
    log("\nClassification table (rows = actual, columns = predicted):")
    log(pd.DataFrame(cm, index=["Actual Healthy", "Actual Unhealthy"],
                     columns=["Pred Healthy", "Pred Unhealthy"]).to_string())
    log(f"Accuracy = {acc_log:.3f}, AUC = {auc:.3f}, Gini coefficient = 2*AUC - 1 = {gini:.3f}")
    log(classification_report(yte, pred_cls, target_names=["Healthy", "Unhealthy"], zero_division=0))

    plt.figure(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Healthy", "Unhealthy"], yticklabels=["Healthy", "Unhealthy"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Logistic regression confusion matrix")
    savefig("12_logistic_confusion_matrix")
except Exception as e:                                           # e.g. perfect separation
    log(f"Logistic regression failed: {e}")
    acc_log, gini = float("nan"), float("nan")

# =========================================================================
# 7. DECISION TREE (CART): AQI category
# =========================================================================
heading("7. DECISION TREE - CART (Good / Moderate / Poor)")
df["Class3"] = pd.cut(df["AQI"], [-1, 100, 200, 1000], labels=["Good", "Moderate", "Poor"])
Xt = df[POLLUTANTS + ["Month"]]
Xt_tr, Xt_te, yt_tr, yt_te = train_test_split(Xt, df["Class3"], test_size=0.2,
                                              random_state=42, stratify=df["Class3"])
tree = DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_leaf=20, random_state=42)
tree.fit(Xt_tr, yt_tr)
tp = tree.predict(Xt_te)
acc_tree = accuracy_score(yt_te, tp)
labels = list(tree.classes_)
log(f"CART accuracy on test data: {acc_tree:.3f}")
log("Confusion matrix (rows = actual, cols = predicted; order " + str(labels) + "):")
log(pd.DataFrame(confusion_matrix(yt_te, tp, labels=labels), index=labels, columns=labels).to_string())
imp = pd.Series(tree.feature_importances_, index=Xt.columns).sort_values(ascending=False).round(3)
log("Feature importance (Gini based):\n" + imp.to_string())
imp.to_csv(f"{TAB}/tree_feature_importance.csv")

plt.figure(figsize=(18, 8))
plot_tree(tree, feature_names=list(Xt.columns), class_names=labels, filled=True,
          rounded=True, fontsize=8)
plt.title("CART decision tree for AQI category")
savefig("13_decision_tree")

log("\nLogistic vs CART:")
log(f"  Logistic regression (Unhealthy vs not) accuracy = {acc_log:.3f}")
log(f"  CART (3-class) accuracy                         = {acc_tree:.3f}")

# =========================================================================
# 8. TIME SERIES: moving average, exponential smoothing, ARIMA
# =========================================================================
heading("8. TIME SERIES ANALYSIS AND FORECASTING")
ts = df.set_index("Date")["AQI"].resample("MS").mean().interpolate()
log(f"Monthly series: {len(ts)} months ({ts.index.min():%b %Y} to {ts.index.max():%b %Y})")

# moving averages
ma3, ma6 = ts.rolling(3).mean(), ts.rolling(6).mean()
plt.figure(figsize=(12, 4.5))
plt.plot(ts, label="Monthly mean AQI", color="grey")
plt.plot(ma3, label="3-month moving average", color="blue")
plt.plot(ma6, label="6-month moving average", color="red")
plt.legend()
plt.title("Moving average smoothing")
savefig("14_moving_average")

# stationarity
adf_p = adfuller(ts)[1]
log(f"ADF stationarity test p-value = {adf_p:.4f} -> " + ("stationary" if adf_p < 0.05
    else "non-stationary; differencing (d=1) considered in ARIMA"))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(ts, lags=24, ax=ax[0])
plot_pacf(ts, lags=24, ax=ax[1], method="ywm")
savefig("15_acf_pacf")

# train / test split: hold out the last 6 months
H = 6
tr, te = ts.iloc[:-H], ts.iloc[-H:]
scores = {}

# naive moving-average forecast (flat at last 3-month average)
scores["Moving average (3)"] = te.index.map(lambda _: tr.iloc[-3:].mean())
ses = SimpleExpSmoothing(tr, initialization_method="estimated").fit()
scores["Simple exp. smoothing"] = ses.forecast(H)
hw = ExponentialSmoothing(tr, trend=None, seasonal="add", seasonal_periods=12,
                          initialization_method="estimated").fit()
scores["Holt-Winters (seasonal)"] = hw.forecast(H)

# ARIMA order search by AIC
best = (None, np.inf)
for p in range(3):
    for d in range(2):
        for q in range(3):
            try:
                a = ARIMA(tr, order=(p, d, q)).fit()
                if a.aic < best[1]:
                    best = ((p, d, q), a.aic)
            except Exception:
                pass
order = best[0]
log(f"Best ARIMA order by AIC: {order}  (AIC = {best[1]:.1f})")
arima = ARIMA(tr, order=order).fit()
scores["ARIMA" + str(order)] = arima.forecast(H)

acc_rows = []
for k, v in scores.items():
    acc_rows.append({"Method": k, "RMSE": rmse(te, v), "MAPE %": mape(te, v)})
acc = pd.DataFrame(acc_rows).round(2)
acc.to_csv(f"{TAB}/forecast_accuracy.csv", index=False)
log("\nForecast accuracy on last 6 months (lower is better):")
log(acc.to_string(index=False))
best_ts = acc.sort_values("RMSE").iloc[0]["Method"]
log(f"Most accurate method: {best_ts}")

plt.figure(figsize=(12, 4.5))
plt.plot(tr.iloc[-24:], label="Training data", color="grey")
plt.plot(te, label="Actual", color="black", lw=2)
for k, v in scores.items():
    plt.plot(te.index, np.asarray(v), marker="o", ms=3, label=k)
plt.legend(fontsize=8)
plt.title("Forecast comparison on hold-out months")
savefig("16_forecast_comparison")

# final forecast: refit on ALL data, forecast next 6 months with ARIMA and Holt-Winters
full_arima = ARIMA(ts, order=order).fit()
fc = full_arima.get_forecast(6)
fc_df = fc.summary_frame(alpha=0.05)
hw_full = ExponentialSmoothing(ts, seasonal="add", seasonal_periods=12,
                               initialization_method="estimated").fit()
hw_fc = hw_full.forecast(6)
out = pd.DataFrame({"ARIMA forecast": fc_df["mean"], "ARIMA lower 95%": fc_df["mean_ci_lower"],
                    "ARIMA upper 95%": fc_df["mean_ci_upper"], "Holt-Winters forecast": hw_fc}).round(1)
out.to_csv(f"{TAB}/next_6_months_forecast.csv")
log("\nForecast for the next 6 months:")
log(out.to_string())

plt.figure(figsize=(12, 4.5))
plt.plot(ts.iloc[-36:], label="Observed", color="black")
plt.plot(fc_df["mean"], color="red", label=f"ARIMA{order} forecast")
plt.fill_between(fc_df.index, fc_df["mean_ci_lower"], fc_df["mean_ci_upper"], color="red", alpha=0.2)
plt.plot(hw_fc, color="green", ls="--", label="Holt-Winters forecast")
plt.legend()
plt.title("Next 6 months AQI forecast")
savefig("17_final_forecast")

# =========================================================================
# 9. CONCLUSIONS
# =========================================================================
heading("9. CONCLUSIONS")
top_corr = corr["AQI"].drop("AQI").idxmax()
season_mean = df.groupby("Season")["AQI"].mean().sort_values(ascending=False).round(1)
log(f"1. {top_corr} has the strongest correlation with AQI (r = {corr.loc[top_corr, 'AQI']:.2f}).")
log(f"2. Simple regression on PM2.5 alone explains {slr.rsquared * 100:.1f}% of AQI variation; "
    f"best regression model (test RMSE) is: {best_reg}.")
log(f"3. Average AQI by season: " + ", ".join(f"{k} {v}" for k, v in season_mean.items()) + ".")
log(f"4. Mean AQI is {m:.1f} with 95% CI ({ci[0]:.1f}, {ci[1]:.1f}); "
    f"{p_unhealthy * 100:.1f}% of days exceed AQI 200.")
log(f"5. Logistic regression accuracy {acc_log:.2f} (Gini {gini:.2f}); CART accuracy {acc_tree:.2f}.")
log(f"6. Best forecasting method on hold-out data: {best_ts}.")

with open("outputs/summary.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(summary))
print("\nDone. See the 'outputs' folder for figures, tables and summary.txt")
