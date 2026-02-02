# 📈 Retail Sales Time Series Forecasting & Anomaly Detection

> **Business-focused time series case study**
> Forecasting retail demand and detecting abnormal sales patterns
> to support inventory planning and operational decision-making.

End-to-end applied machine learning project focused on time series forecasting and anomaly detection using real-world retail sales data.

---

## 📌 Project Overview

**Goal:**
Build a robust forecasting model for daily retail sales and detect abnormal demand patterns that may correspond to holidays, promotions, or unexpected events.

**Key aspects:**
- Real-world multiyear dataset
- Clear business framing
- Strong baselines
- Machine learning forecasting
- Residual-based anomaly detection
- Interpretable results

---

## 📊 Dataset

- **Source:** Kaggle — Store Sales Time Series Forecasting
- **Time range:** 2013-01-01 → 2017-08-15
- **Granularity:** daily sales

**Data aggregated across:**
- all stores
- all product categories

**Target variable:**
- `total_sales` — total daily sales volume

---

## 🧠 Problem Formulation

These tasks reflect real-world retail needs:
accurate demand planning and early detection
of unusual demand behavior.


1. **Demand forecasting**
   Predict future daily total sales based on historical patterns.

2. **Anomaly detection**
   Identify days with unusually high or low demand compared to model expectations.

---

## 🔍 Exploratory Data Analysis (EDA)

**Key findings:**
- Strong weekly seasonality
- Clear annual seasonality
- Presence of extreme peaks and drops
- Sales distribution is right-skewed with heavy tails

These insights guided feature engineering and model choice.

---

## 📉 Baseline Models

Two simple but informative baselines were implemented:

| Model                | Description        |
|----------------------|--------------------|
| Naive (lag-1)        | Yesterday’s sales  |
| Moving Average (7 days) | Weekly smoothing   |

**Baseline performance (test period):**
- MAE ≈ 144k
- RMSE ≈ 187k
- MAPE ≈ 53%

---

## 🤖 Machine Learning Forecasting

**Model:**
- LightGBM Regressor

**Features:**

- **Calendar features:**
  - `day`, `month`, `dayofweek`, `weekofyear`, `is_weekend`
- **Lag features:**
  - `lag_1`, `lag_7`, `lag_14`, `lag_28`
- **Rolling statistics:**
  - `rolling_mean_7`, `rolling_mean_14`

### Results (Test Period)

| Metric | Value    |
|--------|---------:|
| MAE    | ≈ 91,662 |
| RMSE   | ≈ 136,100|
| MAPE   | ≈ 43.0%  |

**Improvement vs baseline:**
- MAE ↓ ~36%
- RMSE ↓ ~27%
- MAPE ↓ ~10 pp

The model was trained using a strictly time-based split
to avoid data leakage and ensure realistic evaluation.


---

## 🚨 Anomaly Detection

Anomaly detection was performed using model residuals:

\[
\text{residual} = \text{actual\_sales} - \text{predicted\_sales}
\]

**Method:**
- Z-score applied to residuals
- Threshold: \(|z| > 3\)

**Findings:**
- 4 strong anomalies detected
- Most notable dates:
  - 2017-01-01 — extremely low sales (New Year)
  - 2017-01-02 — post-holiday demand spike
  - Early April and May — significant positive deviations

These anomalies correspond to meaningful real-world events, not random noise.

---

## 📊 Residual Analysis

- Residuals are centered around zero → no systematic bias
- Heavy tails indicate rare but impactful demand shocks
- Confirms the usefulness of anomaly detection alongside forecasting models

---

## 🧩 Project Structure

```text
time_series_retail_forecasting/
├── data/
│   └── raw/
│       └── train.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_ml_forecasting.ipynb
│   └── 04_anomaly_detection.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ✅ Key Takeaway

Tree-based ML models with lagged and rolling features
provide a strong, interpretable baseline for retail demand forecasting,
while residual analysis enables effective anomaly detection
without complex statistical assumptions.

---

## 🏁 Final Conclusions

- Tree-based ML models with lag and rolling features are highly effective for retail demand forecasting.
- LightGBM significantly outperforms naive and moving average baselines.
- Residual-based anomaly detection successfully identifies business-critical irregularities.
- The pipeline reflects a realistic production-style approach to forecasting and monitoring time series data.

## 🚀 Future Improvements

- Store-level and category-level forecasting
- Incorporation of promotions and holidays
- Hyperparameter tuning
- Probabilistic forecasting
- Deployment as a monitoring service

## 👤 Author

Developed as part of a personal Machine Learning portfolio focused on applied, production-oriented ML systems.

**Albert Nubaryan**
📧 Email: [alnub_work@mail.ru](mailto:alnub_work@mail.ru)
