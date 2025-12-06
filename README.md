SmartRemit AI – Remittance Fee Optimization Decision Support System
==================================================================

Last updated: 06 December 2025

Overview
--------

SmartRemit AI is a modular, data-driven decision support system that helps users sending money from USD to KES optimize three things:

1. Total remittance cost  
2. Provider choice (cost, speed, stability)  
3. Timing of transfers (FX rate forecasting)

The project combines real-world remittance data (World Bank RPW), historical and live FX rates, machine learning models, anomaly detection, and an interactive Streamlit dashboard.

Main Goals
----------

- Provide transparent provider cost comparisons for the USA→Kenya corridor  
- Forecast short-term FX movements to support “send now” vs “wait” decisions  
- Detect anomalous FX and cost events to protect users from bad timing  
- Offer a user-friendly decision support interface for senders, analysts, and policymakers  

System Architecture
-------------------

The system follows a layered, modular architecture:

1. Data Layer  
   - World Bank RPW data (Excel / CSV)  
   - Historical FX rates (Yahoo Finance → cbk_usd_kes_history.csv)  
   - Optional live FX API (Wise, RapidAPI)  
   - Synthetic data generator (fallback)

2. Processing Layer  
   - `data_loader.py` – robust RPW ingestion and cleaning for USA→Kenya  
   - Data validation, parsing of quarters into dates, corridor filtering  
   - Feature engineering for cost, FX margin, and stability metrics  

3. Analytics Layer  
   - `fx-forecasting-expanded.py` – 7-model FX forecasting engine  
     - ARIMA, Prophet, Linear Regression, Random Forest, XGBoost, LightGBM, Voting Ensemble  
   - Model evaluation and comparison (MAE, RMSE, MAPE, R²)  
   - Future extension: clustering and anomaly detection on provider data  

4. Presentation Layer  
   - `streamlit_dashboard_v4.py` – main dashboard  
     - Live FX card (from `cbk_usd_kes_history.csv`)  
     - Provider comparison and filtering  
     - Model performance panel (from `model_comparison.csv`)  
     - Simple rule-based chatbot  
     - Basic gamification and personalization  
   - Future: REST API endpoints and richer frontends  

5. Storage Layer  
   - Cleaned RPW data (CSV)  
   - Historical FX (`cbk_usd_kes_history.csv`)  
   - Model performance summary (`model_comparison.csv`)  
   - Optional anomaly and log files (JSON/CSV)

Key Components
--------------

### 1. Data Ingestion – `data_loader.py`

**Class:** `RemittanceDataLoader`

**Responsibilities:**

- Load World Bank RPW Excel file with multiple sheets  
- Identify and combine relevant data sheets  
- Filter strictly for the USA→Kenya corridor  
- Standardize columns (cost, FX rate, margin, provider, service type)  
- Parse RPW “period” column (e.g., `2015_1Q`) into proper dates  

**Main methods:**

- `load_data(filepath)` – entry point for full processing  
- `_filter_usa_kenya(df)` – select sending=USA, receiving=Kenya  
- `_process_cost_components(df)` – cost/margin calculation and mapping  

Use this script to generate a clean USA→Kenya dataset before modeling or visualization.

### 2. FX Forecasting – `fx-forecasting-expanded.py`

**Class:** `ComprehensiveFXForecaster`

**Responsibilities:**

- Load and clean historical FX data (`cbk_usd_kes_history.csv`)  
- Create lag and rolling-window features for ML models  
- Train and evaluate 7 forecasting models:  
  - ARIMA  
  - Prophet (if installed)  
  - Linear Regression  
  - Random Forest  
  - XGBoost  
  - LightGBM (if installed)  
  - VotingRegressor ensemble  

**Key features:**

- `train_test_split(test_days=90)` – time-based split  
- `create_lag_features(df)` – lags and rolling stats  
- `train_[model]` methods – one per model type  
- Metrics: MAE, RMSE, MAPE, R² for each model  
- Saves `model_comparison.csv` for dashboard consumption  

After running this script, `model_comparison.csv` will contain model rankings, which the dashboard can display. LightGBM usually emerges as the best model if available.

### 3. Dashboard – `streamlit_dashboard_v4.py`

**Purpose:** End-to-end decision support interface.

**Main features:**

- Live FX card (with cached reload every 5 minutes by default)  
- Provider comparison (`provider_comparison.csv` or fallback sample data)  
- Service type filtering (Bank, Mobile Money, Cash Pickup)  
- Model performance table (`model_comparison.csv`)  
- Rule-based chatbot (basic Q&A about rates and cheapest provider)  
- Simple personalization via Streamlit session state  
- Gamification: tracked savings and transfer count  

**Data sources:**

- FX: `cbk_usd_kes_history.csv`  
- Providers: `provider_comparison.csv`  
- Models: `model_comparison.csv`  

To run:

```bash
python fx-forecasting-expanded.py       # optional, updates model_comparison.csv
python Yahoo_fx_loader_enhanced.py      # optional, updates cbk_usd_kes_history.csv
python -m streamlit run streamlit_dashboard_v4.py
```

Project Structure
-----------------


```text
SmartRemitAI/
├── data_loader.py                 # RPW data ingestion
├── fx-forecasting-expanded.py     # FX forecasting models
├── streamlit_dashboard_v4.py      # Main dashboard
├── Yahoo_fx_loader_enhanced.py    # FX history updater (Yahoo Finance)
├── cbk_usd_kes_history.csv        # Historical USD/KES rates
├── provider_comparison.csv        # Provider cost data
├── model_comparison.csv           # Model performance metrics
├── rpw_dataset_2011_2025_q1.xlsx  # World Bank RPW raw data
└── README.md                      # This file
```

Installation
------------

1. Python 3.10+ recommended.  
2. Install dependencies (adjust as needed):

```bash
pip install streamlit pandas numpy plotly statsmodels prophet scikit-learn xgboost lightgbm yfinance openpyxl
```

3. Optional: For faster Prophet or LightGBM, follow each library’s official installation guide.

How Components Work Together
----------------------------

1. Data loader ingests and filters RPW data to USA→Kenya corridor.  
2. FX loader or Yahoo Finance script builds and maintains `cbk_usd_kes_history.csv`.  
3. `fx-forecasting-expanded.py` reads FX history, trains models, and writes `model_comparison.csv`.  
4. `streamlit_dashboard_v4.py` reads:  
   - FX history for the live FX card and historical charts  
   - Provider comparison for cheapest/best provider logic  
   - Model performance for explaining forecast reliability  
5. Users interact with the dashboard:  
   - Choose amount, service type  
   - View providers sorted by cost and payout  
   - See current FX and model-based timing advice  
   - Ask basic questions via the rule-based chatbot  

Running the Full System
-----------------------

Typical workflow:

1. Update FX history (once per day or more often):

```bash
python Yahoo_fx_loader_enhanced.py
```

2. (Optional but recommended) retrain forecasting models:

```bash
python fx-forecasting-expanded.py
```

3. Launch the dashboard:

```bash
python -m streamlit run streamlit_dashboard_v7.py
```

Future Extensions
-----------------

- Replace or augment the rule-based chatbot with an LLM-backed assistant.  
- Implement provider clustering and anomaly detection using RPW and historical provider data.  
- Add REST API endpoints to serve forecasts and recommendations to external clients.  
- Replace CSV persistence with a database (PostgreSQL, SQLite) for robustness and multi-user support.  
- Add a more advanced live FX source (CBK API or commercial FX feeds).


[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/75509243/9645fa87-a209-4281-9c17-198b48800118/fx-forecasting-expanded.py)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/75509243/a8e938b4-0118-4155-83b2-271c247e329f/streamlit_dashboard_v4.py)
