"""
fx_forecasting_models_expanded_v2.py
SmartRemit AI - Comprehensive FX Rate Forecasting (Improved Evaluation)

Key improvements vs v1:
- Clear time-based train/test split (no shuffling)
- Consistent feature engineering with careful NA handling
- Train vs test metrics for ML models
- Directional accuracy metric
- Optional regime-based evaluation (normal vs anomaly days, if column exists)
- More robust ARIMA / Prophet evaluation

Author: Muita Shalyn J. - USIU-A
Date: November 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical models
from statsmodels.tsa.arima.model import ARIMA

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("⚠️ Prophet not installed. Install with: pip install prophet")
    PROPHET_AVAILABLE = False

# ML models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Gradient Boosting
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("⚠️ LightGBM not installed. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False


class ComprehensiveFXForecaster:
    """
    7-model FX forecasting system with improved evaluation.
    """

    def __init__(self, fx_data_path='cbk_usd_kes_history.csv'):
        print("=" * 70)
        print("COMPREHENSIVE FX FORECASTING SYSTEM (Improved Evaluation)")
        print("7 Models: ARIMA | Prophet | Linear Reg | Random Forest | XGBoost | LightGBM | Ensemble")
        print("=" * 70)

        df = pd.read_csv(fx_data_path)
        df['date'] = pd.to_datetime(df['date'])
        df['fx_usd_kes'] = pd.to_numeric(df['fx_usd_kes'], errors='coerce')
        df = df.dropna(subset=['fx_usd_kes']).sort_values('date').reset_index(drop=True)

        self.df = df
        print(f"\n✓ Loaded {len(self.df)} historical FX rates")
        print(f"  Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"  Latest rate: {self.df['fx_usd_kes'].iloc[-1]:.4f} KES/USD")

        # Check if anomaly flag exists (optional)
        self.has_anomaly_flag = 'is_fx_anomaly' in self.df.columns
        if self.has_anomaly_flag:
            print("  Anomaly flag 'is_fx_anomaly' detected for regime evaluation.")

        self.models = {}
        self.metrics = {}      # test metrics
        self.train_metrics = {}  # train metrics
        self.directions = {}   # directional accuracy

    # =======================================================================
    # SPLIT & FEATURES
    # =======================================================================

    def train_test_split(self, test_days=90):
        """Strict time-based train/test split: last N days = test."""
        split_date = self.df['date'].max() - timedelta(days=test_days)
        train = self.df[self.df['date'] <= split_date].copy()
        test = self.df[self.df['date'] > split_date].copy()

        print("\n📊 Train/Test Split (time-based):")
        print(f"  Train: {len(train)} days ({train['date'].min()} → {train['date'].max()})")
        print(f"  Test:  {len(test)} days ({test['date'].min()} → {test['date'].max()})")

        return train, test

    def create_lag_features(self, df, lags=[1, 7, 14, 30]):
        """Create lag + rolling + calendar features; drop NAs from lagging."""
        df = df.copy()

        # Lag features
        for lag in lags:
            df[f'lag_{lag}'] = df['fx_usd_kes'].shift(lag)

        # Rolling statistics (use min_periods to avoid dropping too much early data)
        df['rolling_mean_7'] = df['fx_usd_kes'].rolling(7, min_periods=3).mean()
        df['rolling_std_7'] = df['fx_usd_kes'].rolling(7, min_periods=3).std()
        df['rolling_mean_30'] = df['fx_usd_kes'].rolling(30, min_periods=7).mean()
        df['rolling_std_30'] = df['fx_usd_kes'].rolling(30, min_periods=7).std()

        # Time features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter

        # Drop rows where any feature is NA (mainly early rows)
        df = df.dropna(subset=[c for c in df.columns if c not in ['is_fx_anomaly']])

        return df

    # =======================================================================
    # MODELS
    # =======================================================================

    def train_arima(self, train_data, order=(5, 1, 0)):
        print("\n🔄 [1/7] Training ARIMA...")
        try:
            model = ARIMA(train_data['fx_usd_kes'], order=order)
            fitted = model.fit()
            self.models['arima'] = fitted
            print(f"✓ ARIMA{order} trained (AIC: {fitted.aic:.2f})")
            return fitted
        except Exception as e:
            print(f"⚠️ ARIMA failed: {str(e)[:120]}")
            return None

    def train_prophet(self, train_data):
        if not PROPHET_AVAILABLE:
            print("\n⚠️ [2/7] Skipping Prophet (not installed)")
            return None

        print("\n🔄 [2/7] Training Prophet...")
        try:
            prophet_df = train_data[['date', 'fx_usd_kes']].rename(
                columns={'date': 'ds', 'fx_usd_kes': 'y'}
            )
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(prophet_df)
            self.models['prophet'] = model
            print("✓ Prophet trained")
            return model
        except Exception as e:
            print(f"⚠️ Prophet failed: {str(e)[:120]}")
            return None

    def _train_ml_model(self, name, model, train_data):
        """Helper to train any ML model with shared feature pipeline."""
        print(f"\n🔄 Training {name}...")
        try:
            fe = self.create_lag_features(train_data)
            feature_cols = [c for c in fe.columns if c not in ['date', 'fx_usd_kes', 'is_fx_anomaly']]
            X_train = fe[feature_cols]
            y_train = fe['fx_usd_kes']

            model.fit(X_train, y_train)

            self.models[name] = {'model': model, 'features': feature_cols}

            # Store train metrics for overfitting check
            y_pred_train = model.predict(X_train)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            train_mape = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
            train_r2 = r2_score(y_train, y_pred_train)

            self.train_metrics[name] = {
                'MAE': train_mae,
                'RMSE': train_rmse,
                'MAPE': train_mape,
                'R2': train_r2
            }

            print(f"✓ {name} trained ({len(feature_cols)} features, train R²: {train_r2:.4f}, train MAPE: {train_mape:.2f}%)")
            return model
        except Exception as e:
            print(f"⚠️ {name} failed: {str(e)[:120]}")
            return None

    def train_linear_regression(self, train_data):
        model = LinearRegression()
        return self._train_ml_model('linear_regression', model, train_data)

    def train_random_forest(self, train_data):
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        return self._train_ml_model('random_forest', model, train_data)

    def train_xgboost(self, train_data):
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        return self._train_ml_model('xgboost', model, train_data)

    def train_lightgbm(self, train_data):
        if not LIGHTGBM_AVAILABLE:
            print("\n⚠️ [6/7] Skipping LightGBM (not installed)")
            return None
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        return self._train_ml_model('lightgbm', model, train_data)

    def train_voting_ensemble(self, train_data):
        print("\n🔄 [7/7] Training Voting Ensemble...")
        try:
            fe = self.create_lag_features(train_data)
            feature_cols = [c for c in fe.columns if c not in ['date', 'fx_usd_kes', 'is_fx_anomaly']]
            X_train = fe[feature_cols]
            y_train = fe['fx_usd_kes']

            estimators = []

            # Use already-trained models' hyperparams but re‑instantiate
            if 'linear_regression' in self.models:
                estimators.append(('lr', LinearRegression()))
            if 'random_forest' in self.models:
                estimators.append(('rf', RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)))
            if 'xgboost' in self.models:
                estimators.append(('xgb', xgb.XGBRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)))

            if len(estimators) < 2:
                print("⚠️ Not enough base models to build ensemble")
                return None

            model = VotingRegressor(estimators=estimators)
            model.fit(X_train, y_train)

            self.models['voting_ensemble'] = {'model': model, 'features': feature_cols}

            # Train metrics
            y_pred_train = model.predict(X_train)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            train_mape = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
            train_r2 = r2_score(y_train, y_pred_train)

            self.train_metrics['voting_ensemble'] = {
                'MAE': train_mae,
                'RMSE': train_rmse,
                'MAPE': train_mape,
                'R2': train_r2
            }

            print(f"✓ Voting Ensemble trained (combining {len(estimators)} models, train R²: {train_r2:.4f})")
            return model

        except Exception as e:
            print(f"⚠️ Voting Ensemble failed: {str(e)[:120]}")
            return None

    # =======================================================================
    # EVALUATION
    # =======================================================================

    @staticmethod
    def _directional_accuracy(y_true, y_pred):
        """Percentage of times the model gets direction of change right."""
        # Compare sign of day-to-day change
        true_change = np.sign(np.diff(y_true))
        pred_change = np.sign(np.diff(y_pred))
        # Align lengths
        m = min(len(true_change), len(pred_change))
        if m == 0:
            return np.nan
        return (true_change[:m] == pred_change[:m]).mean() * 100

    def evaluate_model(self, model_name, train_data, test_data):
        """
        Evaluate model on test set + directional accuracy.
        For ML models, also supports anomaly vs normal regime evaluation if flag exists.
        """
        print(f"\n📈 Evaluating {model_name.upper()}...")

        try:
            # --------------------------- ARIMA ---------------------------
            if model_name == 'arima' and isinstance(self.models.get('arima'), ARIMA.__mro__[0]):
                fitted = self.models['arima']
                steps = len(test_data)
                preds = fitted.forecast(steps=steps)
                y_true = test_data['fx_usd_kes'].values

            # --------------------------- PROPHET -------------------------
            elif model_name == 'prophet' and PROPHET_AVAILABLE and self.models.get('prophet') is not None:
                model = self.models['prophet']
                future = pd.DataFrame({'ds': test_data['date']})
                forecast = model.predict(future)
                preds = forecast['yhat'].values
                y_true = test_data['fx_usd_kes'].values

            # ----------------------------- ML ----------------------------
            elif model_name in ['linear_regression', 'random_forest', 'xgboost', 'lightgbm', 'voting_ensemble'] \
                    and self.models.get(model_name):

                # Build features on full data, then slice strictly to *test dates*
                fe_all = self.create_lag_features(pd.concat([train_data, test_data], ignore_index=True))
                feature_cols = self.models[model_name]['features']

                fe_test = fe_all[fe_all['date'].isin(test_data['date'])]
                # Safety: drop any rows with NA in features
                fe_test = fe_test.dropna(subset=feature_cols + ['fx_usd_kes'])

                X_test = fe_test[feature_cols]
                y_true = fe_test['fx_usd_kes'].values
                preds = self.models[model_name]['model'].predict(X_test)

            else:
                print(f"  Model {model_name} not available or not trained.")
                return None

            # --------------- Core metrics (test) ---------------
            mae = mean_absolute_error(y_true, preds)
            rmse = np.sqrt(mean_squared_error(y_true, preds))
            mape = np.mean(np.abs((y_true - preds) / y_true)) * 100
            r2 = r2_score(y_true, preds)
            dir_acc = self._directional_accuracy(y_true, preds)

            self.metrics[model_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2
            }
            self.directions[model_name] = dir_acc

            print(f"  MAE:        {mae:.4f}")
            print(f"  RMSE:       {rmse:.4f}")
            print(f"  MAPE:       {mape:.2f}%")
            print(f"  R²:         {r2:.4f}")
            print(f"  Dir. Acc.:  {dir_acc:.2f}% (directional accuracy)")

            # --------------- Regime metrics (optional) ---------------
            if self.has_anomaly_flag and model_name in ['linear_regression', 'random_forest',
                                                        'xgboost', 'lightgbm', 'voting_ensemble']:
                fe_all = self.create_lag_features(pd.concat([train_data, test_data], ignore_index=True))
                fe_test = fe_all[fe_all['date'].isin(test_data['date'])].copy()
                fe_test = fe_test.dropna(subset=feature_cols + ['fx_usd_kes'])

                # Merge anomaly flag back from original df
                fe_test = fe_test.merge(
                    self.df[['date', 'is_fx_anomaly']],
                    on='date',
                    how='left'
                )

                normal = fe_test[fe_test['is_fx_anomaly'] != True]
                anomalous = fe_test[fe_test['is_fx_anomaly'] == True]

                if len(normal) > 0:
                    y_n = normal['fx_usd_kes'].values
                    p_n = self.models[model_name]['model'].predict(normal[feature_cols])
                    mape_n = np.mean(np.abs((y_n - p_n) / y_n)) * 100
                    print(f"  MAPE (normal days):   {mape_n:.2f}%")

                if len(anomalous) > 0:
                    y_a = anomalous['fx_usd_kes'].values
                    p_a = self.models[model_name]['model'].predict(anomalous[feature_cols])
                    mape_a = np.mean(np.abs((y_a - p_a) / y_a)) * 100
                    print(f"  MAPE (anomaly days): {mape_a:.2f}%")

            return {
                'predictions': preds,
                'actual': y_true,
                'dates': test_data['date'][:len(preds)]
            }

        except Exception as e:
            print(f"  ⚠️ Evaluation failed: {str(e)[:160]}")
            return None

    # =======================================================================
    # FULL PIPELINE
    # =======================================================================

    def run_full_pipeline(self, test_days=90):
        print("\n" + "=" * 70)
        print("RUNNING FULL 7-MODEL PIPELINE (Improved)")
        print("=" * 70)

        train, test = self.train_test_split(test_days=test_days)

        # Train
        self.train_arima(train)
        self.train_prophet(train)
        self.train_linear_regression(train)
        self.train_random_forest(train)
        self.train_xgboost(train)
        self.train_lightgbm(train)
        self.train_voting_ensemble(train)

        # Evaluate
        results = {}
        model_list = ['arima', 'prophet',
                      'linear_regression', 'random_forest',
                      'xgboost', 'lightgbm', 'voting_ensemble']

        for name in model_list:
            res = self.evaluate_model(name, train, test)
            if res:
                results[name] = res

        # Summaries
        if self.metrics:
            print("\n" + "=" * 70)
            print("MODEL PERFORMANCE COMPARISON (TEST SET)")
            print("=" * 70)

            comparison = pd.DataFrame(self.metrics).T
            comparison['Dir_Acc'] = pd.Series(self.directions)
            comparison = comparison.sort_values('MAPE')
            print(comparison.to_string(float_format=lambda x: f"{x:0.4f}"))

            comparison.to_csv('model_comparison_test.csv')
            print("\n✓ Test-set model comparison saved to 'model_comparison_test.csv'")

            # Train metrics table (for overfitting inspection)
            if self.train_metrics:
                print("\n" + "=" * 70)
                print("TRAIN METRICS (for ML models)")
                print("=" * 70)
                train_df = pd.DataFrame(self.train_metrics).T
                print(train_df.to_string(float_format=lambda x: f"{x:0.4f}"))
                train_df.to_csv('model_comparison_train.csv')
                print("\n✓ Train-set metrics saved to 'model_comparison_train.csv'")

            best_model = comparison.index[0]
            print(f"\n🏆 BEST MODEL (by test MAPE): {best_model.upper()}")
            print(f"   Lowest MAPE: {comparison.loc[best_model, 'MAPE']:.2f}%")
            print(f"   MAE:         {comparison.loc[best_model, 'MAE']:.4f}")
            print(f"   R²:          {comparison.loc[best_model, 'R2']:.4f}")
            print(f"   Dir. Acc.:   {comparison.loc[best_model, 'Dir_Acc']:.2f}%")

            return results, comparison
        else:
            print("\n⚠️ No models were successfully trained/evaluated.")
            return None, None


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SmartRemit AI - Comprehensive FX Forecasting System (v2)")
    print("=" * 70)

    forecaster = ComprehensiveFXForecaster('cbk_usd_kes_history.csv')
    results, comparison = forecaster.run_full_pipeline(test_days=90)

    print("\n" + "=" * 70)
    print("✅ FORECASTING COMPLETE (Improved Evaluation)")
    print("=" * 70)
    print("\nOutputs:")
    print("  - model_comparison_test.csv (test performance metrics)")
    print("  - model_comparison_train.csv (train metrics for overfitting check)")
    print("\n7 Models evaluated:")
    print("  1. ARIMA (Statistical)")
    print("  2. Prophet (Facebook)")
    print("  3. Linear Regression (Baseline)")
    print("  4. Random Forest (Ensemble)")
    print("  5. XGBoost (Gradient Boosting)")
    print("  6. LightGBM (Fast Boosting)")
    print("  7. Voting Ensemble (Combined)")
