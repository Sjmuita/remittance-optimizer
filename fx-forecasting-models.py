"""
fx_forecasting_models.py
SmartRemit AI - FX Rate Forecasting Models

Implements multiple forecasting approaches:
- ARIMA/SARIMA (statistical baseline)
- Prophet (Facebook's time series tool)
- Random Forest (ML ensemble)
- XGBoost (gradient boosting)

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
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Facebook Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("⚠️  Prophet not installed. Run: pip install prophet")
    PROPHET_AVAILABLE = False

# ML models
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

class FXForecaster:
    """
    Multi-model FX rate forecasting system
    """
    
    def __init__(self, fx_data_path='cbk_usd_kes_history.csv'):
        """
        Initialize forecaster with historical FX data
        
        Parameters:
        -----------
        fx_data_path : str
            Path to cleaned USD/KES historical rates CSV
        """
        print("="*70)
        print("FX FORECASTING SYSTEM INITIALIZED")
        print("="*70)
        
        self.df = pd.read_csv(fx_data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Ensure fx_usd_kes is numeric
        self.df['fx_usd_kes'] = pd.to_numeric(self.df['fx_usd_kes'], errors='coerce')
        self.df = self.df.dropna(subset=['fx_usd_kes'])
        
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        print(f"✓ Loaded {len(self.df)} historical FX rates")
        print(f"  Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"  Latest rate: {self.df['fx_usd_kes'].iloc[-1]:.4f} KES/USD")
        
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
    def train_test_split(self, test_days=90):
        """Split data into train/test for backtesting"""
        split_date = self.df['date'].max() - timedelta(days=test_days)
        train = self.df[self.df['date'] <= split_date].copy()
        test = self.df[self.df['date'] > split_date].copy()
        
        print(f"\n📊 Train/Test Split:")
        print(f"  Train: {len(train)} days ({train['date'].min()} to {train['date'].max()})")
        print(f"  Test:  {len(test)} days ({test['date'].min()} to {test['date'].max()})")
        
        return train, test
    
    def create_lag_features(self, df, lags=[1, 7, 30]):
        """Create lagged features for ML models"""
        df = df.copy()
        for lag in lags:
            df[f'lag_{lag}'] = df['fx_usd_kes'].shift(lag)
        
        df['rolling_mean_7'] = df['fx_usd_kes'].rolling(7, min_periods=1).mean()
        df['rolling_std_7'] = df['fx_usd_kes'].rolling(7, min_periods=1).std()
        df['rolling_mean_30'] = df['fx_usd_kes'].rolling(30, min_periods=1).mean()
        
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        
        return df.dropna()
    
    def train_arima(self, train_data, order=(5,1,0)):
        """Train ARIMA model"""
        print("\n🔄 Training ARIMA Model...")
        
        try:
            model = ARIMA(train_data['fx_usd_kes'], order=order)
            fitted = model.fit()
            
            self.models['arima'] = fitted
            print(f"✓ ARIMA{order} trained")
            print(f"  AIC: {fitted.aic:.2f}")
            
            return fitted
        except Exception as e:
            print(f"⚠️  ARIMA training failed: {str(e)[:100]}")
            return None
    
    def train_prophet(self, train_data):
        """Train Facebook Prophet model"""
        if not PROPHET_AVAILABLE:
            print("\n⚠️  Skipping Prophet (not installed)")
            return None
            
        print("\n🔄 Training Prophet Model...")
        
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
            print(f"⚠️  Prophet training failed: {str(e)[:100]}")
            return None
    
    def train_random_forest(self, train_data):
        """Train Random Forest model with lag features"""
        print("\n🔄 Training Random Forest...")
        
        try:
            train_fe = self.create_lag_features(train_data)
            
            feature_cols = [c for c in train_fe.columns if c not in ['date', 'fx_usd_kes']]
            X_train = train_fe[feature_cols]
            y_train = train_fe['fx_usd_kes']
            
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            self.models['random_forest'] = {'model': model, 'features': feature_cols}
            print(f"✓ Random Forest trained")
            print(f"  Features: {len(feature_cols)}")
            
            return model
        except Exception as e:
            print(f"⚠️  Random Forest training failed: {str(e)[:100]}")
            return None
    
    def train_xgboost(self, train_data):
        """Train XGBoost model"""
        print("\n🔄 Training XGBoost...")
        
        try:
            train_fe = self.create_lag_features(train_data)
            
            feature_cols = [c for c in train_fe.columns if c not in ['date', 'fx_usd_kes']]
            X_train = train_fe[feature_cols]
            y_train = train_fe['fx_usd_kes']
            
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            self.models['xgboost'] = {'model': model, 'features': feature_cols}
            print(f"✓ XGBoost trained")
            
            return model
        except Exception as e:
            print(f"⚠️  XGBoost training failed: {str(e)[:100]}")
            return None
    
    def evaluate_model(self, model_name, test_data):
        """Evaluate model on test set"""
        print(f"\n📈 Evaluating {model_name.upper()}...")
        
        try:
            if model_name == 'arima' and self.models.get('arima'):
                preds = self.models['arima'].forecast(steps=len(test_data))
                
            elif model_name == 'prophet' and PROPHET_AVAILABLE and self.models.get('prophet'):
                future = pd.DataFrame({'ds': test_data['date']})
                forecast = self.models['prophet'].predict(future)
                preds = forecast['yhat'].values
                
            elif model_name in ['random_forest', 'xgboost'] and self.models.get(model_name):
                test_fe = self.create_lag_features(test_data)
                test_fe = test_fe[test_fe['date'].isin(test_data['date'])]
                
                feature_cols = self.models[model_name]['features']
                X_test = test_fe[feature_cols]
                
                preds = self.models[model_name]['model'].predict(X_test)
                test_data = test_data[test_data['date'].isin(test_fe['date'])].reset_index(drop=True)
            else:
                print(f"  Model {model_name} not available")
                return None
            
            y_true = test_data['fx_usd_kes'].values[:len(preds)]
            preds = preds[:len(y_true)]
            
            mae = mean_absolute_error(y_true, preds)
            rmse = np.sqrt(mean_squared_error(y_true, preds))
            mape = np.mean(np.abs((y_true - preds) / y_true)) * 100
            r2 = r2_score(y_true, preds)
            
            self.metrics[model_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2
            }
            
            print(f"  MAE:  {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  R²:   {r2:.4f}")
            
            return {'predictions': preds, 'actual': y_true, 'dates': test_data['date'][:len(preds)]}
        
        except Exception as e:
            print(f"  ⚠️  Evaluation failed: {str(e)[:100]}")
            return None
    
    def forecast_future(self, model_name='prophet', days=30):
        """Forecast future FX rates"""
        print(f"\n🔮 Forecasting next {days} days with {model_name.upper()}...")
        
        if model_name == 'prophet' and PROPHET_AVAILABLE and self.models.get('prophet'):
            try:
                last_date = self.df['date'].max()
                future_dates = pd.date_range(last_date + timedelta(days=1), periods=days, freq='D')
                
                future = pd.DataFrame({'ds': future_dates})
                forecast = self.models['prophet'].predict(future)
                
                results = pd.DataFrame({
                    'date': future_dates,
                    'predicted_rate': forecast['yhat'].values,
                    'lower_bound': forecast['yhat_lower'].values,
                    'upper_bound': forecast['yhat_upper'].values
                })
                
                print(f"✓ Forecast generated for {days} days")
                print(f"  First: {results['predicted_rate'].iloc[0]:.4f}")
                print(f"  Last:  {results['predicted_rate'].iloc[-1]:.4f}")
                
                return results
            except Exception as e:
                print(f"⚠️  Forecasting failed: {str(e)[:100]}")
                return None
        else:
            print(f"⚠️  {model_name} forecasting not available or model not trained")
            return None
    
    def run_full_pipeline(self, test_days=90):
        """Run complete forecasting pipeline"""
        print("\n" + "="*70)
        print("RUNNING FULL FORECASTING PIPELINE")
        print("="*70)
        
        train, test = self.train_test_split(test_days)
        
        # Train all models
        self.train_arima(train)
        self.train_prophet(train)
        self.train_random_forest(train)
        self.train_xgboost(train)
        
        # Evaluate all models
        results = {}
        for model_name in ['arima', 'prophet', 'random_forest', 'xgboost']:
            result = self.evaluate_model(model_name, test)
            if result:
                results[model_name] = result
        
        # Print comparison
        if self.metrics:
            print("\n" + "="*70)
            print("MODEL COMPARISON")
            print("="*70)
            
            comparison = pd.DataFrame(self.metrics).T
            comparison = comparison.sort_values('MAPE')
            print(comparison.to_string())
            
            best_model = comparison.index[0]
            print(f"\n🏆 Best Model: {best_model.upper()} (Lowest MAPE: {comparison['MAPE'].iloc[0]:.2f}%)")
            
            # Generate future forecast with best model
            forecast = self.forecast_future(model_name=best_model, days=30)
            
            if forecast is not None:
                forecast.to_csv('fx_forecast_30days.csv', index=False)
                print("\n✓ 30-day forecast saved to 'fx_forecast_30days.csv'")
            
            return results, comparison, forecast
        else:
            print("\n⚠️  No models were successfully trained")
            return None, None, None

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Initialize forecaster
    forecaster = FXForecaster('cbk_usd_kes_history.csv')
    
    # Run full pipeline
    results, comparison, forecast = forecaster.run_full_pipeline(test_days=90)
    
    print("\n" + "="*70)
    print("✅ FX FORECASTING COMPLETE")
    print("="*70)
