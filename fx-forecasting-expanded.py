"""
fx_forecasting_models_expanded.py
SmartRemit AI - Comprehensive FX Rate Forecasting

Implements 7 diverse forecasting models:
1. ARIMA - Statistical time series
2. Prophet - Facebook's forecasting tool
3. Linear Regression - Simple baseline
4. Random Forest - Decision tree ensemble
5. XGBoost - Gradient boosting
6. LightGBM - Fast gradient boosting
7. Voting Ensemble - Combines multiple models

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

# Facebook Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("⚠️  Prophet not installed. Install with: pip install prophet")
    PROPHET_AVAILABLE = False

# Machine Learning models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Gradient Boosting
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("⚠️  LightGBM not installed. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False

class ComprehensiveFXForecaster:
    """
    7-model FX forecasting system with ensemble
    """
    
    def __init__(self, fx_data_path='cbk_usd_kes_history.csv'):
        """Initialize forecaster with historical FX data"""
        print("="*70)
        print("COMPREHENSIVE FX FORECASTING SYSTEM")
        print("7 Models: ARIMA | Prophet | Linear Reg | Random Forest | XGBoost | LightGBM | Ensemble")
        print("="*70)
        
        self.df = pd.read_csv(fx_data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['fx_usd_kes'] = pd.to_numeric(self.df['fx_usd_kes'], errors='coerce')
        self.df = self.df.dropna(subset=['fx_usd_kes']).sort_values('date').reset_index(drop=True)
        
        print(f"\n✓ Loaded {len(self.df)} historical FX rates")
        print(f"  Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"  Latest rate: {self.df['fx_usd_kes'].iloc[-1]:.4f} KES/USD")
        
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
    def train_test_split(self, test_days=90):
        """Split data into train/test"""
        split_date = self.df['date'].max() - timedelta(days=test_days)
        train = self.df[self.df['date'] <= split_date].copy()
        test = self.df[self.df['date'] > split_date].copy()
        
        print(f"\n📊 Train/Test Split:")
        print(f"  Train: {len(train)} days ({train['date'].min()} to {train['date'].max()})")
        print(f"  Test:  {len(test)} days ({test['date'].min()} to {test['date'].max()})")
        
        return train, test
    
    def create_lag_features(self, df, lags=[1, 7, 14, 30]):
        """Create lagged features for ML models"""
        df = df.copy()
        
        # Lag features
        for lag in lags:
            df[f'lag_{lag}'] = df['fx_usd_kes'].shift(lag)
        
        # Rolling statistics
        df['rolling_mean_7'] = df['fx_usd_kes'].rolling(7, min_periods=1).mean()
        df['rolling_std_7'] = df['fx_usd_kes'].rolling(7, min_periods=1).std()
        df['rolling_mean_30'] = df['fx_usd_kes'].rolling(30, min_periods=1).mean()
        df['rolling_std_30'] = df['fx_usd_kes'].rolling(30, min_periods=1).std()
        
        # Time features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        return df.dropna()
    
    # =========================================================================
    # MODEL 1: ARIMA
    # =========================================================================
    
    def train_arima(self, train_data, order=(5,1,0)):
        """Train ARIMA model"""
        print("\n🔄 [1/7] Training ARIMA Model...")
        
        try:
            model = ARIMA(train_data['fx_usd_kes'], order=order)
            fitted = model.fit()
            self.models['arima'] = fitted
            print(f"✓ ARIMA{order} trained (AIC: {fitted.aic:.2f})")
            return fitted
        except Exception as e:
            print(f"⚠️  ARIMA failed: {str(e)[:80]}")
            return None
    
    # =========================================================================
    # MODEL 2: PROPHET
    # =========================================================================
    
    def train_prophet(self, train_data):
        """Train Facebook Prophet"""
        if not PROPHET_AVAILABLE:
            print("\n⚠️  [2/7] Skipping Prophet (not installed)")
            return None
            
        print("\n🔄 [2/7] Training Prophet Model...")
        
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
            print(f"⚠️  Prophet failed: {str(e)[:80]}")
            return None
    
    # =========================================================================
    # MODEL 3: LINEAR REGRESSION
    # =========================================================================
    
    def train_linear_regression(self, train_data):
        """Train Linear Regression with lag features"""
        print("\n🔄 [3/7] Training Linear Regression...")
        
        try:
            train_fe = self.create_lag_features(train_data)
            feature_cols = [c for c in train_fe.columns if c not in ['date', 'fx_usd_kes']]
            
            X_train = train_fe[feature_cols]
            y_train = train_fe['fx_usd_kes']
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            self.models['linear_regression'] = {'model': model, 'features': feature_cols}
            print(f"✓ Linear Regression trained (R²: {model.score(X_train, y_train):.4f})")
            return model
        except Exception as e:
            print(f"⚠️  Linear Regression failed: {str(e)[:80]}")
            return None
    
    # =========================================================================
    # MODEL 4: RANDOM FOREST
    # =========================================================================
    
    def train_random_forest(self, train_data):
        """Train Random Forest"""
        print("\n🔄 [4/7] Training Random Forest...")
        
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
            print(f"✓ Random Forest trained ({len(feature_cols)} features)")
            return model
        except Exception as e:
            print(f"⚠️  Random Forest failed: {str(e)[:80]}")
            return None
    
    # =========================================================================
    # MODEL 5: XGBOOST
    # =========================================================================
    
    def train_xgboost(self, train_data):
        """Train XGBoost"""
        print("\n🔄 [5/7] Training XGBoost...")
        
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
            print("✓ XGBoost trained")
            return model
        except Exception as e:
            print(f"⚠️  XGBoost failed: {str(e)[:80]}")
            return None
    
    # =========================================================================
    # MODEL 6: LIGHTGBM
    # =========================================================================
    
    def train_lightgbm(self, train_data):
        """Train LightGBM"""
        if not LIGHTGBM_AVAILABLE:
            print("\n⚠️  [6/7] Skipping LightGBM (not installed)")
            return None
            
        print("\n🔄 [6/7] Training LightGBM...")
        
        try:
            train_fe = self.create_lag_features(train_data)
            feature_cols = [c for c in train_fe.columns if c not in ['date', 'fx_usd_kes']]
            
            X_train = train_fe[feature_cols]
            y_train = train_fe['fx_usd_kes']
            
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
            model.fit(X_train, y_train)
            
            self.models['lightgbm'] = {'model': model, 'features': feature_cols}
            print("✓ LightGBM trained")
            return model
        except Exception as e:
            print(f"⚠️  LightGBM failed: {str(e)[:80]}")
            return None
    
    # =========================================================================
    # MODEL 7: VOTING ENSEMBLE
    # =========================================================================
    
    def train_voting_ensemble(self, train_data):
        """Train Voting Ensemble combining Linear Reg, RF, XGBoost"""
        print("\n🔄 [7/7] Training Voting Ensemble...")
        
        try:
            train_fe = self.create_lag_features(train_data)
            feature_cols = [c for c in train_fe.columns if c not in ['date', 'fx_usd_kes']]
            
            X_train = train_fe[feature_cols]
            y_train = train_fe['fx_usd_kes']
            
            # Create ensemble from successful models
            estimators = []
            
            if self.models.get('linear_regression'):
                estimators.append(('lr', LinearRegression()))
            
            if self.models.get('random_forest'):
                estimators.append(('rf', RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)))
            
            if self.models.get('xgboost'):
                estimators.append(('xgb', xgb.XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)))
            
            if len(estimators) >= 2:
                model = VotingRegressor(estimators=estimators)
                model.fit(X_train, y_train)
                
                self.models['voting_ensemble'] = {'model': model, 'features': feature_cols}
                print(f"✓ Voting Ensemble trained (combining {len(estimators)} models)")
                return model
            else:
                print("⚠️  Not enough models for ensemble")
                return None
                
        except Exception as e:
            print(f"⚠️  Voting Ensemble failed: {str(e)[:80]}")
            return None
    
    # =========================================================================
    # EVALUATION
    # =========================================================================
    
    def evaluate_model(self, model_name, test_data):
        """Evaluate model on test set"""
        print(f"\n📈 Evaluating {model_name.upper()}...")
        
        try:
            # Time series models (ARIMA, Prophet)
            if model_name == 'arima' and self.models.get('arima'):
                preds = self.models['arima'].forecast(steps=len(test_data))
                y_true = test_data['fx_usd_kes'].values
                
            elif model_name == 'prophet' and PROPHET_AVAILABLE and self.models.get('prophet'):
                future = pd.DataFrame({'ds': test_data['date']})
                forecast = self.models['prophet'].predict(future)
                preds = forecast['yhat'].values
                y_true = test_data['fx_usd_kes'].values
            
            # ML models (Linear Reg, RF, XGBoost, LightGBM, Ensemble)
            elif model_name in ['linear_regression', 'random_forest', 'xgboost', 'lightgbm', 'voting_ensemble'] and self.models.get(model_name):
                test_fe = self.create_lag_features(test_data)
                test_fe = test_fe[test_fe['date'].isin(test_data['date'])]
                
                feature_cols = self.models[model_name]['features']
                X_test = test_fe[feature_cols]
                
                preds = self.models[model_name]['model'].predict(X_test)
                y_true = test_fe['fx_usd_kes'].values
            else:
                print(f"  Model {model_name} not available")
                return None
            
            # Calculate metrics
            mae = mean_absolute_error(y_true[:len(preds)], preds[:len(y_true)])
            rmse = np.sqrt(mean_squared_error(y_true[:len(preds)], preds[:len(y_true)]))
            mape = np.mean(np.abs((y_true[:len(preds)] - preds[:len(y_true)]) / y_true[:len(preds)])) * 100
            r2 = r2_score(y_true[:len(preds)], preds[:len(y_true)])
            
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
            print(f"  ⚠️  Evaluation failed: {str(e)[:80]}")
            return None
    
    # =========================================================================
    # FULL PIPELINE
    # =========================================================================
    
    def run_full_pipeline(self, test_days=90):
        """Run complete 7-model forecasting pipeline"""
        print("\n" + "="*70)
        print("RUNNING FULL 7-MODEL PIPELINE")
        print("="*70)
        
        train, test = self.train_test_split(test_days)
        
        # Train all 7 models
        self.train_arima(train)
        self.train_prophet(train)
        self.train_linear_regression(train)
        self.train_random_forest(train)
        self.train_xgboost(train)
        self.train_lightgbm(train)
        self.train_voting_ensemble(train)
        
        # Evaluate all models
        results = {}
        model_list = ['arima', 'prophet', 'linear_regression', 'random_forest', 'xgboost', 'lightgbm', 'voting_ensemble']
        
        for model_name in model_list:
            result = self.evaluate_model(model_name, test)
            if result:
                results[model_name] = result
        
        # Print comparison
        if self.metrics:
            print("\n" + "="*70)
            print("MODEL PERFORMANCE COMPARISON")
            print("="*70)
            
            comparison = pd.DataFrame(self.metrics).T
            comparison = comparison.sort_values('MAPE')
            print(comparison.to_string())
            
            # Save comparison
            comparison.to_csv('model_comparison.csv')
            print("\n✓ Model comparison saved to 'model_comparison.csv'")
            
            # Best model
            best_model = comparison.index[0]
            print(f"\n🏆 BEST MODEL: {best_model.upper()}")
            print(f"   Lowest MAPE: {comparison['MAPE'].iloc[0]:.2f}%")
            print(f"   MAE: {comparison['MAE'].iloc[0]:.4f}")
            print(f"   R²: {comparison['R2'].iloc[0]:.4f}")
            
            return results, comparison
        else:
            print("\n⚠️  No models were successfully trained")
            return None, None

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SmartRemit AI - Comprehensive FX Forecasting System")
    print("="*70)
    
    # Initialize forecaster
    forecaster = ComprehensiveFXForecaster('cbk_usd_kes_history.csv')
    
    # Run full pipeline
    results, comparison = forecaster.run_full_pipeline(test_days=90)
    
    print("\n" + "="*70)
    print("✅ FORECASTING COMPLETE")
    print("="*70)
    print("\nOutputs:")
    print("  - model_comparison.csv (performance metrics)")
    print("\n7 Models evaluated:")
    print("  1. ARIMA (Statistical)")
    print("  2. Prophet (Facebook)")
    print("  3. Linear Regression (Baseline)")
    print("  4. Random Forest (Ensemble)")
    print("  5. XGBoost (Gradient Boosting)")
    print("  6. LightGBM (Fast Boosting)")
    print("  7. Voting Ensemble (Combined)")
