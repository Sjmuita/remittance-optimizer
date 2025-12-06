"""
anomaly_detection.py
SmartRemit AI - FX Rate Anomaly Detection

Detects unusual FX rate movements using:
- Statistical methods (Z-score, IQR)
- Rolling window analysis
- Isolation Forest (ML)

Author: Muita Shalyn J. - USIU-A
Date: November 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from scipy import stats

class FXAnomalyDetector:
    """
    Detect anomalies in FX rate time series
    """
    
    def __init__(self, fx_data_path='cbk_usd_kes_history.csv'):
        """Initialize detector with historical FX data"""
        print("="*70)
        print("FX ANOMALY DETECTION SYSTEM")
        print("="*70)
        
        self.df = pd.read_csv(fx_data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Ensure fx_usd_kes is numeric
        self.df['fx_usd_kes'] = pd.to_numeric(self.df['fx_usd_kes'], errors='coerce')
        self.df = self.df.dropna(subset=['fx_usd_kes'])
        
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        print(f"✓ Loaded {len(self.df)} historical rates")
        print(f"  Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"  Latest rate: {self.df['fx_usd_kes'].iloc[-1]:.4f} KES/USD")
        
        self.anomalies = pd.DataFrame()
        
    def zscore_detection(self, threshold=3):
        """Detect anomalies using Z-score method"""
        print(f"\n🔍 Z-Score Detection (threshold={threshold})...")
        
        self.df['zscore'] = np.abs(stats.zscore(self.df['fx_usd_kes']))
        self.df['zscore_anomaly'] = self.df['zscore'] > threshold
        
        n_anomalies = self.df['zscore_anomaly'].sum()
        print(f"✓ Found {n_anomalies} anomalies ({n_anomalies/len(self.df)*100:.2f}%)")
        
        return n_anomalies
    
    def iqr_detection(self, factor=1.5):
        """Detect anomalies using Interquartile Range (IQR)"""
        print(f"\n🔍 IQR Detection (factor={factor})...")
        
        Q1 = self.df['fx_usd_kes'].quantile(0.25)
        Q3 = self.df['fx_usd_kes'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        self.df['iqr_anomaly'] = (
            (self.df['fx_usd_kes'] < lower_bound) | 
            (self.df['fx_usd_kes'] > upper_bound)
        )
        
        n_anomalies = self.df['iqr_anomaly'].sum()
        print(f"✓ Found {n_anomalies} anomalies ({n_anomalies/len(self.df)*100:.2f}%)")
        print(f"  Bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
        
        return n_anomalies
    
    def rolling_window_detection(self, window=30, threshold=2):
        """Detect anomalies using rolling window statistics"""
        print(f"\n🔍 Rolling Window Detection (window={window}, threshold={threshold})...")
        
        self.df['rolling_mean'] = self.df['fx_usd_kes'].rolling(window, min_periods=1).mean()
        self.df['rolling_std'] = self.df['fx_usd_kes'].rolling(window, min_periods=1).std()
        
        self.df['rolling_zscore'] = np.abs(
            (self.df['fx_usd_kes'] - self.df['rolling_mean']) / (self.df['rolling_std'] + 0.001)
        )
        self.df['rolling_anomaly'] = self.df['rolling_zscore'] > threshold
        
        n_anomalies = self.df['rolling_anomaly'].sum()
        print(f"✓ Found {n_anomalies} anomalies ({n_anomalies/len(self.df)*100:.2f}%)")
        
        return n_anomalies
    
    def isolation_forest_detection(self, contamination=0.05):
        """Detect anomalies using Isolation Forest (ML)"""
        print(f"\n🔍 Isolation Forest Detection (contamination={contamination})...")
        
        # Create features
        features = self.df[['fx_usd_kes']].copy()
        features['rate_change'] = features['fx_usd_kes'].pct_change()
        features['rate_change_abs'] = features['rate_change'].abs()
        features = features.fillna(0)
        
        # Train Isolation Forest
        clf = IsolationForest(contamination=contamination, random_state=42)
        self.df['isolation_anomaly'] = clf.fit_predict(features) == -1
        
        n_anomalies = self.df['isolation_anomaly'].sum()
        print(f"✓ Found {n_anomalies} anomalies ({n_anomalies/len(self.df)*100:.2f}%)")
        
        return n_anomalies
    
    def detect_sudden_changes(self, threshold_pct=2.0):
        """Detect sudden day-to-day changes"""
        print(f"\n🔍 Sudden Change Detection (threshold={threshold_pct}%)...")
        
        self.df['daily_change_pct'] = self.df['fx_usd_kes'].pct_change() * 100
        self.df['sudden_change'] = self.df['daily_change_pct'].abs() > threshold_pct
        
        n_anomalies = self.df['sudden_change'].sum()
        print(f"✓ Found {n_anomalies} sudden changes ({n_anomalies/len(self.df)*100:.2f}%)")
        
        return n_anomalies
    
    def run_all_methods(self):
        """Run all anomaly detection methods"""
        print("\n" + "="*70)
        print("RUNNING ALL ANOMALY DETECTION METHODS")
        print("="*70)
        
        self.zscore_detection(threshold=3)
        self.iqr_detection(factor=1.5)
        self.rolling_window_detection(window=30, threshold=2)
        self.isolation_forest_detection(contamination=0.05)
        self.detect_sudden_changes(threshold_pct=2.0)
        
        # Create consensus anomaly flag
        self.df['anomaly_consensus'] = (
            self.df['zscore_anomaly'].astype(int) +
            self.df['iqr_anomaly'].astype(int) +
            self.df['rolling_anomaly'].astype(int) +
            self.df['isolation_anomaly'].astype(int) +
            self.df['sudden_change'].astype(int)
        )
        
        self.df['is_anomaly'] = self.df['anomaly_consensus'] >= 2
        
        # Extract anomalies
        self.anomalies = self.df[self.df['is_anomaly']].copy()
        
        print("\n" + "="*70)
        print("CONSENSUS ANOMALIES (flagged by 2+ methods)")
        print("="*70)
        print(f"Total anomalies: {len(self.anomalies)}")
        
        if len(self.anomalies) > 0:
            print("\nMost recent anomalies:")
            recent = self.anomalies.tail(5)[['date', 'fx_usd_kes', 'anomaly_consensus', 'daily_change_pct']]
            print(recent.to_string(index=False))
            
            # Save to CSV
            self.anomalies.to_csv('fx_anomalies_detected.csv', index=False)
            print("\n✓ Anomalies saved to 'fx_anomalies_detected.csv'")
        
        return self.anomalies
    
    def check_recent_anomaly(self, days_back=7):
        """Check if recent data contains anomalies"""
        cutoff = self.df['date'].max() - timedelta(days=days_back)
        recent_anomalies = self.anomalies[self.anomalies['date'] >= cutoff]
        
        if len(recent_anomalies) > 0:
            print(f"\n⚠️  ALERT: {len(recent_anomalies)} anomalies in last {days_back} days!")
            print(recent_anomalies[['date', 'fx_usd_kes', 'daily_change_pct']].to_string(index=False))
            return True
        else:
            print(f"\n✓ No anomalies in last {days_back} days")
            return False

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Initialize detector
    detector = FXAnomalyDetector('cbk_usd_kes_history.csv')
    
    # Run all methods
    anomalies = detector.run_all_methods()
    
    # Check recent
    detector.check_recent_anomaly(days_back=7)
    
    print("\n" + "="*70)
    print("✅ ANOMALY DETECTION COMPLETE")
    print("="*70)
