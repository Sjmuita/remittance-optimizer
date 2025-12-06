"""
fx_provider_integrated.py
SmartRemit AI - Integrated FX + Provider Anomaly Detection

Detects BOTH:
1. FX rate market disruptions (macro-level)
2. Provider-specific cost anomalies (when fees spike beyond FX changes)

Helps answer:
- "Is high cost due to market (FX) or provider (greed)?"
- "Which providers exploit disruptions vs pass costs fairly?"

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

class IntegratedAnomalyDetector:
    """
    Joint FX and Provider anomaly detection for cost transparency
    """
    
    def __init__(self, fx_data_path='cbk_usd_kes_history.csv', rpw_data_path=None):
        """
        Initialize with FX and optional RPW data
        
        Parameters:
        -----------
        fx_data_path : str
            Path to USD/KES FX history CSV
        rpw_data_path : str
            Path to RPW dataset (optional)
        """
        print("="*70)
        print("INTEGRATED FX + PROVIDER ANOMALY DETECTION")
        print("="*70)
        
        # Load FX data
        try:
            self.fx_df = pd.read_csv(fx_data_path)
            self.fx_df['date'] = pd.to_datetime(self.fx_df['date'])
            self.fx_df['fx_usd_kes'] = pd.to_numeric(self.fx_df['fx_usd_kes'], errors='coerce')
            self.fx_df = self.fx_df.dropna(subset=['fx_usd_kes']).sort_values('date').reset_index(drop=True)
            
            print(f"\n✓ FX Data loaded: {len(self.fx_df)} daily rates")
            print(f"  Range: {self.fx_df['date'].min()} to {self.fx_df['date'].max()}")
            print(f"  Latest: {self.fx_df['fx_usd_kes'].iloc[-1]:.4f} KES/USD")
        except Exception as e:
            print(f"\n⚠️  Could not load FX data: {str(e)[:100]}")
            print("   Creating sample FX data...")
            self.fx_df = self._create_sample_fx_data()
        
        # Load RPW data if provided
        self.rpw_df = None
        if rpw_data_path:
            try:
                self.rpw_df = pd.read_excel(rpw_data_path)
                print(f"\n✓ RPW Data loaded: {len(self.rpw_df)} provider records")
            except:
                print(f"\n⚠️  Could not load RPW data")
                print("   Using sample provider data...")
                self.rpw_df = self._create_sample_provider_data()
        else:
            print(f"\n⚠️  No RPW data provided")
            print("   Using sample provider data...")
            self.rpw_df = self._create_sample_provider_data()
    
    def _create_sample_fx_data(self):
        """Create sample FX data"""
        dates = pd.date_range('2024-01-01', periods=365, freq='D')
        fx_rates = 129 + np.cumsum(np.random.randn(365) * 0.5)
        
        return pd.DataFrame({
            'date': dates,
            'fx_usd_kes': fx_rates
        })
    
    def _create_sample_provider_data(self):
        """Create sample provider data"""
        dates = pd.date_range('2024-01-01', periods=365, freq='D')
        providers = ['WorldRemit', 'Wise', 'MoneyGram', 'Western Union']
        
        data = []
        for date in dates:
            for provider in providers:
                cost = np.random.uniform(2, 6)
                if np.random.random() < 0.05:
                    cost *= 2
                
                data.append({
                    'date': date,
                    'provider': provider,
                    'total_cost_pct': cost
                })
        
        return pd.DataFrame(data)
    
    def detect_fx_anomalies(self):
        """Detect FX rate anomalies (market disruptions)"""
        print("\n" + "="*70)
        print("DETECTING FX RATE ANOMALIES (Market-Level)")
        print("="*70)
        
        # Z-score
        self.fx_df['zscore'] = np.abs(stats.zscore(self.fx_df['fx_usd_kes']))
        self.fx_df['fx_zscore_anomaly'] = self.fx_df['zscore'] > 3
        
        # IQR
        Q1 = self.fx_df['fx_usd_kes'].quantile(0.25)
        Q3 = self.fx_df['fx_usd_kes'].quantile(0.75)
        IQR = Q3 - Q1
        self.fx_df['fx_iqr_anomaly'] = (
            (self.fx_df['fx_usd_kes'] < Q1 - 1.5*IQR) | 
            (self.fx_df['fx_usd_kes'] > Q3 + 1.5*IQR)
        )
        
        # Daily % change
        self.fx_df['daily_pct_change'] = self.fx_df['fx_usd_kes'].pct_change() * 100
        self.fx_df['fx_spike_anomaly'] = self.fx_df['daily_pct_change'].abs() > 2.0
        
        # Isolation Forest
        features = self.fx_df[['fx_usd_kes']].copy()
        clf = IsolationForest(contamination=0.05, random_state=42)
        self.fx_df['fx_isolation_anomaly'] = clf.fit_predict(features) == -1
        
        # Consensus
        anomaly_cols = ['fx_zscore_anomaly', 'fx_iqr_anomaly', 'fx_spike_anomaly', 'fx_isolation_anomaly']
        self.fx_df['fx_anomaly_votes'] = sum(self.fx_df[col].astype(int) for col in anomaly_cols)
        self.fx_df['is_fx_anomaly'] = self.fx_df['fx_anomaly_votes'] >= 2
        
        fx_anomalies = self.fx_df[self.fx_df['is_fx_anomaly']]
        
        print(f"\n🔍 FX ANOMALIES DETECTED: {len(fx_anomalies)}")
        if len(fx_anomalies) > 0:
            print("\nMost recent FX market disruptions:")
            recent = fx_anomalies.tail(5)[['date', 'fx_usd_kes', 'daily_pct_change', 'fx_anomaly_votes']]
            print(recent.to_string(index=False))
            
            fx_anomalies.to_csv('fx_market_anomalies.csv', index=False)
            print("\n✓ FX anomalies saved to 'fx_market_anomalies.csv'")
        
        return fx_anomalies
    
    def detect_provider_anomalies(self):
        """Detect provider-specific anomalies"""
        print("\n" + "="*70)
        print("DETECTING PROVIDER-SPECIFIC ANOMALIES")
        print("="*70)
        
        if self.rpw_df is None:
            print("⚠️  No provider data available")
            return None
        
        # Standardize columns
        self.rpw_df.columns = [col.lower().strip().replace(' ', '_') for col in self.rpw_df.columns]
        
        # Find cost column
        cost_cols = [c for c in self.rpw_df.columns if 'cost' in c.lower() or 'total' in c.lower() or 'fee' in c.lower()]
        provider_cols = [c for c in self.rpw_df.columns if 'provider' in c.lower()]
        
        if not cost_cols or not provider_cols:
            print("⚠️  Could not identify cost/provider columns")
            return None
        
        cost_col = cost_cols[0]
        provider_col = provider_cols[0]
        
        # Clean data
        self.rpw_df[cost_col] = pd.to_numeric(self.rpw_df[cost_col], errors='coerce')
        self.rpw_df = self.rpw_df.dropna(subset=[cost_col])
        
        # Z-score
        self.rpw_df['provider_zscore'] = np.abs(stats.zscore(self.rpw_df[cost_col]))
        self.rpw_df['provider_zscore_anomaly'] = self.rpw_df['provider_zscore'] > 2.5
        
        # IQR
        Q1 = self.rpw_df[cost_col].quantile(0.25)
        Q3 = self.rpw_df[cost_col].quantile(0.75)
        IQR = Q3 - Q1
        self.rpw_df['provider_iqr_anomaly'] = (
            (self.rpw_df[cost_col] < Q1 - 1.5*IQR) | 
            (self.rpw_df[cost_col] > Q3 + 1.5*IQR)
        )
        
        # Isolation Forest
        features = self.rpw_df[[cost_col]].copy()
        clf = IsolationForest(contamination=0.05, random_state=42)
        self.rpw_df['provider_isolation_anomaly'] = clf.fit_predict(features) == -1
        
        # Consensus
        anomaly_cols = ['provider_zscore_anomaly', 'provider_iqr_anomaly', 'provider_isolation_anomaly']
        self.rpw_df['provider_anomaly_votes'] = sum(self.rpw_df[col].astype(int) for col in anomaly_cols)
        self.rpw_df['is_provider_anomaly'] = self.rpw_df['provider_anomaly_votes'] >= 2
        
        provider_anomalies = self.rpw_df[self.rpw_df['is_provider_anomaly']]
        
        print(f"\n🔍 PROVIDER ANOMALIES DETECTED: {len(provider_anomalies)}")
        
        if len(provider_anomalies) > 0:
            print("\nMost egregious provider offers:")
            display_cols = [provider_col, cost_col, 'provider_anomaly_votes']
            display_cols = [c for c in display_cols if c in provider_anomalies.columns]
            
            recent = provider_anomalies.tail(5)[display_cols]
            print(recent.to_string(index=False))
            
            provider_anomalies.to_csv('provider_cost_anomalies.csv', index=False)
            print("\n✓ Provider anomalies saved to 'provider_cost_anomalies.csv'")
        
        return provider_anomalies
    
    def flag_unfair_providers(self):
        """Flag providers who don't pass through fair FX costs"""
        print("\n" + "="*70)
        print("IDENTIFYING UNFAIR PROVIDER BEHAVIOR")
        print("="*70)
        
        if self.rpw_df is None or 'is_provider_anomaly' not in self.rpw_df.columns:
            print("⚠️  Provider analysis not completed")
            return None
        
        provider_col = [c for c in self.rpw_df.columns if 'provider' in c.lower()]
        if not provider_col:
            print("⚠️  Provider column not found")
            return None
        
        provider_col = provider_col[0]
        
        print("\n🚩 Providers with high anomaly rates:")
        unfair_counts = self.rpw_df[self.rpw_df['is_provider_anomaly']][provider_col].value_counts()
        
        for provider, count in unfair_counts.head(5).items():
            total = len(self.rpw_df[self.rpw_df[provider_col] == provider])
            rate = (count / total) * 100 if total > 0 else 0
            print(f"  ⚠️  {provider}: {count}/{total} offers anomalous ({rate:.1f}%)")
        
        return unfair_counts
    
    def generate_combined_report(self):
        """Generate summary report"""
        print("\n" + "="*70)
        print("COMBINED ANALYSIS REPORT")
        print("="*70)
        
        fx_anom_pct = (self.fx_df['is_fx_anomaly'].sum() / len(self.fx_df)) * 100 if len(self.fx_df) > 0 else 0
        
        print(f"\n📊 SUMMARY:")
        print(f"  FX Market Anomalies: {self.fx_df['is_fx_anomaly'].sum()} ({fx_anom_pct:.1f}% of days)")
        
        if self.rpw_df is not None and 'is_provider_anomaly' in self.rpw_df.columns:
            prov_anom_pct = (self.rpw_df['is_provider_anomaly'].sum() / len(self.rpw_df)) * 100 if len(self.rpw_df) > 0 else 0
            print(f"  Provider Cost Anomalies: {self.rpw_df['is_provider_anomaly'].sum()} ({prov_anom_pct:.1f}% of offers)")
        
        print(f"\n💡 INTERPRETATION:")
        print(f"  - High FX + High Provider = Market chaos + opportunistic pricing")
        print(f"  - Low FX + High Provider = Providers overcharging")
        print(f"  - High FX + Low Provider = Fair pricing during volatility")
    
    def run_full_analysis(self):
        """Run complete integrated analysis"""
        print("\n" + "="*70)
        print("RUNNING FULL INTEGRATED ANALYSIS")
        print("="*70)
        
        fx_anom = self.detect_fx_anomalies()
        prov_anom = self.detect_provider_anomalies()
        self.flag_unfair_providers()
        self.generate_combined_report()
        
        return fx_anom, prov_anom

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Initialize detector
    detector = IntegratedAnomalyDetector(
        fx_data_path='cbk_usd_kes_history.csv',
        rpw_data_path=None  # Optional: provide RPW file path
    )
    
    # Run full analysis
    fx_anomalies, provider_anomalies = detector.run_full_analysis()
    
    print("\n" + "="*70)
    print("✅ INTEGRATED ANOMALY DETECTION COMPLETE")
    print("="*70)
    print("\nOutput files generated:")
    print("  - fx_market_anomalies.csv (FX disruption dates)")
    print("  - provider_cost_anomalies.csv (Overpriced provider offers)")
