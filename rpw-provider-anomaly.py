"""
rpw_provider_anomaly.py
SmartRemit AI - RPW Remittance Provider Anomaly Detection

Detects anomalies in provider fees, spreads, and pricing behavior using:
- Statistical methods (Z-score, IQR, rolling windows)
- Isolation Forest (ML-based outlier detection)
- Provider-specific trend analysis
- Corridor-specific comparative analysis

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

class RPWProviderAnomalyDetector:
    """
    Detect anomalies in remittance provider pricing
    """
    
    def __init__(self, rpw_data_path=None):
        """
        Initialize detector with RPW remittance pricing data
        
        Parameters:
        -----------
        rpw_data_path : str
            Path to World Bank RPW Excel file (optional)
        """
        print("="*70)
        print("RPW PROVIDER ANOMALY DETECTION SYSTEM")
        print("="*70)
        
        if rpw_data_path:
            try:
                # Try loading RPW data
                print(f"\n🔄 Loading RPW data from: {rpw_data_path}")
                self.df = pd.read_excel(rpw_data_path)
                print(f"✓ Loaded {len(self.df)} provider pricing records")
            except FileNotFoundError:
                print(f"⚠️  File not found: {rpw_data_path}")
                print("   Using sample provider data for demonstration...")
                self.df = self._create_sample_rpw_data()
            except Exception as e:
                print(f"⚠️  Could not load file: {str(e)[:100]}")
                print("   Using sample provider data for demonstration...")
                self.df = self._create_sample_rpw_data()
        else:
            print("\n⚠️  No RPW data path provided.")
            print("   Using sample provider data for demonstration...")
            self.df = self._create_sample_rpw_data()
        
        # Standardize columns
        self.df.columns = [col.lower().strip().replace(' ', '_') for col in self.df.columns]
        self._prepare_data()
        
        self.anomalies = pd.DataFrame()
        
    def _create_sample_rpw_data(self):
        """Create sample RPW-like data for testing"""
        print("\n📊 Creating sample provider data...")
        
        dates = pd.date_range('2024-01-01', periods=365, freq='D')
        providers = ['WorldRemit', 'Wise', 'MoneyGram', 'Western Union', 'Paypal', 'Remitly']
        corridors = ['USA-Kenya', 'UK-Kenya']
        amounts = [100, 200, 500]
        
        data = []
        for date in dates:
            for provider in providers:
                for corridor in corridors:
                    for amount in amounts:
                        # Base cost + variation
                        base_cost = np.random.uniform(2, 5)
                        spread = np.random.uniform(0.5, 2.5)
                        total_cost_pct = (base_cost / amount) * 100 + spread
                        
                        # Add some anomalies randomly (5% chance)
                        if np.random.random() < 0.05:
                            total_cost_pct *= np.random.uniform(2, 3)
                        
                        data.append({
                            'date': date,
                            'provider': provider,
                            'corridor': corridor,
                            'amount': amount,
                            'total_cost_pct': total_cost_pct
                        })
        
        df = pd.DataFrame(data)
        print(f"✓ Created {len(df)} sample records")
        return df
    
    def _prepare_data(self):
        """Prepare and clean provider data"""
        print("\n📊 Preparing provider data...")
        
        # Identify date column
        date_cols = [c for c in self.df.columns if 'date' in c.lower()]
        if date_cols:
            self.df['date'] = pd.to_datetime(self.df[date_cols[0]], errors='coerce')
            self.df = self.df.dropna(subset=['date'])
            print(f"  Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        
        # Identify cost column
        cost_cols = [c for c in self.df.columns if 'cost' in c.lower() or 'total' in c.lower() or 'fee' in c.lower()]
        if cost_cols:
            self.cost_col = cost_cols[0]
            self.df[self.cost_col] = pd.to_numeric(self.df[self.cost_col], errors='coerce')
            self.df = self.df.dropna(subset=[self.cost_col])
            print(f"  Cost column: {self.cost_col}")
            print(f"  Valid records: {len(self.df)}")
        else:
            self.cost_col = 'total_cost_pct'
        
        # Identify provider column
        provider_cols = [c for c in self.df.columns if 'provider' in c.lower() or 'service' in c.lower()]
        self.provider_col = provider_cols[0] if provider_cols else 'provider'
        
        # Identify corridor column
        corridor_cols = [c for c in self.df.columns if 'corridor' in c.lower() or 'country' in c.lower()]
        self.corridor_col = corridor_cols[0] if corridor_cols else 'corridor'
        
        if self.provider_col in self.df.columns:
            print(f"  Providers: {self.df[self.provider_col].nunique()}")
        if self.corridor_col in self.df.columns:
            print(f"  Corridors: {self.df[self.corridor_col].nunique()}")
    
    def zscore_detection(self, threshold=2.5):
        """Detect provider costs that are statistical outliers"""
        print(f"\n🔍 Z-Score Detection (threshold={threshold})...")
        
        self.df['zscore'] = np.abs(stats.zscore(self.df[self.cost_col]))
        self.df['zscore_anomaly'] = self.df['zscore'] > threshold
        
        n_anomalies = self.df['zscore_anomaly'].sum()
        print(f"✓ Found {n_anomalies} anomalies ({n_anomalies/len(self.df)*100:.2f}%)")
        
        return n_anomalies
    
    def iqr_detection(self, factor=1.5):
        """Detect provider costs using Interquartile Range"""
        print(f"\n🔍 IQR Detection (factor={factor})...")
        
        Q1 = self.df[self.cost_col].quantile(0.25)
        Q3 = self.df[self.cost_col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        
        self.df['iqr_anomaly'] = (
            (self.df[self.cost_col] < lower) | 
            (self.df[self.cost_col] > upper)
        )
        
        n_anomalies = self.df['iqr_anomaly'].sum()
        print(f"✓ Found {n_anomalies} anomalies ({n_anomalies/len(self.df)*100:.2f}%)")
        
        return n_anomalies
    
    def isolation_forest_detection(self, contamination=0.05):
        """Detect anomalies using Isolation Forest"""
        print(f"\n🔍 Isolation Forest Detection...")
        
        features = self.df[[self.cost_col]].copy()
        features['cost_change'] = features[self.cost_col].pct_change().fillna(0)
        features = features.fillna(0)
        
        clf = IsolationForest(contamination=contamination, random_state=42)
        self.df['isolation_anomaly'] = clf.fit_predict(features) == -1
        
        n_anomalies = self.df['isolation_anomaly'].sum()
        print(f"✓ Found {n_anomalies} anomalies ({n_anomalies/len(self.df)*100:.2f}%)")
        
        return n_anomalies
    
    def run_all_methods(self):
        """Run all anomaly detection methods"""
        print("\n" + "="*70)
        print("RUNNING ALL PROVIDER ANOMALY DETECTION METHODS")
        print("="*70)
        
        self.zscore_detection(threshold=2.5)
        self.iqr_detection(factor=1.5)
        self.isolation_forest_detection(contamination=0.05)
        
        # Create consensus
        self.df['anomaly_votes'] = (
            self.df['zscore_anomaly'].astype(int) +
            self.df['iqr_anomaly'].astype(int) +
            self.df['isolation_anomaly'].astype(int)
        )
        self.df['is_provider_anomaly'] = self.df['anomaly_votes'] >= 2
        
        self.anomalies = self.df[self.df['is_provider_anomaly']].copy()
        
        print("\n" + "="*70)
        print("CONSENSUS ANOMALIES (2+ methods agree)")
        print("="*70)
        print(f"Total anomalies: {len(self.anomalies)}")
        
        if len(self.anomalies) > 0:
            print("\n📌 Most recent anomalies:")
            display_cols = ['date', self.provider_col, self.cost_col, 'anomaly_votes']
            display_cols = [c for c in display_cols if c in self.anomalies.columns]
            
            recent = self.anomalies.tail(10)[display_cols]
            print(recent.to_string(index=False))
            
            # Save
            self.anomalies.to_csv('rpw_provider_anomalies_detected.csv', index=False)
            print("\n✓ Anomalies saved to 'rpw_provider_anomalies_detected.csv'")
        
        return self.anomalies
    
    def flag_worst_providers(self, top_n=5):
        """Identify providers with highest anomaly rates"""
        print(f"\n🚩 Flagging Worst Providers (top {top_n})...")
        
        if len(self.anomalies) > 0 and self.provider_col in self.anomalies.columns:
            provider_counts = self.anomalies[self.provider_col].value_counts().head(top_n)
            
            print("\nProviders with most anomalies:")
            for provider, count in provider_counts.items():
                total = len(self.df[self.df[self.provider_col] == provider])
                rate = (count / total) * 100 if total > 0 else 0
                print(f"  ⚠️  {provider}: {count} anomalies ({rate:.1f}% of offers)")
            
            return provider_counts
        else:
            print("  No anomalies found")
            return None

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Initialize detector (uses sample data if no file provided)
    detector = RPWProviderAnomalyDetector(rpw_data_path=None)
    
    # Run all detections
    anomalies = detector.run_all_methods()
    
    # Flag worst providers
    detector.flag_worst_providers(top_n=5)
    
    print("\n" + "="*70)
    print("✅ PROVIDER ANOMALY DETECTION COMPLETE")
    print("="*70)
