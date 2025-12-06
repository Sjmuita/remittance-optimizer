"""
feature_engineering.py
======================
SmartRemit AI - Feature Engineering for ML Models

Transforms raw RPW data into ML-ready features for:
- Cost prediction models
- Provider recommendation
- Anomaly detection
- Time series forecasting

Author: Muita Shalyn J. (USIU-A)
Date: November 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering for remittance cost prediction

    Creates features for:
    - Time-based patterns (trends, seasonality)
    - Provider characteristics (tier, market share)
    - Cost components (fees, FX margins)
    - Competitive dynamics
    """

    def __init__(self):
        """Initialize feature engineer"""
        print("✅ FeatureEngineer initialized")

    def engineer_features(self, df):
        """
        Main feature engineering pipeline

        Parameters:
        -----------
        df : pd.DataFrame
            Cleaned data from data_loader

        Returns:
        --------
        pd.DataFrame : Data with engineered features
        """
        print("\n" + "="*70)
        print("FEATURE ENGINEERING")
        print("="*70 + "\n")

        df = df.copy()

        # 1. Time-based features
        print("🕐 Creating time-based features...")
        df = self._create_time_features(df)

        # 2. Provider features
        print("🏢 Creating provider features...")
        df = self._create_provider_features(df)

        # 3. Cost decomposition features
        print("💰 Creating cost features...")
        df = self._create_cost_features(df)

        # 4. Market dynamics features
        print("📊 Creating market dynamics features...")
        df = self._create_market_features(df)

        # 5. SDG compliance features
        print("🎯 Creating SDG 10.c features...")
        df = self._create_sdg_features(df)

        # 6. Statistical features
        print("📈 Creating statistical features...")
        df = self._create_statistical_features(df)

        # Summary
        self._print_feature_summary(df)

        return df

    def _create_time_features(self, df):
        """Create time-based features"""
        if 'date' not in df.columns or df['date'].isna().all():
            print("   ⚠️ No valid dates - skipping time features")
            return df

        # Extract time components
        df['year'] = df['date'].dt.year
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['year_month'] = df['date'].dt.to_period('M')

        # Time since first transaction
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

        # Quarter indicators (for seasonality)
        df['is_q1'] = (df['quarter'] == 1).astype(int)
        df['is_q2'] = (df['quarter'] == 2).astype(int)
        df['is_q3'] = (df['quarter'] == 3).astype(int)
        df['is_q4'] = (df['quarter'] == 4).astype(int)

        # Methodology period (pre/post Q2 2016)
        df['post_2016'] = (df['date'] >= '2016-04-01').astype(int)

        print(f"   ✅ Created {9} time features")
        return df

    def _create_provider_features(self, df):
        """Create provider-specific features"""
        if 'provider' not in df.columns:
            print("   ⚠️ No provider column - skipping provider features")
            return df

        # Provider transaction count (market activity)
        provider_counts = df.groupby('provider').size()
        df['provider_txn_count'] = df['provider'].map(provider_counts)

        # Provider market share
        total_txns = len(df)
        df['provider_market_share'] = df['provider_txn_count'] / total_txns

        # Provider average cost
        provider_avg_cost = df.groupby('provider')['total_cost_pct'].mean()
        df['provider_avg_cost'] = df['provider'].map(provider_avg_cost)

        # Provider cost volatility (std dev)
        provider_std_cost = df.groupby('provider')['total_cost_pct'].std()
        df['provider_cost_volatility'] = df['provider'].map(provider_std_cost).fillna(0)

        # Provider tier (based on avg cost)
        def assign_tier(avg_cost):
            if avg_cost < 3:
                return 'Budget'
            elif avg_cost < 6:
                return 'Standard'
            else:
                return 'Premium'

        df['provider_tier'] = df['provider_avg_cost'].apply(assign_tier)

        # One-hot encode tier
        df['tier_budget'] = (df['provider_tier'] == 'Budget').astype(int)
        df['tier_standard'] = (df['provider_tier'] == 'Standard').astype(int)
        df['tier_premium'] = (df['provider_tier'] == 'Premium').astype(int)

        print(f"   ✅ Created {8} provider features")
        return df

    def _create_cost_features(self, df):
        """Create cost decomposition features"""

        # Cost per $100 (normalized)
        if 'total_cost_pct' in df.columns:
            df['cost_per_100'] = df['total_cost_pct']

        # Fee vs FX margin ratio
        if 'explicit_fee_usd' in df.columns and 'fx_margin_usd' in df.columns:
            df['fee_to_fx_ratio'] = df['explicit_fee_usd'] / (df['fx_margin_usd'] + 0.01)  # Avoid div by zero

        # Total cost in KES
        if 'total_cost_usd' in df.columns and 'fx_rate' in df.columns:
            df['total_cost_kes'] = df['total_cost_usd'] * df['fx_rate']

        # FX rate deviation from mean
        if 'fx_rate' in df.columns:
            mean_fx = df['fx_rate'].mean()
            df['fx_rate_deviation'] = df['fx_rate'] - mean_fx
            df['fx_rate_deviation_pct'] = (df['fx_rate_deviation'] / mean_fx) * 100

        # Cost category
        if 'total_cost_pct' in df.columns:
            def categorize_cost(cost_pct):
                if cost_pct < 3:
                    return 'Low'
                elif cost_pct < 5:
                    return 'Medium'
                elif cost_pct < 8:
                    return 'High'
                else:
                    return 'Very High'

            df['cost_category'] = df['total_cost_pct'].apply(categorize_cost)

            # One-hot encode
            df['cost_low'] = (df['cost_category'] == 'Low').astype(int)
            df['cost_medium'] = (df['cost_category'] == 'Medium').astype(int)
            df['cost_high'] = (df['cost_category'] == 'High').astype(int)
            df['cost_very_high'] = (df['cost_category'] == 'Very High').astype(int)

        print(f"   ✅ Created {10} cost features")
        return df

    def _create_market_features(self, df):
        """Create market dynamics features"""

        # Number of providers active in same period
        if 'year_month' in df.columns and 'provider' in df.columns:
            period_provider_count = df.groupby('year_month')['provider'].nunique()
            df['market_competition'] = df['year_month'].map(period_provider_count)

        # Period average cost
        if 'year_month' in df.columns and 'total_cost_pct' in df.columns:
            period_avg_cost = df.groupby('year_month')['total_cost_pct'].mean()
            df['market_avg_cost'] = df['year_month'].map(period_avg_cost)

            # Cost relative to market
            df['cost_vs_market'] = df['total_cost_pct'] - df['market_avg_cost']

        # Period minimum cost (cheapest option available)
        if 'year_month' in df.columns and 'total_cost_pct' in df.columns:
            period_min_cost = df.groupby('year_month')['total_cost_pct'].min()
            df['market_min_cost'] = df['year_month'].map(period_min_cost)

            # How much more expensive than cheapest
            df['cost_premium'] = df['total_cost_pct'] - df['market_min_cost']

        print(f"   ✅ Created {5} market dynamics features")
        return df

    def _create_sdg_features(self, df):
        """Create SDG 10.c compliance features"""

        if 'total_cost_pct' not in df.columns:
            print("   ⚠️ No cost data - skipping SDG features")
            return df

        # SDG 10.c target: <3%
        df['sdg_compliant'] = (df['total_cost_pct'] < 3).astype(int)
        df['distance_from_sdg'] = df['total_cost_pct'] - 3
        df['distance_from_sdg'] = df['distance_from_sdg'].clip(lower=0)

        # Period SDG compliance rate
        if 'year_month' in df.columns:
            period_sdg_rate = df.groupby('year_month')['sdg_compliant'].mean()
            df['period_sdg_compliance_rate'] = df['year_month'].map(period_sdg_rate)

        print(f"   ✅ Created {3} SDG features")
        return df

    def _create_statistical_features(self, df):
        """Create rolling statistics and lagged features"""

        if 'date' not in df.columns or df['date'].isna().all():
            print("   ⚠️ No valid dates - skipping statistical features")
            return df

        # Sort by date
        df = df.sort_values('date')

        # Rolling statistics (30-day window)
        if 'total_cost_pct' in df.columns:
            df['cost_rolling_mean_30d'] = df['total_cost_pct'].rolling(window=30, min_periods=1).mean()
            df['cost_rolling_std_30d'] = df['total_cost_pct'].rolling(window=30, min_periods=1).std().fillna(0)

        # Lagged features (previous period cost)
        if 'total_cost_pct' in df.columns:
            df['cost_lag_1'] = df['total_cost_pct'].shift(1)
            df['cost_change'] = df['total_cost_pct'] - df['cost_lag_1']
            df = df.fillna({'cost_lag_1': df['total_cost_pct'].mean(), 'cost_change': 0})

        print(f"   ✅ Created {4} statistical features")
        return df

    def _print_feature_summary(self, df):
        """Print feature engineering summary"""
        print("\n" + "="*70)
        print("✅ FEATURE ENGINEERING COMPLETE")
        print("="*70)

        print(f"\n📊 Feature Summary:")
        print(f"   Total Columns: {len(df.columns)}")

        # Count feature types
        time_features = [c for c in df.columns if any(x in c for x in ['year', 'quarter', 'month', 'days_since', 'is_q', 'post_'])]
        provider_features = [c for c in df.columns if 'provider' in c or 'tier' in c]
        cost_features = [c for c in df.columns if 'cost' in c or 'fee' in c or 'fx' in c]
        market_features = [c for c in df.columns if 'market' in c or 'competition' in c]
        sdg_features = [c for c in df.columns if 'sdg' in c]

        print(f"   Time Features: {len(time_features)}")
        print(f"   Provider Features: {len(provider_features)}")
        print(f"   Cost Features: {len(cost_features)}")
        print(f"   Market Features: {len(market_features)}")
        print(f"   SDG Features: {len(sdg_features)}")

        print(f"\n🎯 Ready for Machine Learning!")
        print()

    def get_feature_importance_ready(self, df, target='total_cost_pct'):
        """
        Prepare data for ML models

        Returns:
        --------
        X : pd.DataFrame - Features
        y : pd.Series - Target
        feature_names : list - Feature column names
        """
        # Select numeric features only
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target and ID columns
        exclude_cols = [target, 'id', 'days_since_start'] if target in numeric_features else ['id', 'days_since_start']
        feature_cols = [c for c in numeric_features if c not in exclude_cols]

        X = df[feature_cols].fillna(0)
        y = df[target] if target in df.columns else None

        return X, y, feature_cols


# Test feature engineering
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING FEATURE ENGINEERING")
    print("="*70 + "\n")

    from data_loader import RemittanceDataLoader

    # Load data
    loader = RemittanceDataLoader()
    filepath = r"C:\Users\USER\OneDrive\Desktop\SmartRemitAI\rpw_dataset_2011_2025_q1.xlsx"
    df = loader.load_data(filepath)

    # Engineer features
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df)

    print("\n📊 Sample Features (First 5 Records):")
    feature_cols = ['date', 'provider', 'total_cost_pct', 'provider_market_share', 
                   'cost_vs_market', 'sdg_compliant']
    available = [c for c in feature_cols if c in df_features.columns]
    print(df_features[available].head())

    print("\n✅ Feature engineering test complete!")
    print("\nNext step: Model training or analysis")
