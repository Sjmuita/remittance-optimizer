"""
data_loader.py
==============
SmartRemit AI - Production Data Loader for World Bank RPW Dataset (with Date Fix)

Based on World Bank Remittance Prices Worldwide (RPW) Database
Dataset covers 2011-2025 Q1, 367 corridors, 48 sending & 105 receiving countries

Author: Muita Shalyn J. (USIU-A)
Date: November 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class RemittanceDataLoader:
    """
    Production-grade loader for World Bank RPW Excel datasets

    Handles:
    - Multiple sheets (pre-2016 and post-2016 data)
    - Column name variations
    - USA→Kenya corridor filtering
    - Cost component extraction (cc1, cc2)
    - FX rate and margin calculations
    - Date parsing from RPW period format
    """

    def __init__(self):
        """Initialize loader"""
        print("✅ RemittanceDataLoader initialized")
        print("   Dataset: World Bank Remittance Prices Worldwide (RPW)")
        print("   Coverage: 2011-2025 Q1")

    def load_data(self, filepath, combine_sheets=True):
        """
        Load and process RPW data from Excel file

        Parameters:
        -----------
        filepath : str
            Path to RPW Excel file
        combine_sheets : bool
            If True, combines pre-2016 and post-2016 sheets

        Returns:
        --------
        pd.DataFrame : Processed USA→Kenya remittance data
        """
        print("\n" + "="*70)
        print("LOADING WORLD BANK RPW DATA")
        print("="*70 + "\n")

        print(f"📂 File: {filepath}")

        # Load Excel file and inspect sheets
        xls = pd.ExcelFile(filepath)
        print(f"\n📋 Available sheets: {xls.sheet_names}")

        # Identify data sheets (exclude metadata sheets)
        data_sheets = [s for s in xls.sheet_names 
                      if 'Dataset' in s or 'Data' in s]

        if not data_sheets:
            raise ValueError("No data sheets found. Available: " + str(xls.sheet_names))

        print(f"\n📊 Data sheets identified: {data_sheets}")

        # Load sheets
        dfs = []
        for sheet in data_sheets:
            print(f"\n   Loading: {sheet}")
            df = pd.read_excel(filepath, sheet_name=sheet)
            print(f"   ✅ {len(df):,} records loaded")
            dfs.append(df)

        # Combine if multiple sheets
        if combine_sheets and len(dfs) > 1:
            print(f"\n🔗 Combining {len(dfs)} sheets...")
            df = pd.concat(dfs, ignore_index=True)
            print(f"   ✅ Total records: {len(df):,}")
        else:
            df = dfs[0]

        # Display actual columns
        print(f"\n📋 Columns found ({len(df.columns)}): ")
        for i, col in enumerate(df.columns[:15], 1):  # Show first 15
            print(f"   {i:2d}. {col}")
        if len(df.columns) > 15:
            print(f"   ... and {len(df.columns) - 15} more")

        # Filter USA → Kenya corridor
        print(f"\n🔍 Filtering USA → Kenya corridor...")
        df_filtered = self._filter_usa_kenya(df)

        if len(df_filtered) == 0:
            raise ValueError("No USA→Kenya records found. Check source/destination values.")

        print(f"   ✅ Found {len(df_filtered):,} USA→Kenya transactions")

        # Process and clean
        print(f"\n🧮 Processing cost components...")
        df_processed = self._process_cost_components(df_filtered)

        print(f"\n🧹 Cleaning data...")
        df_clean = self._clean_data(df_processed)

        # Summary
        self._print_summary(df_clean)

        return df_clean

    def _filter_usa_kenya(self, df):
        """Filter for USA → Kenya corridor"""
        source_cols = ['source', 'source_code', 'source_name', 'sending']
        dest_cols = ['destination', 'destination_code', 'destination_name', 'receiving']

        source_col = None
        dest_col = None

        for col in source_cols:
            if col in df.columns:
                source_col = col
                break

        for col in dest_cols:
            if col in df.columns:
                dest_col = col
                break

        if not source_col or not dest_col:
            raise ValueError(f"Could not find source/destination columns")

        print(f"   Using columns: '{source_col}' → '{dest_col}'")

        usa_mask = df[source_col].astype(str).str.upper().str.contains('USA|US|UNITED STATES', na=False)
        ken_mask = df[dest_col].astype(str).str.upper().str.contains('KEN|KENYA', na=False)

        df_filtered = df[usa_mask & ken_mask].copy()
        df_filtered['corridor'] = 'USA→KEN'

        return df_filtered

    def _process_cost_components(self, df):
        """Process RPW cost components with date parsing fix"""

        # Standardize column names
        df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]

        # Extract cc1 (USD 200) components
        cc1_mapping = {
            'send_amount': 'cc1_denomination_amount',
            'total_cost_pct': 'cc1_total_cost_%',
            'fee_lcu': 'cc1_lcu_fee',
            'fx_rate': 'cc1_lcu_fx_rate',
            'fx_margin_pct': 'cc1_fx_margin',
        }

        for new_name, old_name in cc1_mapping.items():
            if old_name in df.columns:
                df[new_name] = df[old_name]

        # Use cc1 as primary
        if 'send_amount' not in df.columns and 'cc1_denomination_amount' in df.columns:
            df['send_amount'] = df['cc1_denomination_amount']

        if 'total_cost_pct' not in df.columns and 'cc1_total_cost_%' in df.columns:
            df['total_cost_pct'] = df['cc1_total_cost_%']

        if 'fx_rate' not in df.columns and 'cc1_lcu_fx_rate' in df.columns:
            df['fx_rate'] = df['cc1_lcu_fx_rate']

        if 'fx_margin_pct' not in df.columns and 'cc1_fx_margin' in df.columns:
            df['fx_margin_pct'] = df['cc1_fx_margin']

        # Calculate derived fields
        if 'send_amount' in df.columns and 'total_cost_pct' in df.columns:
            df['total_cost_usd'] = (df['send_amount'] * df['total_cost_pct'] / 100).round(2)

        if 'send_amount' in df.columns and 'fx_margin_pct' in df.columns:
            df['fx_margin_usd'] = (df['send_amount'] * df['fx_margin_pct'] / 100).round(2)

        if 'total_cost_usd' in df.columns and 'fx_margin_usd' in df.columns:
            df['explicit_fee_usd'] = (df['total_cost_usd'] - df['fx_margin_usd']).clip(lower=0).round(2)

        # Provider info
        if 'firm' in df.columns:
            df['provider'] = df['firm']

        # Service type
        if 'payment_instrument' in df.columns:
            df['service_type'] = df['payment_instrument']

        # ===== DATE PARSING FIX =====
        def parse_rpw_period(period_str):
            """Parse RPW period format: YYYY_NQ where N is quarter number"""
            try:
                if pd.isna(period_str):
                    return pd.NaT
                parts = str(period_str).split('_')
                if len(parts) == 2:
                    year = int(parts[0])
                    quarter_str = parts[1].replace('Q', '')
                    quarter = int(quarter_str)
                    # Convert quarter to first month of that quarter
                    month = (quarter - 1) * 3 + 1
                    return pd.Timestamp(year=year, month=month, day=1)
            except:
                pass
            return pd.NaT

        if 'period' in df.columns:
            df['date'] = df['period'].apply(parse_rpw_period)
            print(f"   ✅ Parsed dates from 'period' column")
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            print(f"   ✅ Parsed dates from 'date' column")
        # ============================

        # Speed
        if 'speed_actual' in df.columns:
            df['delivery_speed'] = df['speed_actual']

        return df

    def _clean_data(self, df):
        """Clean and validate data"""
        initial = len(df)

        required = ['provider', 'send_amount', 'total_cost_pct']
        available_required = [col for col in required if col in df.columns]

        if available_required:
            df = df.dropna(subset=available_required)

        if 'total_cost_pct' in df.columns:
            df = df[df['total_cost_pct'] > 0]
            df = df[df['total_cost_pct'] < 50]

        if 'send_amount' in df.columns:
            df = df[df['send_amount'] > 0]

        if 'date' in df.columns and not df['date'].isna().all():
            df = df.sort_values('date')

        df = df.reset_index(drop=True)

        removed = initial - len(df)
        if removed > 0:
            print(f"   ✅ Removed {removed:,} invalid records")

        return df

    def _print_summary(self, df):
        """Print dataset summary"""
        print("\n" + "="*70)
        print("✅ DATA LOADED SUCCESSFULLY")
        print("="*70)

        print(f"\n📊 Dataset Summary:")
        print(f"   Total Records: {len(df):,}")

        if 'provider' in df.columns:
            print(f"   Providers: {df['provider'].nunique()}")
            top_providers = df['provider'].value_counts().head(5)
            print(f"   Top 5: {', '.join(top_providers.index.tolist())}")

        if 'date' in df.columns and not df['date'].isna().all():
            print(f"   Date Range: {df['date'].min().date()} to {df['date'].max().date()}")

        if 'send_amount' in df.columns:
            amounts = df['send_amount'].unique()
            print(f"   Send Amounts: ${', $'.join(map(str, sorted(amounts)[:5]))}")

        if 'total_cost_pct' in df.columns:
            print(f"   Avg Cost per $100: {df['total_cost_pct'].mean():.2f}%")
            print(f"   Cost Range: {df['total_cost_pct'].min():.2f}% - {df['total_cost_pct'].max():.2f}%")

        if 'total_cost_usd' in df.columns:
            print(f"   Avg Total Cost: ${df['total_cost_usd'].mean():.2f}")

        if 'fx_rate' in df.columns:
            print(f"   Avg FX Rate: {df['fx_rate'].mean():.2f} KES/USD")

        if 'total_cost_pct' in df.columns:
            sdg_compliant = (df['total_cost_pct'] < 3).sum()
            sdg_pct = (sdg_compliant / len(df)) * 100
            print(f"   SDG 10.c Compliant (<3%): {sdg_compliant:,} ({sdg_pct:.1f}%)")

        print()

    def get_data_info(self, df):
        """Get detailed dataset info"""
        info = {
            'total_records': len(df),
            'columns': list(df.columns),
            'date_range': (df['date'].min(), df['date'].max()) if 'date' in df.columns else None,
            'providers': df['provider'].unique().tolist() if 'provider' in df.columns else [],
            'send_amounts': sorted(df['send_amount'].unique().tolist()) if 'send_amount' in df.columns else [],
            'avg_cost_pct': df['total_cost_pct'].mean() if 'total_cost_pct' in df.columns else None,
            'sdg_compliance_pct': (df['total_cost_pct'] < 3).mean() * 100 if 'total_cost_pct' in df.columns else None
        }
        return info


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING DATA LOADER (WITH DATE FIX)")
    print("="*70 + "\n")

    loader = RemittanceDataLoader()

    filepath = r"C:\Users\USER\OneDrive\Desktop\SmartRemitAI\rpw_dataset_2011_2025_q1.xlsx"

    try:
        df = loader.load_data(filepath)

        print("\n📊 Sample Data (First 10 Records):")
        cols_to_show = ['date', 'provider', 'send_amount', 'total_cost_pct', 
                       'total_cost_usd', 'fx_rate']
        available_cols = [c for c in cols_to_show if c in df.columns]
        print(df[available_cols].head(10))

        print("\n✅ Data loader test complete!")
        print("\nNext step: Feature engineering")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
