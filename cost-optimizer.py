"""
cost_optimizer.py
SmartRemit AI - Provider Cost Optimization Engine

Finds the cheapest remittance provider for any scenario by:
- Comparing real-time fees across all providers
- Incorporating FX rates and margins
- Calculating true total cost (fee + spread)
- Ranking providers by best value

Author: Muita Shalyn J. - USIU-A
Date: November 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RemittanceCostOptimizer:
    """
    Find the cheapest remittance provider for any amount and corridor
    """
    
    def __init__(self, rpw_data_path=None, fx_data_path='cbk_usd_kes_history.csv'):
        """
        Initialize optimizer with provider pricing and FX data
        
        Parameters:
        -----------
        rpw_data_path : str
            Path to World Bank RPW dataset (optional)
        fx_data_path : str
            Path to USD/KES FX history
        """
        print("="*70)
        print("REMITTANCE COST OPTIMIZER")
        print("="*70)
        
        # Load FX data
        self.fx_df = pd.read_csv(fx_data_path)
        self.fx_df['date'] = pd.to_datetime(self.fx_df['date'])
        self.fx_df['fx_usd_kes'] = pd.to_numeric(self.fx_df['fx_usd_kes'], errors='coerce')
        self.fx_df = self.fx_df.dropna().sort_values('date')
        
        self.current_fx = self.fx_df['fx_usd_kes'].iloc[-1]
        print(f"\n✓ Current FX Rate: {self.current_fx:.4f} KES/USD")
        
        # Load RPW data if provided
        self.rpw_df = None
        if rpw_data_path:
            try:
                xls = pd.ExcelFile(rpw_data_path)
                df_list = []
                for sheet in xls.sheet_names:
                    if 'dataset' in sheet.lower():
                        df_list.append(pd.read_excel(rpw_data_path, sheet_name=sheet))
                
                if df_list:
                    self.rpw_df = pd.concat(df_list, ignore_index=True)
                    print(f"✓ Loaded {len(self.rpw_df)} provider pricing records")
                else:
                    self.rpw_df = pd.read_excel(rpw_data_path)
            except:
                print("⚠️  Could not load RPW data. Using sample providers.")
                self.rpw_df = self._create_sample_providers()
        else:
            print("⚠️  No RPW data provided. Using sample providers.")
            self.rpw_df = self._create_sample_providers()
        
        self._prepare_provider_data()
        
    def _create_sample_providers(self):
        """Create sample provider data if RPW not available"""
        providers = {
            'provider': ['WorldRemit', 'Wise', 'MoneyGram', 'Western Union', 'Paypal', 'Remitly'],
            'base_fee_usd': [2.99, 5.21, 4.99, 8.00, 4.99, 3.99],
            'fx_margin_pct': [1.5, 0.5, 2.0, 3.5, 3.0, 1.8],
            'transfer_speed': ['Instant', 'Hours', 'Minutes', 'Minutes', 'Instant', 'Express']
        }
        return pd.DataFrame(providers)
    
    def _prepare_provider_data(self):
        """Prepare and standardize provider data"""
        if self.rpw_df is not None:
            self.rpw_df.columns = [col.lower().strip().replace(' ', '_') for col in self.rpw_df.columns]
            
            # Identify key columns
            provider_col = [c for c in self.rpw_df.columns if 'provider' in c or 'service' in c]
            fee_col = [c for c in self.rpw_df.columns if 'fee' in c or 'cost' in c]
            
            if provider_col:
                self.provider_col = provider_col[0]
            if fee_col:
                self.fee_col = fee_col[0]
    
    def calculate_total_cost(self, amount_usd, base_fee, fx_margin_pct):
        """
        Calculate total cost including fees and FX margin
        
        Parameters:
        -----------
        amount_usd : float
            Amount to send in USD
        base_fee : float
            Provider's base fee in USD
        fx_margin_pct : float
            Provider's FX margin as percentage
        
        Returns:
        --------
        dict : Total cost breakdown
        """
        # Provider's FX rate (worse than market by margin)
        provider_fx = self.current_fx * (1 - fx_margin_pct / 100)
        
        # Amount after fee
        amount_after_fee = amount_usd - base_fee
        
        # Amount received in KES
        amount_received_kes = amount_after_fee * provider_fx
        
        # Market rate amount (what they should receive)
        market_amount_kes = amount_usd * self.current_fx
        
        # Total cost in USD
        total_cost_usd = market_amount_kes / self.current_fx - amount_received_kes / self.current_fx
        
        # Cost as percentage
        cost_pct = (total_cost_usd / amount_usd) * 100
        
        return {
            'total_cost_usd': total_cost_usd,
            'total_cost_pct': cost_pct,
            'base_fee_usd': base_fee,
            'fx_margin_cost_usd': total_cost_usd - base_fee,
            'provider_fx_rate': provider_fx,
            'amount_received_kes': amount_received_kes
        }
    
    def compare_providers(self, amount_usd=200, corridor='USA-Kenya'):
        """
        Compare all providers for a given amount and corridor
        
        Parameters:
        -----------
        amount_usd : float
            Amount to send in USD
        corridor : str
            Remittance corridor (e.g., 'USA-Kenya')
        
        Returns:
        --------
        pd.DataFrame : Ranked providers by total cost
        """
        print("\n" + "="*70)
        print(f"COMPARING PROVIDERS FOR ${amount_usd} USD → Kenya")
        print("="*70)
        
        results = []
        
        # Use sample provider data for demonstration
        providers_data = [
            {'name': 'Wise', 'fee': 5.21, 'margin': 0.5, 'speed': '1-2 hours'},
            {'name': 'WorldRemit', 'fee': 2.99, 'margin': 1.5, 'speed': 'Instant'},
            {'name': 'Remitly', 'fee': 3.99, 'margin': 1.8, 'speed': 'Express'},
            {'name': 'MoneyGram', 'fee': 4.99, 'margin': 2.0, 'speed': 'Minutes'},
            {'name': 'Paypal', 'fee': 4.99, 'margin': 3.0, 'speed': 'Instant'},
            {'name': 'Western Union', 'fee': 8.00, 'margin': 3.5, 'speed': 'Minutes'},
        ]
        
        for provider in providers_data:
            costs = self.calculate_total_cost(
                amount_usd, 
                provider['fee'], 
                provider['margin']
            )
            
            results.append({
                'Provider': provider['name'],
                'Base Fee (USD)': provider['fee'],
                'FX Margin (%)': provider['margin'],
                'FX Cost (USD)': costs['fx_margin_cost_usd'],
                'Total Cost (USD)': costs['total_cost_usd'],
                'Total Cost (%)': costs['total_cost_pct'],
                'Amount Received (KES)': costs['amount_received_kes'],
                'Transfer Speed': provider['speed']
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('Total Cost (USD)')
        
        print("\n📊 PROVIDER COMPARISON:")
        print(df.to_string(index=False))
        
        # Best deal
        best = df.iloc[0]
        print(f"\n🏆 BEST DEAL: {best['Provider']}")
        print(f"   Total Cost: ${best['Total Cost (USD)']:.2f} ({best['Total Cost (%)']:.2f}%)")
        print(f"   Recipient gets: {best['Amount Received (KES)']:.2f} KES")
        print(f"   Transfer Speed: {best['Transfer Speed']}")
        
        # Worst deal
        worst = df.iloc[-1]
        savings = worst['Total Cost (USD)'] - best['Total Cost (USD)']
        print(f"\n💰 SAVINGS vs {worst['Provider']}: ${savings:.2f}")
        
        df.to_csv('provider_comparison.csv', index=False)
        print("\n✓ Comparison saved to 'provider_comparison.csv'")
        
        return df
    
    def recommend_provider(self, amount_usd=200, priority='cost'):
        """
        Recommend best provider based on priority
        
        Parameters:
        -----------
        amount_usd : float
            Amount to send
        priority : str
            'cost' (cheapest), 'speed' (fastest), or 'balanced'
        
        Returns:
        --------
        dict : Recommended provider details
        """
        print(f"\n🎯 RECOMMENDATION (Priority: {priority.upper()}):")
        
        df = self.compare_providers(amount_usd)
        
        if priority == 'cost':
            rec = df.iloc[0]
            print(f"\n   Provider: {rec['Provider']}")
            print(f"   Why: Lowest total cost at ${rec['Total Cost (USD)']:.2f}")
            
        elif priority == 'speed':
            instant = df[df['Transfer Speed'].str.contains('Instant|Express', case=False)]
            if len(instant) > 0:
                rec = instant.iloc[0]
                print(f"\n   Provider: {rec['Provider']}")
                print(f"   Why: Fastest transfer ({rec['Transfer Speed']}) with reasonable cost")
            else:
                rec = df.iloc[0]
                print(f"\n   Provider: {rec['Provider']} (fastest available)")
        
        elif priority == 'balanced':
            # Score based on cost (70%) and speed (30%)
            df['cost_score'] = 1 - (df['Total Cost (%)'] / df['Total Cost (%)'].max())
            df['speed_score'] = df['Transfer Speed'].apply(
                lambda x: 1.0 if 'Instant' in x else (0.8 if 'Express' in x else 0.5)
            )
            df['balanced_score'] = 0.7 * df['cost_score'] + 0.3 * df['speed_score']
            
            rec = df.sort_values('balanced_score', ascending=False).iloc[0]
            print(f"\n   Provider: {rec['Provider']}")
            print(f"   Why: Best balance of cost and speed")
        
        return rec.to_dict()
    
    def savings_calculator(self, amount_usd, current_provider='Western Union'):
        """Calculate potential savings by switching providers"""
        print(f"\n💡 SAVINGS CALCULATOR:")
        print(f"   Current provider: {current_provider}")
        print(f"   Amount: ${amount_usd}")
        
        df = self.compare_providers(amount_usd)
        
        current = df[df['Provider'] == current_provider]
        best = df.iloc[0]
        
        if len(current) > 0:
            current_cost = current['Total Cost (USD)'].iloc[0]
            best_cost = best['Total Cost (USD)']
            savings = current_cost - best_cost
            
            print(f"\n   Current cost: ${current_cost:.2f}")
            print(f"   Best available: ${best_cost:.2f} ({best['Provider']})")
            print(f"   💰 You could save: ${savings:.2f} per transaction")
            
            # Annual savings
            annual = savings * 12
            print(f"   📅 Annual savings (1x/month): ${annual:.2f}")
            
            return savings
        else:
            print(f"\n   ⚠️  Provider '{current_provider}' not found in data")
            return 0

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Initialize optimizer
    optimizer = RemittanceCostOptimizer(
        rpw_data_path=None,  # Optional: provide RPW Excel file path
        fx_data_path='cbk_usd_kes_history.csv'
    )
    
    # Compare providers for $200
    comparison = optimizer.compare_providers(amount_usd=200)
    
    # Get recommendation
    recommendation = optimizer.recommend_provider(amount_usd=200, priority='cost')
    
    # Calculate savings
    savings = optimizer.savings_calculator(amount_usd=200, current_provider='Western Union')
    
    print("\n" + "="*70)
    print("✅ COST OPTIMIZATION COMPLETE")
    print("="*70)
