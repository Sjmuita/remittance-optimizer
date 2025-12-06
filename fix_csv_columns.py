"""
Quick Fix Script - Run this first!
Creates provider_comparison_enhanced.csv with all required columns
"""

import pandas as pd

print("SmartRemit AI - CSV Fix Script")
print("=" * 50)

# Load existing data
try:
    df = pd.read_csv('provider_comparison.csv')
    print("✓ Loaded provider_comparison.csv")
except:
    print("✗ Could not find provider_comparison.csv")
    print("Creating sample data...")
    df = pd.DataFrame({
        'Provider': ['WorldRemit', 'Wise', 'Remitly', 'MoneyGram', 'Paypal', 'Western Union'],
        'Base Fee (USD)': [2.99, 5.21, 3.99, 4.99, 4.99, 8.00],
        'FX Margin (%)': [1.5, 0.5, 1.8, 2.0, 3.0, 3.5],
        'FX Cost (USD)': [2.96, 0.97, 3.53, 3.90, 5.85, 6.82],
        'Total Cost (USD)': [5.95, 6.18, 7.52, 8.89, 10.84, 14.82],
        'Total Cost (%)': [2.97, 3.09, 3.76, 4.45, 5.42, 7.41],
        'Amount Received (KES)': [25080, 25049, 24876, 24699, 24447, 23876],
        'Transfer Speed': ['Instant', '1-2 hours', 'Express', 'Minutes', 'Instant', 'Minutes']
    })

# Add missing columns
service_type_map = {
    'WorldRemit': 'Mobile Money',
    'Wise': 'Bank Transfer',
    'Remitly': 'Cash Pickup',
    'MoneyGram': 'Cash Pickup',
    'Paypal': 'Bank Transfer',
    'Western Union': 'Cash Pickup',
    'Sendwave': 'Mobile Money',
    'Xoom': 'Bank Transfer'
}

delivery_days_map = {
    'Instant': 0.01,
    '1-2 hours': 0.08,
    'Express': 0.5,
    'Minutes': 0.02,
    '1-3 days': 2.0
}

# Add Service Type
if 'Service Type' not in df.columns:
    df['Service Type'] = df['Provider'].map(service_type_map)
    df['Service Type'] = df['Service Type'].fillna('Bank Transfer')
    print("✓ Added Service Type column")

# Add Corridor
if 'Corridor' not in df.columns:
    df['Corridor'] = 'USA → Kenya'
    print("✓ Added Corridor column")

# Add Amount Sent
if 'Amount Sent (USD)' not in df.columns:
    df['Amount Sent (USD)'] = 200
    print("✓ Added Amount Sent column")

# Add Exchange Rate
if 'Exchange Rate' not in df.columns:
    df['Exchange Rate'] = 129.0
    print("✓ Added Exchange Rate column")

# Add Delivery Speed Days
if 'Delivery Speed Days' not in df.columns:
    df['Delivery Speed Days'] = df['Transfer Speed'].map(delivery_days_map)
    df['Delivery Speed Days'] = df['Delivery Speed Days'].fillna(1.0)
    print("✓ Added Delivery Speed Days column")

# Reorder columns
column_order = [
    'Provider', 'Service Type', 'Corridor', 'Amount Sent (USD)',
    'Base Fee (USD)', 'FX Margin (%)', 'Exchange Rate', 'FX Cost (USD)',
    'Total Cost (USD)', 'Total Cost (%)', 'Amount Received (KES)',
    'Transfer Speed', 'Delivery Speed Days'
]

existing_cols = [col for col in column_order if col in df.columns]
df = df[existing_cols]

# Save
df.to_csv('provider_comparison_enhanced.csv', index=False)

print("\n" + "=" * 50)
print("✓ SUCCESS! Created provider_comparison_enhanced.csv")
print(f"✓ {len(df)} providers with {len(df.columns)} columns")
print("\nColumns included:")
for col in df.columns:
    print(f"  - {col}")

print("\n" + "=" * 50)
print("NEXT STEP:")
print("Run: streamlit run streamlit_dashboard_final.py")
print("=" * 50)
