# %%
# IMPORT LIBRARIES
##################

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pytz
import warnings
warnings.filterwarnings('ignore')

# %%
# VISUAL SETTINGS
##################

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 12
sns.set_palette("viridis")

# %%
# LOADING CSV
##################

def load_latest_csv(data_dir='data', rename_files=False):
    """Load the most recent CSV file from the data directory"""
    files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f)) and f.endswith('.csv')
    ]

    sorted_files = sorted(files, key=os.path.getctime, reverse=True)

    for file_path in sorted_files:
        created_at = datetime.fromtimestamp(os.path.getctime(file_path))
        timestamp_str = created_at.strftime('%Y-%m-%d_%H-%M-%S')
        original_name = os.path.basename(file_path)
        new_name = f"{timestamp_str}_{original_name}"
        new_path = os.path.join(data_dir, new_name)

        if rename_files:
            if not original_name.startswith(timestamp_str):
                os.rename(file_path, new_path)
                print(f"Renamed: {original_name} → {new_name}")
                file_path = new_path
            else:
                print(f"Already renamed: {original_name}")
        else:
            print(f"[DEV] Would rename: {original_name} → {new_name}")

    return pd.read_csv(sorted_files[0])

# Load data
df_raw = load_latest_csv()

# %%
# DATA PREPROCESSING
##################

def preprocess_data(df):
    """Clean and preprocess the subscription data"""
    df = df.copy()

    # Date conversion
    date_cols = [col for col in df.columns if '(UTC)' in col]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)

    df = df.sort_values(by='Created (UTC)')

    # Column selection and renaming
    columns_to_keep = [
        'Customer Name', 'Status', 'Cancellation Reason',
        'Created (UTC)', 'Start (UTC)', 'Start Date (UTC)', 
        'Current Period Start (UTC)', 'Current Period End (UTC)', 
        'Trial Start (UTC)', 'Trial End (UTC)',
        'Canceled At (UTC)', 'Ended At (UTC)', 
        'senderShopifyCustomerId (metadata)'
    ]
    
    df = df[columns_to_keep]

    df.rename(columns={
        'Customer ID': 'customer_id',
        'Customer Name': 'customer_name',
        'Status': 'status',
        'Cancellation Reason': 'cancellation_reason',
        'Created (UTC)': 'created_utc',
        'Start (UTC)': 'start_utc',
        'Start Date (UTC)': 'start_date_utc',
        'Current Period Start (UTC)': 'current_period_start_utc',
        'Current Period End (UTC)': 'current_period_end_utc',
        'Trial Start (UTC)': 'trial_start_utc',
        'Trial End (UTC)': 'trial_end_utc',
        'Canceled At (UTC)': 'canceled_at_utc',
        'Ended At (UTC)': 'ended_at_utc',
        'senderShopifyCustomerId (metadata)': 'is_gifted_member'
    }, inplace=True)

    # Convert is_gifted_member to boolean
    df['is_gifted_member'] = df['is_gifted_member'].notna() 

    # Reference date for analysis
    reference_date = pd.Timestamp.now(tz='UTC')
    print(f"📅 Reference date (TODAY) for analysis: {reference_date.strftime('%d-%m-%Y')}")

    # Consolidate status
    df.loc[df['status'].isin(['past_due', 'incomplete_expired']), 'status'] = 'canceled'

    return df, reference_date

df, reference_date = preprocess_data(df_raw)

# %%
#df.info()
df[(df['ended_at_utc'].isna()) & (df['status'] == 'canceled')]

# %%
# HELPER FUNCTIONS
##################

def clean_membership_data(df):
    """Clean and prepare membership data for analysis"""
    # Remove very short subscriptions (likely test accounts)
    df['duration_days'] = (pd.to_datetime(df['ended_at_utc']) - pd.to_datetime(df['created_utc'])).dt.days
    
    # Keep accounts that are either:
    # 1. Longer than 1 day, OR
    # 2. Still active/trialing (even if recent)
    df_clean = df[~((df['duration_days'] < 1) & ~(df['status'].isin(['active', 'trialing'])))]
    
    # Remove duplicate signups (within 12 hours)
    df_clean = df_clean.sort_values(['customer_name', 'created_utc'], ascending=[True, False])
    df_clean['time_diff'] = df_clean.groupby('customer_name')['created_utc'].diff()
    
    # Remove duplicates but keep the most recent (due to descending sort)
    df_clean = df_clean[~((df_clean['time_diff'] < pd.Timedelta(hours=12)) & (df_clean['time_diff'].notna()))]
    df_clean = df_clean.sort_values('created_utc', ascending=True)
    
    return df_clean.drop(['duration_days', 'time_diff'], axis=1)



def calculate_real_duration(row):
    """Calculate actual subscription duration from signup to end/current"""
    # Always start from actual signup date
    start_date = row['created_utc']
    
    # End date logic based on account status
    if pd.notna(row['ended_at_utc']):
        # Account has ended (canceled, expired, etc.)
        end_date = row['ended_at_utc']
    elif row['status'] in ['canceled'] and pd.notna(row['canceled_at_utc']):
        # Edge case: canceled but no ended_at_utc
        end_date = row['canceled_at_utc']
    else:
        # Active account: use end of current period
        end_date = row['current_period_end_utc']
    
    return (end_date - start_date).days



def calculate_trial_duration(row):
    """Calculate trial duration in days"""
    if pd.notna(row['trial_start_utc']) and pd.notna(row['trial_end_utc']):
        start_date = row['trial_start_utc']
        end_date = row['trial_end_utc']
        return (end_date - start_date).days
    return 0

def calculate_period_duration(row):
    """Calculate current billing period duration"""
    if pd.notna(row['current_period_start_utc']) and pd.notna(row['current_period_end_utc']):
        return (row['current_period_end_utc'] - row['current_period_start_utc']).dt.days
    return np.nan

def calculate_unknown_period(row):
    """Calculate gap between signup and first billing period (onboarding time)"""
    if pd.notna(row['created_utc']) and pd.notna(row['current_period_start_utc']):
        return (row['current_period_start_utc'] - row['created_utc']).dt.days
    return 0



def categorize_duration(duration):
    """Categorize subscription durations into business-meaningful groups"""
    if duration <= 9:
        return "Early Cancellation (0-9 days)"
    elif 10 <= duration <= 24:
        return "10-day Trial + Refund (10-24 days)"
    elif 25 <= duration <= 34:
        return "20-day Trial + Refund (25-34 days)" 
    elif 35 <= duration <= 55:
        return "30-day Trial + Refund (35-55 days)"
    elif 360 <= duration <= 420:
        return "Annual Subscription (360-420 days)"
    elif 730 <= duration <= 770:
        return "Two Year Subscription (730-770 days)"
    else:
        return f"Other ({duration} days)"
    

def categorize_status_detailed(row):
    """Enhanced status categorization based on multiple fields"""
    status = row['status']
    has_cancellation_reason = pd.notna(row['cancellation_reason'])
    cancel_at_period_end = row.get('cancel_at_period_end', False)
    
    if status == 'canceled':
        return 'Canceled'
    elif status == 'active' and has_cancellation_reason:
        return 'Active - Pending Cancellation'
    elif status == 'active':
        return 'Active'
    elif status == 'trialing' and has_cancellation_reason:
        return 'Trial - Canceled'
    elif status == 'trialing':
        return 'Trial - Active'
    elif status == 'past_due':
        return 'Payment Issues'
    elif status == 'incomplete_expired':
        return 'Payment Failed'
    else:
        return status.title()

# %%
df[(df['ended_at_utc'].isna()) & (df['status']== 'canceled')].head(20)

# %%
def analyze_subscription_patterns(df):
    """Main analysis function that applies all calculations"""
    print("Starting subscription analysis...")
    
    # Clean the data
    df_clean = clean_membership_data(df)
    print(f"Data cleaned: {len(df)} → {len(df_clean)} records")
    
    # Calculate all duration metrics
    df_clean['real_duration'] = df_clean.apply(calculate_real_duration, axis=1)
    df_clean['trial_duration'] = df_clean.apply(calculate_trial_duration, axis=1)
    
    # Calculate period metrics
    df_clean['period_duration'] = (
        pd.to_datetime(df_clean['current_period_end_utc']) - 
        pd.to_datetime(df_clean['current_period_start_utc'])
    ).dt.days
    
    df_clean['unknown_period'] = (
        pd.to_datetime(df_clean['current_period_start_utc']) - 
        pd.to_datetime(df_clean['created_utc'])
    ).dt.days
    
    # Apply categorizations
    df_clean['duration_category'] = df_clean['real_duration'].apply(categorize_duration)
    df_clean['status_detailed'] = df_clean.apply(categorize_status_detailed, axis=1)
    
    # Add helper flags
    df_clean['only_trial'] = df_clean['period_duration'] == df_clean['trial_duration']
    df_clean['has_trial'] = df_clean['trial_duration'] > 0
    df_clean['is_pending_cancellation'] = (
        (df_clean['status'] == 'active') & 
        pd.notna(df_clean['cancellation_reason'])
    )
    
    print("Analysis complete!")
    return df_clean

# %%
# Example usage:

# Apply the analysis
analysis_df = analyze_subscription_patterns(df)

# View results
print("Duration Categories:")
print(analysis_df['duration_category'].value_counts())

print("\nDetailed Status:")
print(analysis_df['status_detailed'].value_counts())

print("\nTrial Analysis:")
print(f"Users with trials: {analysis_df['has_trial'].sum():,}")
print(f"Trial-only accounts: {analysis_df['only_trial'].sum():,}")
print(f"Pending cancellations: {analysis_df['is_pending_cancellation'].sum():,}")


# %%
analysis_df['is_gifted_member']

# %%
analysis_df.columns

# %%
date_range_for_plots = pd.date_range(
    analysis_df['created_utc'].min(), 
    reference_date, 
    freq='W'  
)

# %%
analysis_df['duration_category'].value_counts().plot(kind='bar')

# %%



j
# %%
