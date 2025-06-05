# %%


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from datetime import datetime

# %%

# Setting up the plotting style
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 10, 'axes.titlesize': 16})
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['figure.figsize'] = (22, 11)

# Grid with opacity and in background
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = 'lightgray'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['axes.axisbelow'] = True

plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlecolor'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
#plt.rcParams['legend.title_fontsize'] = 'large'
plt.rcParams['legend.labelcolor'] = 'black'
plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['legend.edgecolor'] = 'gray'
plt.rcParams['text.color'] = 'white'
sns.set_palette("viridis")

# %%

# LOADING CSV
##################
# Set TODAY DATE
reference_date = pd.Timestamp.now(tz='UTC')


# Toggle this flag to True in production
RENAME_FILES = False

data_dir = 'data'

# List and sort files by creation time

files = [
    os.path.join(data_dir, f)
    for f in os.listdir(data_dir)
    if os.path.isfile(os.path.join(data_dir, f)) and f.endswith('.csv')
]

sorted_files = sorted(files, key=os.path.getctime, reverse=True)

# Loop over files
for file_path in sorted_files:

    created_at = datetime.fromtimestamp(os.path.getctime(file_path))
    timestamp_str = created_at.strftime('%Y-%m-%d_%H-%M-%S')
    original_name = os.path.basename(file_path)
    new_name = f"{timestamp_str}_{original_name}"
    new_path = os.path.join(data_dir, new_name)

    if RENAME_FILES:
        if not original_name.startswith(timestamp_str):
            os.rename(file_path, new_path)
            print(f"Renamed: {original_name} → {new_name}")
            file_path = new_path
    #     else:
    #         print(f"Already renamed: {original_name}")
    # else:
    #     print(f"[DEV] Would rename: {original_name} → {new_name}")
    #

df_raw = pd.read_csv(file_path, low_memory=False)

# %%

# DATA PREPROCESSING
##################

def preprocess_data(input_df):
    """Clean and preprocess the subscription data"""
    df = input_df.copy()

    # Date conversion
    date_cols = [col for col in df.columns if '(UTC)' in col]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)

    df = df.sort_values(by='Created (UTC)')

    # Column selection and renaming
    columns_to_keep = [
        'Customer Name', 'Status', 'Cancellation Reason',
        'Created (UTC)', 'Start (UTC)', 'Current Period Start (UTC)', 
        'Current Period End (UTC)', 'Trial Start (UTC)', 'Trial End (UTC)',
        'Canceled At (UTC)', 'Ended At (UTC)', 'senderShopifyCustomerId (metadata)'
    ]
    
    df = df[columns_to_keep]

    df.rename(columns={
        'Customer Name': 'customer_name',
        'Status': 'status',
        'Cancellation Reason': 'cancellation_reason',
        'Created (UTC)': 'created_utc',
        'Start (UTC)': 'start_utc',
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
    print('--------------------------------------')
    print(f"📅 Reference date (TODAY) : {reference_date.strftime('%d-%m-%Y')}")
    print(f"{len(df)} entries loaded from {file_path}")
    print('--------------------------------------')

    return df


df = preprocess_data(df_raw)

# %%

# Removing customers with more than 7 subscriptions (Probably testing accounts)
def remove_high_volume_customers(df, threshold=7):
    """Remove customers with more than a specified number of subscriptions"""
    
    original_count = len(df)

    customer_counts = df['customer_name'].value_counts()
    high_volume_customers = customer_counts[customer_counts > threshold].index
    
    df = df[~df['customer_name'].isin(high_volume_customers)]
    
    print(f'{original_count - len(df)} subscriptions removed,')
    print(f'from {len(high_volume_customers)} customers with more than {threshold} subscriptions')
    print('--------------------------------------')

    return df


df = remove_high_volume_customers(df)

# %%

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


    print(f"📊 {len(df)} subscriptions before cleaning")
    print(f"📊 {len(df_clean)} subscriptions after cleaning")
    return df_clean.drop(['duration_days', 'time_diff'], axis=1)

df =  clean_membership_data(df)

# %%

# Adding end date for subscriptions that are canceled but have no ended_at_utc date.
def add_ended_at_for_canceled(df):
    """Set ended_at_utc to current_period_end_utc for canceled subscriptions without ended_at_utc"""
    
    # Ensure ended_at_utc is set to current_period_end_utc for canceled subscriptions
    mask = (df['ended_at_utc'].isna()) & (df['canceled_at_utc'].notna())
    
    df.loc[mask, 'ended_at_utc'] = df['current_period_end_utc']
    
    return df


df = add_ended_at_for_canceled(df)

# %%

# CALCULATING DURATIONS
def calculate_duration(df):
    """Calculate various duration in days"""
    
    df['trial_duration'] = \
            (df['trial_end_utc'] - df['trial_start_utc']).dt.days.fillna(0)
    
    df['current_period_duration'] = \
            (df['current_period_end_utc'] - df['current_period_start_utc']).dt.days
    
    df['trial_only_subscription'] = (
        df['trial_start_utc'].notna() & 
        df['trial_end_utc'].notna() & 
        (df['trial_duration'] == df['current_period_duration'])
    )

    df['gift_duration'] = df['current_period_duration'].where(df['is_gifted_member'], 0)

    df['end_in'] = \
            ((df['current_period_end_utc'] - reference_date).dt.days).where(df['status'] == 'active', np.nan)

    df['total_duration'] = (df['ended_at_utc'] - df['created_utc']).dt.days

    return df


df = calculate_duration(df)

# %%

# IN CHURN PERIOD
def in_churn_period(df):
    """Check if a member is in the churn period (14 days after trial end or subscription end)"""
    df['in_churn_period'] = (
        (df['status'] == 'active') & 
        (
            # Recently ended trial (within last 14 days)
            ((df['trial_end_utc'].notna()) & 
             (df['trial_end_utc'] <= reference_date) & 
             (df['trial_end_utc'] + pd.Timedelta(days=14) >= reference_date)) |
            # Current period recently ended (within last 14 days) 
            ((df['current_period_end_utc'].notna()) &
             (df['current_period_end_utc'] <= reference_date) & 
             (df['current_period_end_utc'] + pd.Timedelta(days=14) >= reference_date))
        )
    )
    return df

df = in_churn_period(df)

# %%

# CANCEL DURING TRIAL PERIOD
def cancel_during_trial(df):
    """Check if a member canceled during their trial period"""
    # Ensure columns are in datetime format
    df['canceled_during_trial'] = (
        (df['canceled_at_utc'].notna()) & 
        (df['trial_end_utc'] > df['canceled_at_utc']) 
    )
    return df

df = cancel_during_trial(df) 


# CANCEL DURING CHURN PERIOD
# if not canceled during trial, check if canceled during churn period (14days after trial end)
def cancel_during_churn(df):
    """Check if a member canceled during their churn period (14 days after trial end)"""
    # Ensure columns are in datetime format
    df['canceled_during_churn'] = (
        (df['canceled_during_trial'] == False) &
        (df['canceled_at_utc'].notna()) & 
        (df['trial_end_utc'] + pd.Timedelta(days=14) > df['canceled_at_utc']) &
        (df['trial_end_utc'].notna())
    )
    return df

df = cancel_during_churn(df)

# %%

df_filtered = df[df['trial_only_subscription'] == False]
df_filtered = df_filtered[['customer_name', 'status', 'created_utc',
       'start_utc', 'current_period_start_utc', 'current_period_end_utc',
       'trial_start_utc', 'trial_end_utc', 'canceled_at_utc', 'ended_at_utc',
       'is_gifted_member', 'trial_duration', 'current_period_duration',
       'trial_only_subscription', 'gift_duration', 'end_in', 'total_duration']]
df_filtered[df_filtered['total_duration'].isin([0, 10])]


