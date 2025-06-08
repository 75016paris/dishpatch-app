# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tseries.offsets import Day
import seaborn as sns
import os
from datetime import datetime

# %%

# Setting up the plotting style
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 10, 'axes.titlesize': 16})
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
#plt.rcParams['xtick.color'] = 'black'
plt.rcParams['xtick.color'] = 'white'
#plt.rcParams['ytick.color'] = 'black'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['figure.figsize'] = (22, 11)

# Grid with opacity and in background
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = 'lightgray'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['axes.axisbelow'] = True

plt.rcParams['axes.titleweight'] = 'bold'
#plt.rcParams['axes.titlecolor'] = 'black'
plt.rcParams['axes.titlecolor'] = 'white'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['legend.labelcolor'] = 'black'
plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['legend.edgecolor'] = 'gray'
plt.rcParams['text.color'] = 'black'
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
        'Customer Name', 'Customer ID', 'Status', 'Cancellation Reason',
        'Created (UTC)', 'Start (UTC)', 'Current Period Start (UTC)', 
        'Current Period End (UTC)', 'Trial Start (UTC)', 'Trial End (UTC)',
        'Canceled At (UTC)', 'Ended At (UTC)', 'senderShopifyCustomerId (metadata)'
    ]
    
    df = df[columns_to_keep]

    df.rename(columns={
        'Customer ID': 'customer_id',
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

    # Concert past_due and incomplete_expired to canceled
    df['status'] = df['status'].replace({'past_due': 'canceled', 'incomplete_expired': 'canceled'})

    # Reference date for analysis
    print('--------------------------------------')
    print(f"📅 Reference date (TODAY) : {reference_date.strftime('%d-%m-%Y')}")
    print(f"{len(df)} entries loaded from {file_path}")
    print('--------------------------------------')

    return df


df = preprocess_data(df_raw)

# %%

# # Adding end date for subscriptions that are canceled but have no ended_at_utc date.
# def add_ended_at_for_canceled(df):
#     """Set ended_at_utc to current_period_end_utc for canceled subscriptions without ended_at_utc"""
#
#     # Ensure ended_at_utc is set to current_period_end_utc for canceled subscriptions
#     mask = (df['ended_at_utc'].isna()) & (df['canceled_at_utc'].notna()) & (df['status'] == 'canceled') 
#
#     df.loc[mask, 'ended_at_utc'] = df['current_period_end_utc']
#
#
#     mask2 = (df['ended_at_utc'].isna()) & (df['trial_end_utc'].notna()) & ((df['status'] == 'canceled'))
#
#     df.loc[mask2, 'ended_at_utc'] = df['trial_end_utc']
#
#
#     mask3 = (df['ended_at_utc'].isna()) & (df['trial_end_utc'].isna()) & ((df['status'] == 'canceled'))
#
#     df.loc[mask3, 'ended_at_utc'] = df['current_period_start_utc']
#
#
#     return df
#
#
# df = add_ended_at_for_canceled(df)

def add_ended_at_for_canceled(df):

    cancel_mask = (df['status'] == 'canceled') & (df['ended_at_utc'].isna()) & (df['canceled_at_utc'].notna()) 
    active_mask = (df['status'] == 'active') & (df['ended_at_utc'].isna() ) & (df['canceled_at_utc'].notna())
    past_due_mask = (df['status'] == 'canceled') & (df['ended_at_utc'].isna()) & (df['canceled_at_utc'].isna())

    mask1 = cancel_mask & \
        (df['trial_end_utc'].notna()) & \
        (df['canceled_at_utc'] <= df['trial_end_utc'])
    # If canceled before trial end, set ended_at_utc to trial_end_utc
    df.loc[mask1, 'ended_at_utc'] = df['trial_end_utc']

    mask2 = cancel_mask  & \
        (df['trial_end_utc'].notna()) & \
        (df['canceled_at_utc'] > df['trial_end_utc']) & \
        (df['canceled_at_utc'] <= df['trial_end_utc'] + pd.Timedelta(days=14))
    # If canceled after trial end but within 14 days, set ended_at_utc to canceled_at_utc
    df.loc[mask2, 'ended_at_utc'] = df['canceled_at_utc']


    mask3 = cancel_mask & \
        (df['trial_end_utc'].notna()) & \
        (df['canceled_at_utc'] > df['trial_end_utc'] + pd.Timedelta(days=14))
    # If canceled after trial end and more than 14 days, set ended_at_utc to current_period_end_utc
    df.loc[mask3, 'ended_at_utc'] = df['current_period_end_utc']

    mask4 = cancel_mask  & \
        (df['trial_end_utc'].isna()) 
    # If canceled with no trial end, set ended_at_utc to current_period_start_utc
    df.loc[mask4, 'ended_at_utc'] = df['current_period_start_utc'] 

    mask5 = active_mask
    # If active subscription with no ended_at_utc, set ended_at_utc to current_period_end_utc
    df.loc[mask5, 'ended_at_utc'] = df['current_period_end_utc']

    mask6 = active_mask & \
        (df['trial_end_utc'].notna()) & \
        (df['canceled_at_utc'] >= df['trial_end_utc'] + pd.Timedelta(days=14))
    # If active and canceled after trial end + 14 days, set ended_at_utc to current_period_end_utc
    df.loc[mask6, 'ended_at_utc'] = df['canceled_at_utc']

    mask7 = (df['status'] == 'trialing') & (df['ended_at_utc'].isna()) & (df['canceled_at_utc'].notna())
    # If trialing and canceled, set ended_at_utc to trial_end_utc
    df.loc[mask7, 'ended_at_utc'] = df['trial_end_utc']

    df.loc[past_due_mask, 'ended_at_utc'] = df['current_period_start_utc']
    df.loc[past_due_mask, 'canceled_at_utc'] = df['current_period_start_utc']


    return df

df = add_ended_at_for_canceled(df)

print('***************************************************')
print(f'{df.info()}')
print('***************************************************')
trouble_df = df[df['canceled_at_utc'].notna() & df['ended_at_utc'].isna()]
print(trouble_df['status'].value_counts())


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
df_original = df.copy()  # Keep original for later analysis


# Si canceled_at < trial_end → annulation pendant trial
# Si trial_end < canceled_at < trial_end + 14 jours → annulation pendant refund
# %%

# def clean_customer_data(df):
#     """
#     Clean and prepare membership data for analysis
#     Removes very short subscriptions and keeps most recent subscription for duplicates
#     """
#     original_count = len(df)
#
#     # Calculate duration_days handling active subscriptions
#     df['ended_or_now'] = df['ended_at_utc'].fillna(reference_date)
#     df['duration_days'] = (df['ended_or_now'] - df['created_utc']).dt.days
#
#     # Remove very short durations for non-active subscriptions (~15 minutes)
#     df_clean = df[~((df['duration_days'] < 0.01) & (df['status'] != 'active'))]
#
#     # Remove duplicates: keep most recent subscription for same customer within 2 hours
#     df_clean = df_clean.sort_values(['customer_name', 'created_utc'])
#     df_clean['time_diff'] = df_clean.groupby('customer_name')['created_utc'].diff()
#
#     # Remove earlier subscriptions (keep most recent)
#     df_clean = df_clean[~((df_clean['time_diff'] < pd.Timedelta(hours=2)) & (df_clean['time_diff'].notna()))]
#
#     removed_count = original_count - len(df_clean)
#     print(f"📊 {original_count} → {len(df_clean)} subscriptions after cleaning")
#     print(f"🧹 Removed {removed_count} subscriptions ({removed_count/original_count*100:.1f}%)")
#
#     return df_clean.drop(['duration_days', 'time_diff', 'ended_or_now'], axis=1)
#
# df = clean_customer_data(df)
#
def clean_customer_data_preserve_business_stats(df):
    """
    Clean subscription data while preserving business statistics
    
    Rules:
    1. KEEP ALL subscriptions with 'active' status (regardless of duration)
    2. Remove technical duplicates (< 15 minutes between subscriptions from same customer)
    3. Ensure only one active subscription per customer (keep most recent)
    
    Returns:
        DataFrame: Cleaned data without biasing conversion stats
        list: Customers with multiple active subscriptions
    """
    original_count = len(df)
    print(f"Rows before cleaning: {original_count}")
    
    # ===== STEP 1: Remove technical duplicates (< 15 min) =====
    df_clean = df.sort_values(['customer_name', 'created_utc'])
    df_clean['next_time'] = df_clean.groupby('customer_name')['created_utc'].shift(-1)
    df_clean['time_to_next'] = df_clean['next_time'] - df_clean['created_utc']
    
    # Mark old duplicates (< 15 minutes) but NEVER remove active subscriptions
    is_technical_duplicate = (df_clean['time_to_next'] < pd.Timedelta(minutes=15)) & df_clean['time_to_next'].notna()
    duplicate_non_active = is_technical_duplicate & (df_clean['status'] != 'active')
    
    # Remove ONLY non-active duplicates
    df_clean = df_clean[~duplicate_non_active]
    
    # ===== STEP 2: Ensure one active subscription per customer =====
    # Find customers with multiple active subscriptions
    active_df = df_clean[df_clean['status'] == 'active']
    customer_active_counts = active_df['customer_name'].value_counts()
    multi_active_customers = customer_active_counts[customer_active_counts > 1].index.tolist()
    
    if len(multi_active_customers) > 0:
        def keep_most_recent_active_only(group):
            active_subs = group[group['status'] == 'active']
            non_active_subs = group[group['status'] != 'active']
            
            if len(active_subs) <= 1:
                return group
            
            # Keep most recent active subscription
            most_recent_active_idx = active_subs['created_utc'].idxmax()
            most_recent_active = active_subs.loc[[most_recent_active_idx]]
            
            # Combine non-active + most recent active
            result = pd.concat([non_active_subs, most_recent_active])
            return result.sort_values('created_utc')
        
        df_clean = df_clean.groupby('customer_name').apply(keep_most_recent_active_only).reset_index(drop=True)
    
    # Clean temporary columns
    columns_to_drop = ['next_time', 'time_to_next']
    columns_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
    df_clean = df_clean.drop(columns_to_drop, axis=1)
    
    final_count = len(df_clean)
    print(f"Rows after cleaning: {final_count}")
    print(f"Customers with multiple active subscriptions: {len(multi_active_customers)}")
    
    return df_clean, multi_active_customers


# df, multi_active_customers = clean_customer_data_preserve_business_stats(df)


# %%

# # CALCULATING DURATIONS
def calculate_duration(df):
    """Calculate various durations in days with proper business logic"""
    
    # Trial duration (if trial exists)
    df['trial_duration'] = (df['trial_end_utc'] - df['trial_start_utc']).dt.days.fillna(0)
    
    # Current period duration
    df['current_period_duration'] = (df['current_period_end_utc'] - df['current_period_start_utc']).dt.days
    
    # Trial-only subscription
    df['trial_only_subscription'] = (
        df['trial_start_utc'].notna() & 
        df['trial_end_utc'].notna() & 
        (df['trial_duration'] == df['current_period_duration'])
    )
    
    # Gift duration (only for gifted members)
    df['gift_duration'] = df['current_period_duration'].where(df['is_gifted_member'], 0)
    
    # Days until end for active subscriptions
    df['end_in'] = ((df['current_period_end_utc'] - reference_date).dt.days).where(df['status'] == 'active', np.nan)
    
    # For active subscriptions: from created_utc to current_period_end_utc (projected)
    # For ended subscriptions: from created_utc to ended_at_utc (actual)
    df['expected_duration'] = np.where(
        df['status'] == 'active',
        (df['current_period_end_utc'] - df['created_utc']).dt.days,  # Active: projected duration
        (df['ended_at_utc'] - df['created_utc']).dt.days             # Ended: actual duration
    )

    df['real_duration'] = np.where(
            df['ended_at_utc'].notna(),
            (df['ended_at_utc'] - df['created_utc']).dt.days,  # Ended: actual duration
            (reference_date - df['created_utc']).dt.days  # Active: duration until now
    )
    
    
    # Void duration (time between creation and start - should be minimal)
    df['void_duration'] = (df['start_utc'] - df['created_utc']).dt.days
    
   
    return df

df = calculate_duration(df)


# %%

# IN REFUND PERIOD

def in_refund_period(df):
   """Check if a member is in the refund period (14 days after trial end or subscription start)"""
   df['in_refund_period'] = (
       (df['status'] == 'active') & 
       (
           # Post-trial refund: 14 days after trial ends
           ((df['trial_end_utc'].notna()) & 
            (df['trial_only_subscription'] == False) &
            (df['trial_end_utc'] <= reference_date) & 
            (df['trial_end_utc'] + pd.Timedelta(days=14) >= reference_date)) |
           # Post-renewal refund: 14 days after period starts 
           ((df['current_period_start_utc'].notna()) &
            (df['current_period_start_utc'] <= reference_date) & 
            (df['current_period_start_utc'] + pd.Timedelta(days=14) >= reference_date))
       )
   )
   return df

df = in_refund_period(df)


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


# CANCEL DURING REFUND PERIOD
# if not canceled during trial, check if canceled during refund period (14days after trial end)
def cancel_during_refund_period(df):
    """Check if canceled during refund period (14 days after trial end OR subscription start)"""
    
    # For subscriptions with trials: 14 days after trial end
    trial_refund_condition = (
        (df['trial_end_utc'].notna()) &
        (df['canceled_at_utc'].notna()) &
        (df['canceled_at_utc'] > df['trial_end_utc']) &
        (df['canceled_at_utc'] <= df['trial_end_utc'] + pd.Timedelta(days=14))
    )
    
    # For subscriptions without trials: 14 days after start
    no_trial_refund_condition = (
        (df['trial_end_utc'].isna()) &
        (df['canceled_at_utc'].notna()) &
        (df['start_utc'].notna()) &
        (df['canceled_at_utc'] <= df['current_period_start_utc'] + pd.Timedelta(days=14))
    )
    
    df['canceled_during_refund_period'] = (
        (~df['canceled_during_trial']) &
        (df['canceled_at_utc'].notna()) & 
        (trial_refund_condition | no_trial_refund_condition)
    )

    return df


df = cancel_during_refund_period(df)

# %%

# MULTI SUBSCRIPTION ANALYSIS
df_multi = df.copy()

# %%

# GROUPBY CUSTOMER NAME
def groupby_customer_name(df):
    """Group by customer name and aggregate relevant metrics"""
    
    df_grouped = df.groupby('customer_name').agg({
        'customer_id': 'first',
        'status': 'max',
        'created_utc': 'min',
        'start_utc': 'min',
        'current_period_start_utc': 'min',
        'current_period_end_utc': 'max',
        'trial_start_utc': 'min',
        'trial_end_utc': 'max',
        'canceled_at_utc': 'max',
        'ended_at_utc': 'max',
        'is_gifted_member': 'any',
        'canceled_during_trial': 'any',
        'canceled_during_refund_period': 'any',
        'trial_duration': 'sum',
        'current_period_duration': 'sum',
        'real_duration': 'sum',
        'expected_duration': 'sum',
        'void_duration': 'sum'
    }).reset_index()
    
    # Logique pour full_member : pas d'abandon précoce
    df_grouped['full_member'] = (
        ~df_grouped['canceled_during_trial'] & 
        ~df_grouped['canceled_during_refund_period']
    )
    
    # Clients qui ont upgradé depuis un gift
    df_grouped['upgraded_from_gift'] = (
        df_grouped['is_gifted_member'] &  # A eu au moins un gift
        df_grouped['full_member']         # ET est devenu full member
    )
    
    return df_grouped

df = groupby_customer_name(df)

# %%

print(f'\nGROUPED BY CUSTOMER NAME STATUS - {len(df)}')
print(df['status'].value_counts())

print(f'\nALL SUBSCRIPTIONS STATUS - {len(df_multi)}')
print(df_multi['status'].value_counts())

# %%


# SUBSCRIPTION TIMELINE
events = []
for _, row in df.iterrows():  # Fixed syntax
    # Creation (+1 total, +1 historically active)
    events.append({
        'date': row['created_utc'].date(),
        'total_change': +1,
        'historical_active_change': +1,
        'still_active_change': 0  # Will calculate differently
    })
    
    # Subscription end (-1 historically active)
    if pd.notna(row['ended_at_utc']):
        events.append({
            'date': row['ended_at_utc'].date(),
            'total_change': 0,
            'historical_active_change': -1,
            'still_active_change': 0
        })

events_df = pd.DataFrame(events)
events_df = events_df.sort_values('date')

# Group by date
daily_changes = events_df.groupby('date').sum()

# Complete timeline
date_range = pd.date_range(
    start=daily_changes.index.min(),
    end=pd.Timestamp.now().date(),
    freq='D'
)

daily_changes = daily_changes.reindex(date_range, fill_value=0)

# Calculate cumulative values
total_cumulative = daily_changes['total_change'].cumsum()
historical_active_cumulative = daily_changes['historical_active_change'].cumsum()

# For "Still Active Today": count only those currently active
still_active_events = []
for _, row in df[df['status'] == 'active'].iterrows():
    still_active_events.append({
        'date': row['created_utc'].date(),
        'change': +1
    })

still_active_df = pd.DataFrame(still_active_events)
if not still_active_df.empty:
    still_active_daily = still_active_df.groupby('date')['change'].sum()
    still_active_daily = still_active_daily.reindex(date_range, fill_value=0)
    still_active_cumulative = still_active_daily.cumsum()
else:
    still_active_cumulative = pd.Series(0, index=date_range)

# Final plot
plt.figure(figsize=(14, 8))
plt.plot(total_cumulative.index, total_cumulative.values, 
         label='Total Subscriptions', color='gray', alpha=0.5)
plt.plot(historical_active_cumulative.index, historical_active_cumulative.values, 
         label='Historically Active', color='black')

# Get final values - CORRECTED
final_total = total_cumulative.iloc[-1]
current_historical = historical_active_cumulative.iloc[-1]  # Current value
peak_historical = historical_active_cumulative.max()       # Peak value
still_active_today = still_active_cumulative.iloc[-1]      # Current value, not max

plt.title('Subscription Timeline')
plt.xlabel('Date')
plt.ylabel('Number of Subscriptions')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Corrected prints
print(f"Total Subscriptions: {final_total:,.0f}")
print(f"Historical Peak ({historical_active_cumulative.idxmax().date()}): {peak_historical:,.0f}")
print(f"Active Today: {still_active_today:,.0f}")



# SUBSCRIPTION TIMELINE WITH FULL MEMBERS
events = []

# Pour tous les abonnements (Total et Historically Active)
for _, row in df.iterrows():
    # Creation (+1 total, +1 historically active)
    events.append({
        'date': row['created_utc'].date(),
        'total_change': +1,
        'historical_active_change': +1,
        'full_member_change': 0  # Sera calculé séparément
    })
    
    # Subscription end (-1 historically active)
    if pd.notna(row['ended_at_utc']):
        events.append({
            'date': row['ended_at_utc'].date(),
            'total_change': 0,
            'historical_active_change': -1,
            'full_member_change': 0
        })

# Pour les full members uniquement
full_members = df[df['full_member'] == True]
print(f"Total customers: {len(df)}")
print(f"Full members: {len(full_members)} ({len(full_members)/len(df)*100:.1f}%)")

for _, row in full_members.iterrows():
    # Creation (+1 full member)
    events.append({
        'date': row['created_utc'].date(),
        'total_change': 0,
        'historical_active_change': 0,
        'full_member_change': +1
    })
    
    # Subscription end (-1 full member)
    if pd.notna(row['ended_at_utc']):
        events.append({
            'date': row['ended_at_utc'].date(),
            'total_change': 0,
            'historical_active_change': 0,
            'full_member_change': -1
        })

events_df = pd.DataFrame(events)
events_df = events_df.sort_values('date')

# Group by date
daily_changes = events_df.groupby('date').sum()

# Complete timeline
date_range = pd.date_range(
    start=daily_changes.index.min(),
    end=pd.Timestamp.now().date(),
    freq='D'
)

daily_changes = daily_changes.reindex(date_range, fill_value=0)

# Calculate cumulative values
total_cumulative = daily_changes['total_change'].cumsum()
historical_active_cumulative = daily_changes['historical_active_change'].cumsum()
full_member_cumulative = daily_changes['full_member_change'].cumsum()

# For "Still Active Today": count only those currently active
still_active_events = []
for _, row in df[df['status'] == 'active'].iterrows():
    still_active_events.append({
        'date': row['created_utc'].date(),
        'change': +1
    })

still_active_df = pd.DataFrame(still_active_events)
if not still_active_df.empty:
    still_active_daily = still_active_df.groupby('date')['change'].sum()
    still_active_daily = still_active_daily.reindex(date_range, fill_value=0)
    still_active_cumulative = still_active_daily.cumsum()
else:
    still_active_cumulative = pd.Series(0, index=date_range)

# For "Full Members Still Active Today"
full_active_events = []
for _, row in full_members[full_members['status'] == 'active'].iterrows():
    full_active_events.append({
        'date': row['created_utc'].date(),
        'change': +1
    })

full_active_df = pd.DataFrame(full_active_events)
if not full_active_df.empty:
    full_active_daily = full_active_df.groupby('date')['change'].sum()
    full_active_daily = full_active_daily.reindex(date_range, fill_value=0)
    full_active_cumulative = full_active_daily.cumsum()
else:
    full_active_cumulative = pd.Series(0, index=date_range)

# Final plot
plt.figure(figsize=(14, 8))
plt.plot(total_cumulative.index, total_cumulative.values, 
         label='Total Subscriptions', color='gray', alpha=0.5, linewidth=2)
plt.plot(historical_active_cumulative.index, historical_active_cumulative.values, 
         label='Historically Active', color='black', linewidth=2)
plt.plot(full_member_cumulative.index, full_member_cumulative.values, 
         label='Full Members', color='blue', linewidth=2)
plt.plot(full_active_cumulative.index, full_active_cumulative.values, 
         label='Full Members (Active Today)', color='green', linewidth=2)

# Get final values
final_total = total_cumulative.iloc[-1]
current_historical = historical_active_cumulative.iloc[-1]
peak_historical = historical_active_cumulative.max()
still_active_today = still_active_cumulative.iloc[-1]
current_full_members = full_member_cumulative.iloc[-1]
peak_full_members = full_member_cumulative.max()
full_active_today = full_active_cumulative.iloc[-1]

plt.title('Subscription Timeline - with Full Members')
plt.xlabel('Date')
plt.ylabel('Number of Subscriptions')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Statistics
print(f"\n=== METRICS ===")
print(f"Total Subscriptions: {final_total:,.0f}")
print(f"Historical Peak ({historical_active_cumulative.idxmax().date()}): {peak_historical:,.0f}")
print(f"Currently Historical: {current_historical:,.0f}")
print(f"Active Today: {still_active_today:,.0f}")
print(f"\n=== FULL MEMBERS ===")
print(f"Full Members Peak ({full_member_cumulative.idxmax().date()}): {peak_full_members:,.0f}")
print(f"Currently Full Members: {current_full_members:,.0f}")
print(f"Full Members Active Today: {full_active_today:,.0f}")
print(f"\n=== QUALITY METRICS ===")
print(f"Full Member Rate: {(current_full_members/final_total)*100:.1f}%")
print(f"Full Member Retention: {(full_active_today/current_full_members)*100:.1f}%")
print(f"Overall Retention: {(still_active_today/final_total)*100:.1f}%")
