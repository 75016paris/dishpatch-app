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
# today_date = pd.Timestamp.now(tz='UTC')
today_date = pd.Timestamp('2025-05-23', tz='UTC')  # For testing purposes

# Set REFUND PERDIOD DURATION
REFUND_PERIOD_DAYS = 14  # Duration of the refund period in days
# Set thresholds for cleaning
HIGH_VOLUME_THRESHOLD = 5
DUPLICATE_THRESHOLD_MINUTES = 15


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
        'id', 'Customer Name', 'Customer ID', 'Status', 'Cancellation Reason',
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
    print(f"📅 Reference date (TODAY) : {today_date.strftime('%d-%m-%Y')}")
    print(f"{len(df)} entries loaded from {file_path}")
    print('--------------------------------------')

    return df


df = preprocess_data(df_raw)
#####################################################################################################
# %%

def investigate_gift_cancellations(df):
    """
    Analyse les patterns de cancellation des gifts
    """
    gifts = df[df['is_gifted_member'] == True].copy()
    
    print(f"=== ANALYSE DES {len(gifts)} GIFTS ===")
    
    # 1. Timing de cancellation
    gifts['time_to_cancel'] = (gifts['canceled_at_utc'] - gifts['created_utc']).dt.total_seconds()
    
    print("\n📊 TIMING DE CANCELLATION:")
    print(f"Cancelés immédiatement (0 sec): {(gifts['time_to_cancel'] == 0).sum()}")
    print(f"Cancelés < 1 min: {(gifts['time_to_cancel'] < 60).sum()}")
    print(f"Cancelés < 1 heure: {(gifts['time_to_cancel'] < 3600).sum()}")
    print(f"Jamais cancelés: {gifts['canceled_at_utc'].isna().sum()}")
    
    # 2. Status des gifts
    print(f"\n📈 STATUS ACTUEL:")
    print(gifts['status'].value_counts())
    
    # 3. Raisons de cancellation
    print(f"\n❌ RAISONS DE CANCELLATION:")
    print(gifts['cancellation_reason'].value_counts())
    
    # 4. Relation avec les périodes
    gifts['same_as_created'] = gifts['canceled_at_utc'] == gifts['created_utc']
    gifts['same_as_period_end'] = gifts['canceled_at_utc'] == gifts['current_period_end_utc']
    
    print(f"\n🕐 PATTERNS TEMPORELS:")
    print(f"Canceled = Created: {gifts['same_as_created'].sum()}")
    print(f"Canceled = Period End: {gifts['same_as_period_end'].sum()}")
    
    # 5. Échantillon détaillé
    print(f"\n🔍 ÉCHANTILLON (5 premiers gifts):")
    sample_cols = ['customer_name', 'status', 'created_utc', 'canceled_at_utc', 
                   'current_period_end_utc', 'time_to_cancel', 'cancellation_reason']
    print(gifts[sample_cols].head())
    
    return gifts

# Utilisez comme ça :
gift_analysis = investigate_gift_cancellations(df)  



# %%
######################################################################################################

def refund_period_end_utc(df, REFUND_PERIOD_DAYS):
    df['refund_period_end_utc'] = np.where(
        df['trial_end_utc'].notna(), df['trial_end_utc'] + pd.Timedelta(days=REFUND_PERIOD_DAYS),
        df['current_period_start_utc'] + pd.Timedelta(days=REFUND_PERIOD_DAYS))

    return df

df = refund_period_end_utc(df, REFUND_PERIOD_DAYS)

# %%

def add_ended_at_for_canceled(df):

    cancel_mask = (df['status'] == 'canceled') & (df['ended_at_utc'].isna()) & (df['canceled_at_utc'].notna()) 
    
    mask1 = cancel_mask & \
        (df['trial_end_utc'].notna()) & \
        (df['canceled_at_utc'] <= df['trial_end_utc'])
    df.loc[mask1, 'ended_at_utc'] = df['trial_end_utc']

    mask2 = cancel_mask  & \
        (df['trial_end_utc'].notna()) & \
        (df['canceled_at_utc'] > df['trial_end_utc']) & \
        (df['canceled_at_utc'] <= df['trial_end_utc'] + pd.Timedelta(days=14))
    df.loc[mask2, 'ended_at_utc'] = df['canceled_at_utc']

    mask3 = cancel_mask & \
        (df['trial_end_utc'].notna()) & \
        (df['canceled_at_utc'] > df['trial_end_utc'] + pd.Timedelta(days=14))
    df.loc[mask3, 'ended_at_utc'] = df['current_period_end_utc']

    mask4 = cancel_mask  & \
        (df['trial_end_utc'].isna()) & \
        (df['canceled_at_utc'] > df['refund_period_end_utc'])
    # If canceled with no trial end, set ended_at_utc to current_period_start_utc
    df.loc[mask4, 'ended_at_utc'] = df['current_period_end_utc'] 

    mask4b = cancel_mask & \
        (df['trial_end_utc'].isna()) & \
        (df['canceled_at_utc'] <= df['refund_period_end_utc'])
    # If canceled with no trial end, set ended_at_utc to canceled_at_utc
    df.loc[mask4b, 'ended_at_utc'] = df['canceled_at_utc']


    active_mask = (df['status'] == 'active') & (df['ended_at_utc'].isna() ) & (df['canceled_at_utc'].notna())

    mask5 = active_mask & \
            (df['canceled_at_utc'] > df['refund_period_end_utc']) 
    df.loc[mask5, 'ended_at_utc'] = df['current_period_end_utc']


    mask6 = active_mask & \
            (df['canceled_at_utc'] < df['refund_period_end_utc']) 
    df.loc[mask6, 'ended_at_utc'] = df['canceled_at_utc']   


    mask7 = (df['status'] == 'trialing') & (df['ended_at_utc'].isna()) & (df['canceled_at_utc'].notna())
    # If trialing and canceled, set ended_at_utc to trial_end_utc
    df.loc[mask7, 'ended_at_utc'] = df['trial_end_utc']


    past_due_mask = (df['status'] == 'canceled') & (df['ended_at_utc'].isna()) & (df['canceled_at_utc'].isna())

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
def remove_high_volume_customers(df, threshold=HIGH_VOLUME_THRESHOLD):
    """Remove customers with more than a specified number of subscriptions"""
    
    original_count = len(df)

    customer_counts = df['customer_id'].value_counts()
    high_volume_customers = customer_counts[customer_counts > threshold].index
    
    df = df[~df['customer_id'].isin(high_volume_customers)]
    
    print(f'{original_count - len(df)} subscriptions removed,')
    print(f'from {len(high_volume_customers)} customers with more than {threshold} subscriptions')
    print('--------------------------------------')

    return df


df = remove_high_volume_customers(df)
df_original = df.copy()  # Keep original for later analysis


# %%

#
def clean_customer_data_preserve_business_stats(df, DUPLICATE_THRESHOLD_MINUTES):
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
    df_clean = df.sort_values(['customer_id', 'created_utc'])
    df_clean['next_time'] = df_clean.groupby('customer_id')['created_utc'].shift(-1)
    df_clean['time_to_next'] = df_clean['next_time'] - df_clean['created_utc']
    
    # Mark old duplicates (< 15 minutes) but NEVER remove active subscriptions
    is_technical_duplicate = (df_clean['time_to_next'] < pd.Timedelta(minutes=DUPLICATE_THRESHOLD_MINUTES)) & df_clean['time_to_next'].notna()
    duplicate_non_active = is_technical_duplicate & (df_clean['status'] != 'active')
    
    # Remove ONLY non-active duplicates
    df_clean = df_clean[~duplicate_non_active]
    
    # ===== STEP 2: Ensure one active subscription per customer =====
    # Find customers with multiple active subscriptions
    active_df = df_clean[df_clean['status'] == 'active']
    customer_active_counts = active_df['customer_id'].value_counts()
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
        
        df_clean = df_clean.groupby('customer_id').apply(keep_most_recent_active_only).reset_index(drop=True)
    
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
    df['end_in'] = ((df['current_period_end_utc'] - today_date).dt.days).where(df['status'] == 'active', np.nan)
    
    # For active subscriptions: from created_utc to current_period_end_utc (projected)
    # For ended subscriptions: from created_utc to ended_at_utc (actual)
    df['expected_duration'] = np.where(
        (df['ended_at_utc'].isna()),
        (df['current_period_end_utc'] - df['created_utc']).dt.days,  # Active: projected duration
        (df['ended_at_utc'] - df['created_utc']).dt.days             # Ended: actual duration
    )

    df['real_duration'] = np.where(
            df['ended_at_utc'].notna(),
            (df['ended_at_utc'] - df['created_utc']).dt.days,  # Ended: actual duration
            (today_date - df['created_utc']).dt.days  # Active: duration until now
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
            (df['trial_end_utc'] <= today_date) & 
            (df['trial_end_utc'] + pd.Timedelta(days=14) >= today_date)) |
           # Post-renewal refund: 14 days after period starts 
           ((df['current_period_start_utc'].notna()) &
            (df['current_period_start_utc'] <= today_date) & 
            (df['current_period_start_utc'] + pd.Timedelta(days=14) >= today_date))
       ))
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
    df['canceled_during_refund_period'] = (
        (~df['canceled_during_trial']) &
        (df['canceled_at_utc'].notna()) & 
        (df['refund_period_end_utc'] >= df['canceled_at_utc'])
    )

    return df


df = cancel_during_refund_period(df)

# %%

# FULL MEMBER STATUS
def full_member_status(df):
    """Determine if a customer is a full member based on business logic"""
    
    # Full member if:
    # 1. Not canceled during trial
    # 2. Not canceled during refund period
    # 3. Not gifted
    # 4. Trial ended more than 14 days ago (if no trial, current_period_start_utc > 14 days ago)
    
    no_early_cancellation = (
        (~df['canceled_during_trial']) & 
        (~df['canceled_during_refund_period'])
    )

    not_gifted = (~df['is_gifted_member'])

    refund_period_passed = (today_date > df['refund_period_end_utc'])


    df['full_member'] = (
        no_early_cancellation & 
        not_gifted & 
        refund_period_passed
    )
    
    return df

df = full_member_status(df)


# %%

# GROUPBY CUSTOMER ID
def groupby_customer_id(df):
    """Group by customer name with proper business logic - English version"""
    
    # Sort by created_utc to have chronological order
    df_sorted = df.sort_values(['customer_id', 'created_utc'])
    
    def get_most_recent_value(series):
        """Get value from most recent subscription"""
        return series.iloc[-1]
    
   
    # Main aggregation
    customer_df = df_sorted.groupby('customer_id').agg({
        'customer_name': 'first',
        'created_utc': 'min',                    # First subscription
        'start_utc': 'min',                      # First start
        'canceled_at_utc': 'max',                # Last cancellation
        'ended_at_utc': 'max',                   # Last end
        'trial_duration': 'sum',                 # Total trial time
        'current_period_duration': 'sum',        # Total period time
        'real_duration': 'sum',                  # Total real duration
        'expected_duration': 'sum',              # Total expected duration
        'void_duration': 'sum'                   # Total void time
    }).reset_index()
    
    # Add values from the most recent subscription
    latest_subscription_data = df_sorted.groupby('customer_id').agg({
        'status': get_most_recent_value,                    # Most recent status
        'is_gifted_member': get_most_recent_value,          # Most recent gift status
        'current_period_start_utc': get_most_recent_value,  # Current period start
        'current_period_end_utc': get_most_recent_value,    # Current period end
        'trial_start_utc': get_most_recent_value,           # Most recent trial start
        'trial_end_utc': get_most_recent_value,             # Most recent trial end
        'canceled_during_trial': get_most_recent_value,          # Last trial cancellation
        'canceled_during_refund_period': get_most_recent_value,  # Last refund cancellation
        'full_member': get_most_recent_value,                # Last full member status
        'refund_period_end_utc': get_most_recent_value,      # Last refund period end
    }).reset_index()
    
    # Gifted history
    gift_history = df_sorted.groupby('customer_id').agg({
        'is_gifted_member': 'any'}).reset_index()
    gift_history.rename(columns={'is_gifted_member': 'ever_had_gift'}, inplace=True)


    # Merge the data
    customer_df = customer_df.merge(latest_subscription_data, on='customer_id')
    customer_df = customer_df.merge(gift_history, on='customer_id')
    
    
    # Upgraded from gift: had gifts AND became full member
    customer_df['upgraded_from_gift'] = (
        customer_df['ever_had_gift'] &      # A eu un cadeau avant
        ~customer_df['is_gifted_member'] &  # Subscription actuelle N'EST PAS un cadeau
        customer_df['full_member']          # Et est full member
    )
    
    return customer_df


customer_df = groupby_customer_id(df)


# %%
# WEEKS ARE FROM MONDAY TO SUNDAY

def get_specific_past_week(weeks_back=1, reference_date=None):
    """
    Get specific date for a specific week.
    weeks_back=1 : last week (from Monday to Sunday)
    weeks_back=2 : previous week (from Monday to Sunday)
    weeks_back=3 : three weeks ago (from Monday to Sunday)
    """

    if reference_date is None:
        today = pd.Timestamp.now(tz='UTC')
    else:
        if hasattr(reference_date, 'tz') and reference_date.tz is not None:
            today = pd.to_datetime(reference_date).tz_convert('UTC')
        else:
            today = pd.to_datetime(reference_date).tz_localize('UTC')
      
    
    # Finding the Monday of the target week
    days_since_monday = today.weekday()
    this_monday = today - pd.Timedelta(days=days_since_monday)
    target_monday = this_monday - pd.Timedelta(days=7 * weeks_back)
    target_sunday = target_monday + pd.Timedelta(days=6)

    week_start = target_monday.normalize()  # 00:00:00
    week_end = target_sunday.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)  # 23:59:59
    
    # Las week info
    week_info = {
        'weeks_ago': weeks_back,
        'week_start': week_start,
        'week_end': week_end, 
        'year': target_monday.year,
        'week_number': target_monday.isocalendar().week,
        'year_week': f"{target_monday.year}-W{target_monday.isocalendar().week:02d}",
    }
    
    return week_info

# %% 
##########################################################################################################

def analyze_stripe_date_patterns(df):
    """
    Analyze date patterns to understand Stripe's behavior during automatic renewals
    """
    print("🔍 STRIPE DATE PATTERNS ANALYSIS")
    print("=" * 50)
    
    # Filter subscriptions without trial and non-gifted
    no_trial_subs = df[(df['trial_duration'] == 0) & (~df['is_gifted_member'])].copy()
    
    # Calculate differences between key dates
    no_trial_subs['created_to_start_hours'] = (
        no_trial_subs['start_utc'] - no_trial_subs['created_utc']
    ).dt.total_seconds() / 3600
    
    no_trial_subs['created_to_period_start_hours'] = (
        no_trial_subs['current_period_start_utc'] - no_trial_subs['created_utc']
    ).dt.total_seconds() / 3600
    
    no_trial_subs['start_to_period_start_hours'] = (
        no_trial_subs['current_period_start_utc'] - no_trial_subs['start_utc']
    ).dt.total_seconds() / 3600
    
    print(f"📊 ANALYSIS OF {len(no_trial_subs)} SUBSCRIPTIONS WITHOUT TRIAL")
    print(f"\n🕐 TIME DIFFERENCES:")
    
    # Analysis created_utc vs start_utc
    print(f"Created → Start:")
    print(f"  - Same moment (0h): {(no_trial_subs['created_to_start_hours'] == 0).sum()}")
    print(f"  - < 1h: {(no_trial_subs['created_to_start_hours'].abs() < 1).sum()}")
    print(f"  - < 8h: {(no_trial_subs['created_to_start_hours'].abs() < 8).sum()}")
    print(f"  - Average: {no_trial_subs['created_to_start_hours'].mean():.2f}h")
    
    # Analysis created_utc vs current_period_start_utc
    print(f"\nCreated → Period Start:")
    print(f"  - Same moment (0h): {(no_trial_subs['created_to_period_start_hours'] == 0).sum()}")
    print(f"  - < 1h: {(no_trial_subs['created_to_period_start_hours'].abs() < 1).sum()}")
    print(f"  - < 8h: {(no_trial_subs['created_to_period_start_hours'].abs() < 8).sum()}")
    print(f"  - Average: {no_trial_subs['created_to_period_start_hours'].mean():.2f}h")
    
    # Analysis start_utc vs current_period_start_utc
    print(f"\nStart → Period Start:")
    print(f"  - Same moment (0h): {(no_trial_subs['start_to_period_start_hours'] == 0).sum()}")
    print(f"  - < 1h: {(no_trial_subs['start_to_period_start_hours'].abs() < 1).sum()}")
    print(f"  - < 8h: {(no_trial_subs['start_to_period_start_hours'].abs() < 8).sum()}")
    print(f"  - Average: {no_trial_subs['start_to_period_start_hours'].mean():.2f}h")
    
    # Look for suspicious patterns for renewals
    df_sorted = df.sort_values(['customer_id', 'created_utc'])
    
    potential_renewals = []
    for idx, row in no_trial_subs.iterrows():
        customer_id = row['customer_id']
        current_created = row['created_utc']
        
        # Customer history
        customer_history = df_sorted[
            (df_sorted['customer_id'] == customer_id) & 
            (df_sorted['created_utc'] < current_created)
        ]
        
        if len(customer_history) > 0:
            # Suspicious pattern: created_utc very close to current_period_start_utc
            # but customer has history
            created_to_period_hours = abs(row['created_to_period_start_hours'])
            
            if created_to_period_hours <= 8:  # Less than 8h difference
                last_sub = customer_history.iloc[-1]
                
                potential_renewals.append({
                    'customer_id': customer_id,
                    'current_created': current_created,
                    'current_period_start': row['current_period_start_utc'],
                    'hours_diff': created_to_period_hours,
                    'last_sub_created': last_sub['created_utc'],
                    'last_sub_canceled': last_sub['canceled_at_utc'],
                    'last_sub_ended': last_sub['ended_at_utc']
                })
    
    print(f"\n🚨 POTENTIAL RENEWALS DETECTED: {len(potential_renewals)}")
    
    if len(potential_renewals) > 0:
        renewal_df = pd.DataFrame(potential_renewals)
        print(f"Distribution by time gap:")
        print(f"  - < 1h: {(renewal_df['hours_diff'] < 1).sum()}")
        print(f"  - 1-8h: {((renewal_df['hours_diff'] >= 1) & (renewal_df['hours_diff'] <= 8)).sum()}")
        print(f"  - 8h+: {(renewal_df['hours_diff'] > 8).sum()}")
        
        # Show some examples
        print(f"\n📋 EXAMPLES (first 5):")
        for i, row in renewal_df.head().iterrows():
            print(f"  Customer {row['customer_id']}: {row['hours_diff']:.1f}h gap")
    
    return no_trial_subs, potential_renewals


def classify_no_trial_subscriptions(df):
    """
    New classification logic with updated rules:
    1. Gift-to-Pay: Had at least 1 gift + current subscription without trial + not currently gifted
    2. Renewal: Gap ≤ 1 day OR detected via Stripe patterns
    3. Winback: Gap > 1 day (regardless of whether previous was trial-only)
    4. Edge Case/Unknown: Everything else
    """
    
    df_sorted = df.sort_values(['customer_id', 'created_utc'])
    no_trial_mask = (df['trial_duration'] == 0) & (~df['is_gifted_member'])
    
    results = []
    
    for idx, row in df[no_trial_mask].iterrows():
        customer_id = row['customer_id']
        current_created = row['created_utc']
        current_period_start = row['current_period_start_utc']
        
        # Customer history
        customer_history = df_sorted[
            (df_sorted['customer_id'] == customer_id) & 
            (df_sorted['created_utc'] < current_created)
        ]
        
        # Stripe renewal pattern detection
        created_to_period_hours = abs((current_period_start - current_created).total_seconds() / 3600)
        potential_stripe_renewal = (len(customer_history) > 0) and (created_to_period_hours <= 8)
        
        if len(customer_history) == 0:
            # No history = first subscription
            classification = 'unknown'
            previous_status = None
            previous_created = None
            previous_canceled = None
            gap_days = None
            stripe_renewal_detected = False
            
        else:
            last_subscription = customer_history.iloc[-1]
            previous_status = last_subscription['status']
            previous_created = last_subscription['created_utc']
            previous_canceled = last_subscription['canceled_at_utc']
            stripe_renewal_detected = potential_stripe_renewal
            
            # 1. GIFT-TO-PAY: Had at least 1 gift in history
            had_gift = customer_history['is_gifted_member'].any()
            
            if had_gift:
                classification = 'gift_to_paid_winback'
                gap_days = None if pd.isna(previous_canceled) else (current_created - previous_canceled).days
                
            # 2. RENEWAL: Gap ≤ 1 day OR Stripe pattern detected
            elif (pd.notna(previous_canceled) and 
                  (current_created - previous_canceled).days <= 1) or potential_stripe_renewal:
                classification = 'renewal'
                gap_days = None if pd.isna(previous_canceled) else (current_created - previous_canceled).days
                
            # 3. WINBACK: Gap > 1 day
            elif (pd.notna(previous_canceled) and 
                  (current_created - previous_canceled).days > 1):
                classification = 'winback'
                gap_days = (current_created - previous_canceled).days
                
            # 4. EDGE CASE/UNKNOWN
            else:
                classification = 'unknown'
                gap_days = None
        
        results.append({
            'index': idx,
            'customer_id': customer_id,
            'classification': classification,
            'previous_status': previous_status,
            'previous_created': previous_created,
            'previous_canceled': previous_canceled,
            'gap_days': gap_days,
            'stripe_renewal_detected': stripe_renewal_detected,
            'created_to_period_hours': created_to_period_hours
        })
    
    return pd.DataFrame(results)


def detailed_weekly_analysis(df, weeks_back=1, reference_date=None):
    """
    Updated version of weekly analysis with new logic
    """
    if reference_date is None:
        reference_date = pd.Timestamp.now(tz='UTC')
        
    week_info = get_specific_past_week(weeks_back=weeks_back, reference_date=reference_date)
    week_start = week_info['week_start']
    week_end = week_info['week_end']

    print(f'🗓️  WEEK {week_info["year_week"]} ({weeks_back} week{"s" if weeks_back > 1 else ""} ago)')
    print(f'From Monday {week_start.strftime("%d/%m")} to Sunday {week_end.strftime("%d/%m/%Y")}')
    print('=' * 60)
    
    # NEW SUBSCRIPTIONS FOR THE WEEK
    new_subscriptions = df[
        (df['created_utc'] >= week_start) &
        (df['created_utc'] <= week_end)
    ]
    len_new_subscriptions = len(new_subscriptions)
    
    # TRIALERS
    new_trialers = new_subscriptions[new_subscriptions['trial_duration'] > 0]
    len_new_trialers = len(new_trialers)
    
    # GIFTED
    new_gifted = new_subscriptions[new_subscriptions['is_gifted_member'] == True]
    len_new_gifted = len(new_gifted)

    # CLASSIFICATION OF SUBSCRIPTIONS WITHOUT TRIAL
    classification_results = classify_no_trial_subscriptions(df)
    weekly_no_trial = new_subscriptions[
        (new_subscriptions['trial_duration'] == 0) & 
        (~new_subscriptions['is_gifted_member'])
    ].copy()

    if len(weekly_no_trial) == 0:
        len_renewals = len_winbacks = len_gift_to_paid = len_unknowns = 0
        len_stripe_renewals = 0
        renewal_subscriptions = winback_subscriptions = pd.DataFrame()
        gift_to_paid_subscriptions = unknown_subscriptions = pd.DataFrame()
    else:
        weekly_classified = weekly_no_trial.merge(
            classification_results, 
            left_index=True, 
            right_on='index', 
            how='left'
        )
        weekly_classified['classification'] = weekly_classified['classification'].fillna('unknown')
        
        # Count each category
        len_renewals = (weekly_classified['classification'] == 'renewal').sum()
        len_winbacks = (weekly_classified['classification'] == 'winback').sum()
        len_gift_to_paid = (weekly_classified['classification'] == 'gift_to_paid_winback').sum()
        len_unknowns = (weekly_classified['classification'] == 'unknown').sum()
        len_stripe_renewals = (weekly_classified['stripe_renewal_detected'] == True).sum()
        
        # Create DataFrames
        renewal_subscriptions = weekly_classified[weekly_classified['classification'] == 'renewal']
        winback_subscriptions = weekly_classified[weekly_classified['classification'] == 'winback']
        gift_to_paid_subscriptions = weekly_classified[weekly_classified['classification'] == 'gift_to_paid_winback']
        unknown_subscriptions = weekly_classified[weekly_classified['classification'] == 'unknown']

    # OTHER METRICS
    new_canceled_during_trial = df[
        (df['canceled_during_trial'] == True) &
        (df['canceled_at_utc'].notna()) &
        (df['canceled_at_utc'] >= week_start) &
        (df['canceled_at_utc'] <= week_end)
    ]
    len_new_canceled_during_trial = len(new_canceled_during_trial)
    
    new_canceled_during_refund = df[
        (df['canceled_during_refund_period'] == True) &
        (df['canceled_at_utc'].notna()) &
        (df['canceled_at_utc'] >= week_start) &
        (df['canceled_at_utc'] <= week_end)
    ]
    len_new_canceled_during_refund = len(new_canceled_during_refund)

    new_full_members = df[
        (df['full_member'] == True) &
        (df['refund_period_end_utc'] >= week_start) &
        (df['refund_period_end_utc'] <= week_end)
    ]
    len_new_full_members = len(new_full_members)
    
    # DISPLAY
    print(f"📈 {len_new_subscriptions} NEW SUBSCRIPTIONS")
    print(f"  ├─ 🎯 {len_new_trialers} New Trialers")
    print(f"  ├─ 🎁 {len_new_gifted} Gifted Members")
    print(f"  ├─ 🔄 {len_renewals} RENEWALS (automatic renewals)")
    if len_stripe_renewals > 0:
        print(f"  │   └─ 🤖 {len_stripe_renewals} detected via Stripe pattern")
    print(f"  ├─ 🔙 {len_winbacks} WINBACKS (returning customers)")
    print(f"  ├─ 💝 {len_gift_to_paid} Gift-to-Paid (ex-gifts → paid)")
    print(f"  └─ ❓ {len_unknowns} UNKNOWN")
    print()
    print(f"✅ {len_new_full_members} Became Full Members")
    print(f"❌ {len_new_canceled_during_trial} Canceled during trial")
    print(f"💔 {len_new_canceled_during_refund} Canceled during refund")
    print()

    # Clean merge columns for returned DataFrames
    def clean_merge_columns(df):
        if len(df) > 0 and 'customer_id_x' in df.columns:
            df = df.drop(['customer_id_y', 'index'], axis=1, errors='ignore')
            df = df.rename(columns={'customer_id_x': 'customer_id'})
        return df

    renewal_subscriptions = clean_merge_columns(renewal_subscriptions)
    winback_subscriptions = clean_merge_columns(winback_subscriptions) 
    gift_to_paid_subscriptions = clean_merge_columns(gift_to_paid_subscriptions)
    unknown_subscriptions = clean_merge_columns(unknown_subscriptions)
    
    # SUMMARY DICTIONARY
    week_dict = {
        'week_start': week_start, 'week_end': week_end, 'year_week': week_info['year_week'],
        'weeks_ago': week_info['weeks_ago'], 'new_subscriptions': len_new_subscriptions,
        'new_trialers': len_new_trialers, 'new_gifted': len_new_gifted,
        'renewals': len_renewals, 'winbacks': len_winbacks,
        'gift_to_paid_converters': len_gift_to_paid, 'unknowns': len_unknowns,
        'stripe_renewals_detected': len_stripe_renewals,
        'new_full_members': len_new_full_members,
        'new_canceled_during_trial': len_new_canceled_during_trial,
        'new_canceled_during_refund': len_new_canceled_during_refund
    }
    
    # DETAILED DATA
    detailed_data = {
        'renewal_subscriptions': renewal_subscriptions,
        'winback_subscriptions': winback_subscriptions,
        'gift_to_paid_subscriptions': gift_to_paid_subscriptions,
        'unknown_subscriptions': unknown_subscriptions,
        'new_gifted': new_gifted,
        'new_trialers': new_trialers
    }
   
    return week_dict, detailed_data


def analyze_winback_timing(winback_subscriptions, df):
    """Analyze timing of winback returns"""
    
    if len(winback_subscriptions) == 0:
        print("No winbacks this week.")
        return pd.DataFrame()
    
    winback_analysis = []
    
    for idx, winback in winback_subscriptions.iterrows():
        customer_id = winback['customer_id']
        previous_created = winback['previous_created']
        current_created = winback['created_utc']
        
        # Calculate return delay
        time_away = (current_created - previous_created).days
        
        winback_analysis.append({
            'customer_id': customer_id,
            'time_away_days': time_away,
            'previous_created': previous_created,
            'current_created': current_created,
            'time_away_months': round(time_away / 30.44, 1)  # Rough month approximation
        })
    
    winback_df = pd.DataFrame(winback_analysis)
    
    print(f"\n📊 WINBACK TIMING ANALYSIS ({len(winback_df)} customers):")
    print(f"Average return delay: {winback_df['time_away_days'].mean():.1f} days ({winback_df['time_away_months'].mean():.1f} months)")
    print(f"Median delay: {winback_df['time_away_days'].median():.0f} days ({winback_df['time_away_months'].median():.1f} months)")
    print(f"Fastest: {winback_df['time_away_days'].min()} days")
    print(f"Slowest: {winback_df['time_away_days'].max()} days ({winback_df['time_away_months'].max():.1f} months)")
    
    # Distribution by brackets
    bins = [0, 30, 90, 180, 365, float('inf')]
    labels = ['<1 month', '1-3 months', '3-6 months', '6-12 months', '>1 year']
    winback_df['time_category'] = pd.cut(winback_df['time_away_days'], bins=bins, labels=labels)
    
    print(f"\n🗓️ DISTRIBUTION BY PERIOD:")
    distribution = winback_df['time_category'].value_counts().sort_index()
    for category, count in distribution.items():
        percentage = (count / len(winback_df)) * 100
        print(f"  {category}: {count} customers ({percentage:.1f}%)")
    
    return winback_df


def test_new_classification(df, reference_date):
    """
    Test the new classification and compare with old one if needed
    """
    print("🧪 NEW CLASSIFICATION TEST")
    print("=" * 50)
    
    # 1. Date pattern analysis
    date_analysis, potential_renewals = analyze_stripe_date_patterns(df)
    
    print("\n" + "=" * 50)
    
    # 2. Test on last 4 weeks
    for week in range(1, 5):
        week_dict, detailed_data = detailed_weekly_analysis(df, weeks_back=week, reference_date=reference_date)
        
        # Quick winback analysis if any
        if len(detailed_data['winback_subscriptions']) > 0:
            winback_timing = analyze_winback_timing(detailed_data['winback_subscriptions'], df)
    
    return date_analysis, potential_renewals


# USAGE EXAMPLE - REPLACE THE END OF YOUR SCRIPT WITH THIS:

# Replace your current calls with this:
date_analysis, potential_renewals = test_new_classification(df, today_date)

# Or for specific week analysis:
# week_dict, detailed_data = detailed_weekly_analysis(df, weeks_back=1, reference_date=today_date)
# winback_timing = analyze_winback_timing(detailed_data['winback_subscriptions'], df)

def diagnose_stripe_data_coherence(df):
    """
    Diagnostic pour vérifier la cohérence des données avec la logique Stripe
    Maintenant avec la vraie colonne 'id' (subscription_id)
    """
    """
    Diagnostic pour vérifier la cohérence des données avec la logique Stripe
    """
    print("🔍 DIAGNOSTIC DE COHÉRENCE STRIPE")
    print("=" * 50)
    
    # 1. Vérifier les subscription_id (colonne 'id')
    has_subscription_id = 'id' in df.columns
    print(f"📋 Subscription ID présent: {has_subscription_id}")
    
    if has_subscription_id:
        # Cas où on a l'ID : vérifier les duplicatas
        duplicate_subs = df.groupby('id').size()
        duplicates = duplicate_subs[duplicate_subs > 1]
        print(f"🚨 Subscriptions dupliquées: {len(duplicates)}")
        
        if len(duplicates) > 0:
            print("⚠️  ALERTE: Plusieurs lignes pour même subscription_id")
            print("   Ceci viole la logique Stripe (1 sub = 1 ligne)")
            print(f"   Exemples: {duplicates.head().to_dict()}")
            
            # Analyser les duplicatas
            duplicate_examples = df[df['id'].isin(duplicates.index)].sort_values(['id', 'created_utc'])
            print("   Détail des duplicatas:")
            for sub_id in duplicates.index[:3]:  # 3 premiers exemples
                sub_data = duplicate_examples[duplicate_examples['id'] == sub_id]
                print(f"   {sub_id}:")
                for idx, row in sub_data.iterrows():
                    print(f"     - Created: {row['created_utc']}, Status: {row['status']}")
        else:
            print("✅ Pas de subscription_id dupliqué - données cohérentes")
    
    # 2. Analyser les patterns temporels suspects
    print(f"\n🕐 ANALYSE DES PATTERNS TEMPORELS")
    
    # Clients avec multiples souscriptions
    customer_counts = df['customer_id'].value_counts()
    multi_sub_customers = customer_counts[customer_counts > 1]
    print(f"👥 Clients avec multiples subscriptions: {len(multi_sub_customers)}")
    
    if len(multi_sub_customers) > 0:
        # Analyser les gaps temporels
        df_sorted = df.sort_values(['customer_id', 'created_utc'])
        df_sorted['prev_created'] = df_sorted.groupby('customer_id')['created_utc'].shift(1)
        df_sorted['gap_hours'] = (df_sorted['created_utc'] - df_sorted['prev_created']).dt.total_seconds() / 3600
        
        valid_gaps = df_sorted['gap_hours'].dropna()
        
        print(f"⏱️  Gaps entre subscriptions:")
        print(f"   < 1 heure: {(valid_gaps < 1).sum()} cas")
        print(f"   < 1 jour: {(valid_gaps < 24).sum()} cas") 
        print(f"   < 1 semaine: {(valid_gaps < 168).sum()} cas")
        print(f"   Médiane: {valid_gaps.median():.1f} heures")
        
        # Cas suspects (très courts gaps)
        suspect_gaps = valid_gaps[valid_gaps < 1]
        if len(suspect_gaps) > 0:
            print(f"🚨 {len(suspect_gaps)} gaps < 1h (suspects de duplicatas techniques)")
    
    # 3. Vérifier la cohérence des statuts
    print(f"\n📊 ANALYSE DES STATUTS")
    status_counts = df['status'].value_counts()
    print(f"Distribution des statuts:")
    for status, count in status_counts.items():
        pct = (count / len(df)) * 100
        print(f"   {status}: {count} ({pct:.1f}%)")
    
    # 4. Vérifier les champs critiques manquants
    print(f"\n📋 CHAMPS CRITIQUES")
    critical_fields = [
        'subscription_id', 'billing_reason', 'invoice_id', 
        'payment_intent_id', 'latest_invoice'
    ]
    
    for field in critical_fields:
        present = field in df.columns
        symbol = "✅" if present else "❌"
        print(f"   {symbol} {field}: {'présent' if present else 'manquant'}")
    
    # 5. Recommandations
    print(f"\n💡 RECOMMANDATIONS")
    
    if not has_subscription_id:
        print("⚠️  CRITIQUE: Ajoutez subscription_id pour une analyse correcte")
    
    if len(multi_sub_customers) > len(df) * 0.1:  # >10% de clients avec multi-subs
        print("⚠️  Beaucoup de multi-subscriptions. Vérifiez:")
        print("   - Source des données (Dashboard vs Webhook vs DB)")
        print("   - Processus d'export")
        print("   - Possible dédoublonnement technique")
    
    # 6. Test de cohérence Stripe spécifique
    print(f"\n🧪 TEST COHÉRENCE STRIPE")
    
    if has_subscription_id:
        # Vérifier unicité des subscription_ids
        unique_subs = df['id'].nunique()
        total_rows = len(df)
        print(f"   Subscriptions uniques: {unique_subs}")
        print(f"   Total lignes: {total_rows}")
        
        if unique_subs == total_rows:
            print("   ✅ 1 ligne = 1 subscription (cohérent avec Stripe)")
        else:
            print("   🚨 Plusieurs lignes pour certaines subscriptions")
            print("      Ceci contredit la logique Stripe native")
    
    # Test des patterns de renouvellement suspects
    df_sorted = df.sort_values(['customer_id', 'created_utc'])
    customer_multiple_subs = df_sorted.groupby('customer_id').size()
    customers_with_multiple = customer_multiple_subs[customer_multiple_subs > 1]
    
    if len(customers_with_multiple) > 0:
        print(f"\n📊 ANALYSE CLIENTS MULTIPLES SUBSCRIPTIONS")
        print(f"   {len(customers_with_multiple)} clients avec multiples subscriptions")
        
        # Analyser un exemple détaillé
        example_customer = customers_with_multiple.index[0]
        example_subs = df_sorted[df_sorted['customer_id'] == example_customer]
        
        print(f"\n   Exemple client {example_customer}:")
        for idx, row in example_subs.iterrows():
            status = row['status']
            created = row['created_utc'].strftime('%Y-%m-%d %H:%M')
            sub_id = row['id'][:20] + "..." if len(row['id']) > 20 else row['id']
            print(f"     {created} | {status} | {sub_id}")
    
    # Cas impossible selon Stripe
    impossible_cases = []
    
    # Trial qui commence après created
    trial_after_created = df[
        (df['trial_start_utc'].notna()) & 
        (df['trial_start_utc'] > df['created_utc'])
    ]
    if len(trial_after_created) > 0:
        impossible_cases.append(f"Trial start > created: {len(trial_after_created)} cas")
    
    # Current period start avant created
    period_before_created = df[
        (df['current_period_start_utc'].notna()) & 
        (df['current_period_start_utc'] < df['created_utc'])
    ]
    if len(period_before_created) > 0:
        impossible_cases.append(f"Period start < created: {len(period_before_created)} cas")
    
    if impossible_cases:
        print("🚨 CAS IMPOSSIBLES DÉTECTÉS:")
        for case in impossible_cases:
            print(f"   - {case}")
    else:
        print("✅ Pas d'incohérences temporelles détectées")
    
    return {
        'has_subscription_id': has_subscription_id,
        'duplicate_subscriptions': len(duplicates) if has_subscription_id else 0,
        'multi_subscription_customers': len(multi_sub_customers),
        'impossible_cases': len(impossible_cases)
    }

# Usage
diagnostic_result = diagnose_stripe_data_coherence(df)
#########################################################################################################
# %%
# def classify_no_trial_subscriptions(df):
#     """
#     Logique corrigée avec toutes les colonnes nécessaires
#     """
#     df_sorted = df.sort_values(['customer_id', 'created_utc'])
#     no_trial_mask = (df['trial_duration'] == 0) & (~df['is_gifted_member'])
#
#     results = []
#
#     for idx, row in df[no_trial_mask].iterrows():
#         customer_id = row['customer_id']
#         current_created = row['created_utc']
#
#         customer_history = df_sorted[
#             (df_sorted['customer_id'] == customer_id) & 
#             (df_sorted['created_utc'] < current_created)
#         ]
#
#         if len(customer_history) == 0:
#             classification = 'renewal'
#             previous_status = None
#             previous_created = None
#             previous_canceled = None
#             gap_days = None
#         else:
#             last_subscription = customer_history.iloc[-1]
#             previous_status = last_subscription['status']
#             previous_created = last_subscription['created_utc']  # ← AJOUTÉ
#             previous_canceled = last_subscription['canceled_at_utc']
#
#             # Vérifier si c'est un vrai winback (gap > 1 jour)
#             if (last_subscription['status'] in ['canceled', 'trialing'] and
#                 pd.notna(last_subscription['canceled_at_utc']) and
#                 pd.notna(current_created)):
#
#                 gap_days = (current_created - last_subscription['canceled_at_utc']).days
#
#                 if gap_days > 1:  # Plus d'1 jour = vrai winback
#                     had_gift = customer_history['is_gifted_member'].any()
#                     had_trial = (customer_history['trial_duration'] > 0).any()
#
#                     if had_gift and not had_trial:
#                         classification = 'gift_to_paid_winback'
#                     elif had_trial or had_gift:
#                         classification = 'winback'
#                     else:
#                         classification = 'winback'
#                 else:
#                     # Gap ≤ 1 jour = renewal automatique
#                     classification = 'renewal'
#             elif last_subscription['status'] == 'active':
#                 # Edge case = unknown
#                 classification = 'unknown'
#                 gap_days = None
#             else:
#                 # Autres cas = renewal
#                 classification = 'renewal'
#                 gap_days = None
#
#         results.append({
#             'index': idx,
#             'customer_id': customer_id,
#             'classification': classification,
#             'previous_status': previous_status,
#             'previous_created': previous_created,  # ← AJOUTÉ
#             'previous_canceled': previous_canceled,
#             'gap_days': gap_days
#         })
#
#     return pd.DataFrame(results)
#
# # %%
#
# def detailed_weekly_analysis(df, weeks_back=1, reference_date=today_date):
#     week_info = get_specific_past_week(weeks_back=weeks_back, reference_date=reference_date)
#     week_start = week_info['week_start']
#     week_end = week_info['week_end']
#
#     print(f'From Monday {week_start} to Sunday {week_end}')
#     print('-------------------------------------------')
#
#     # NEW SUBSCRIPTIONS
#     new_subscriptions = df[
#         (df['created_utc'] >= pd.Timestamp(week_start)) &
#         (df['created_utc'] <= pd.Timestamp(week_end))]
#     len_new_subscriptions = len(new_subscriptions)
#
#     new_trialers = new_subscriptions[new_subscriptions['trial_duration'] > 0]
#     len_new_trialers = len(new_trialers)
#
#     new_gifted = new_subscriptions[new_subscriptions['is_gifted_member'] == True]
#     len_new_gifted = len(new_gifted)
#
#     classification_results = classify_no_trial_subscriptions(df)
#     weekly_no_trial = new_subscriptions[
#         (new_subscriptions['trial_duration'] == 0) & 
#         (~new_subscriptions['is_gifted_member'])
#     ].copy()
#
#     if len(weekly_no_trial) == 0:
#         len_renewals = len_winbacks = len_gift_to_paid = len_unknowns = 0
#         renewal_subscriptions = winback_subscriptions = pd.DataFrame()
#         gift_to_paid_subscriptions = unknown_subscriptions = pd.DataFrame()
#     else:
#         weekly_classified = weekly_no_trial.merge(
#             classification_results, 
#             left_index=True, 
#             right_on='index', 
#             how='left'
#         )
#         weekly_classified['classification'] = weekly_classified['classification'].fillna('unknown')
#
#         # Compter chaque catégorie
#         len_renewals = (weekly_classified['classification'] == 'renewal').sum()
#         len_winbacks = (weekly_classified['classification'] == 'winback').sum()
#         len_gift_to_paid = (weekly_classified['classification'] == 'gift_to_paid_winback').sum()
#         len_unknowns = (weekly_classified['classification'] == 'unknown').sum()
#
#         # Créer les DataFrames
#         renewal_subscriptions = weekly_classified[weekly_classified['classification'] == 'renewal']
#         winback_subscriptions = weekly_classified[weekly_classified['classification'] == 'winback']
#         gift_to_paid_subscriptions = weekly_classified[weekly_classified['classification'] == 'gift_to_paid_winback']
#         unknown_subscriptions = weekly_classified[weekly_classified['classification'] == 'unknown']
#
#     # Autres métriques
#     new_canceled_during_trial = df[
#         (df['canceled_during_trial'] == True) &
#         (df['canceled_at_utc'].notna()) &
#         (df['canceled_at_utc'] >= week_start) &
#         (df['canceled_at_utc'] <= week_end)]
#     len_new_canceled_during_trial = len(new_canceled_during_trial)
#
#     new_canceled_during_refund = df[
#         (df['canceled_during_refund_period'] == True) &
#         (df['canceled_at_utc'].notna()) &
#         (df['canceled_at_utc'] >= pd.Timestamp(week_start)) &
#         (df['canceled_at_utc'] <= pd.Timestamp(week_end))]
#     len_new_canceled_during_refund = len(new_canceled_during_refund)
#
#     new_full_members = df[
#         (df['full_member'] == True) &
#         (df['refund_period_end_utc'] >= week_start) &
#         (df['refund_period_end_utc'] <= week_end)]
#     len_new_full_members = len(new_full_members)
#
#     # AFFICHAGE (TOUJOURS AFFICHÉ)
#     print(f"{len_new_subscriptions} New Subscriptions")
#     print(f"  ├─ {len_new_trialers} New Trialers")
#     print(f"  ├─ {len_new_gifted} Gifted Members (no trial)")
#     print(f"  ├─ {len_renewals} RENEWALS (automatic renewals)")
#     print(f"  ├─ {len_winbacks} WINBACKS (returning customers)")
#     print(f"  ├─ {len_gift_to_paid} Gift-to-Paid Winbacks")
#     print(f"  └─ {len_unknowns} UNKNOWN")
#
#     print(f"{len_new_full_members} Became Full Members (after refund period)")
#     print(f"{len_new_canceled_during_trial} Canceled during trial")
#     print(f"{len_new_canceled_during_refund} Canceled during refund")
#     print('\n')
#
#     # Nettoyage des colonnes
#     def clean_merge_columns(df):
#         if len(df) > 0 and 'customer_id_x' in df.columns:
#             df = df.drop(['customer_id_y', 'index'], axis=1, errors='ignore')
#             df = df.rename(columns={'customer_id_x': 'customer_id'})
#         return df
#
#     renewal_subscriptions = clean_merge_columns(renewal_subscriptions)
#     winback_subscriptions = clean_merge_columns(winback_subscriptions) 
#     gift_to_paid_subscriptions = clean_merge_columns(gift_to_paid_subscriptions)
#     unknown_subscriptions = clean_merge_columns(unknown_subscriptions)
#
#     # DICTIONNAIRE
#     week_dict = {
#         'week_start': week_start, 'week_end': week_end, 'year_week': week_info['year_week'],
#         'weeks_ago': week_info['weeks_ago'], 'new_subscriptions': len_new_subscriptions,
#         'new_trialers': len_new_trialers, 'new_gifted': len_new_gifted,
#         'renewals': len_renewals, 'winbacks': len_winbacks,
#         'gift_to_paid_converters': len_gift_to_paid, 'unknowns': len_unknowns,
#         'new_full_members': len_new_full_members,
#         'new_canceled_during_trial': len_new_canceled_during_trial,
#         'new_canceled_during_refund': len_new_canceled_during_refund
#     }
#
#     # DONNÉES DÉTAILLÉES (TOUJOURS PRÉSENTES)
#     detailed_data = {
#         'renewal_subscriptions': renewal_subscriptions,
#         'winback_subscriptions': winback_subscriptions,
#         'gift_to_paid_subscriptions': gift_to_paid_subscriptions,
#         'unknown_subscriptions': unknown_subscriptions,
#         'new_gifted': new_gifted,
#         'new_trialers': new_trialers
#     }
#
#     return week_dict, detailed_data
#
# # %%
#
# #########################################################################################################################
#
#
#
#
# # %%
#
# def analyze_winback_timing(winback_subscriptions, df):
#     """Analyse les délais de retour des winbacks"""
#
#     if len(winback_subscriptions) == 0:
#         print("Aucun winback cette semaine.")
#         return pd.DataFrame()
#
#     winback_analysis = []
#
#     for idx, winback in winback_subscriptions.iterrows():
#         # CORRECTION : utiliser customer_id au lieu de customer_id_x
#         customer_id = winback['customer_id']  # ← CHANGEMENT ICI
#         previous_created = winback['previous_created']
#         current_created = winback['created_utc']
#
#         # Calculer délai de retour
#         time_away = (current_created - previous_created).days
#
#         winback_analysis.append({
#             'customer_id': customer_id,
#             'time_away_days': time_away,
#             'previous_created': previous_created,
#             'current_created': current_created,
#             'time_away_months': round(time_away / 30.44, 1)  # Approximation mois
#         })
#
#     winback_df = pd.DataFrame(winback_analysis)
#
#     print(f"\n📊 ANALYSE WINBACK TIMING ({len(winback_df)} clients):")
#     print(f"Délai moyen de retour: {winback_df['time_away_days'].mean():.1f} jours ({winback_df['time_away_months'].mean():.1f} mois)")
#     print(f"Délai médian: {winback_df['time_away_days'].median():.0f} jours ({winback_df['time_away_months'].median():.1f} mois)")
#     print(f"Plus rapide: {winback_df['time_away_days'].min()} jours")
#     print(f"Plus lent: {winback_df['time_away_days'].max()} jours ({winback_df['time_away_months'].max():.1f} mois)")
#
#     # Distribution par tranches
#     bins = [0, 30, 90, 180, 365, float('inf')]
#     labels = ['<1 mois', '1-3 mois', '3-6 mois', '6-12 mois', '>1 an']
#     winback_df['time_category'] = pd.cut(winback_df['time_away_days'], bins=bins, labels=labels)
#
#     print(f"\n🗓️ DISTRIBUTION PAR PÉRIODE:")
#     distribution = winback_df['time_category'].value_counts().sort_index()
#     for category, count in distribution.items():
#         percentage = (count / len(winback_df)) * 100
#         print(f"  {category}: {count} clients ({percentage:.1f}%)")
#
#     return winback_df
#
# # Appelez cette fonction après votre analyse :
#
# def investigate_anomalies(anomaly_subscriptions, df):
#     """
#     Analyse approfondie des anomalies (direct subscriptions sans historique)
#     """
#     if len(anomaly_subscriptions) == 0:
#         print("Aucune anomalie cette semaine.")
#         return
#
#     print(f"\n🔍 INVESTIGATION DES {len(anomaly_subscriptions)} ANOMALIES:")
#     print("=" * 60)
#
#     for idx, anomaly in anomaly_subscriptions.iterrows():
#         customer_id = anomaly['customer_id']
#         customer_name = anomaly['customer_name']
#         created_utc = anomaly['created_utc']
#         current_period_duration = anomaly['current_period_duration']
#
#         print(f"\n👤 {customer_name} ({customer_id})")
#         print(f"   📅 Created: {created_utc}")
#         print(f"   💰 Period: {current_period_duration} jours")
#         print(f"   📊 Status: {anomaly['status']}")
#
#         # Vérifier s'il y a eu des erreurs de données
#         customer_all_history = df[df['customer_id'] == customer_id]
#         print(f"   📈 Total subscriptions: {len(customer_all_history)}")
#
#         # Analyser le pattern de nom/email
#         if customer_name:
#             # Chercher des noms similaires
#             similar_names = df[df['customer_name'].str.contains(
#                 customer_name.split()[0] if ' ' in customer_name else customer_name[:5], 
#                 case=False, na=False
#             )]['customer_name'].unique()
#
#             if len(similar_names) > 1:
#                 print(f"   ⚠️  Noms similaires trouvés: {len(similar_names)} ({similar_names[:3]}...)")
#
#         # Vérifier timing suspect
#         created_hour = created_utc.hour
#         if created_hour < 6 or created_hour > 22:
#             print(f"   🕐 Timing suspect: {created_hour}h (hors heures normales)")
#
#         # Comparer avec patterns normaux
#         avg_period = df[(df['trial_duration'] == 0) & (~df['is_gifted_member'])]['current_period_duration'].median()
#         if current_period_duration != avg_period:
#             print(f"   📊 Durée anormale: {current_period_duration} jours (normal: {avg_period})")
#
#     print("\n" + "=" * 60)
#
#     return anomaly_subscriptions
#
# # Appelez cette fonction :
#
#
#
# week_dict, renewals = detailed_weekly_analysis(df, weeks_back=1, reference_date=today_date)
# winback_timing = analyze_winback_timing(renewals['winback_subscriptions'], df)
# investigate_anomalies(renewals['unknown_subscriptions'], df)
#
#
# week_dict, renewals = detailed_weekly_analysis(df, weeks_back=2, reference_date=today_date)
# winback_timing = analyze_winback_timing(renewals['winback_subscriptions'], df)
# investigate_anomalies(renewals['unknown_subscriptions'], df)
#
#
# week_dict, renewals = detailed_weekly_analysis(df, weeks_back=3, reference_date=today_date)
# winback_timing = analyze_winback_timing(renewals['winback_subscriptions'], df)
# investigate_anomalies(renewals['unknown_subscriptions'], df)
#
#
# week_dict, renewals = detailed_weekly_analysis(df, weeks_back=4, reference_date=today_date)
# winback_timing = analyze_winback_timing(renewals['winback_subscriptions'], df)
# investigate_anomalies(renewals['unknown_subscriptions'], df)
#
#
# def weekly_dashboard_summary(weeks_data):
#     """
#     Créer un dashboard comparatif des dernières semaines
#     """
#
#     summary = []
#     for week_data in weeks_data:
#         summary.append({
#             'week': week_data['year_week'],
#             'new_subs': week_data['new_subscriptions'],
#             'trialers': week_data['new_trialers'],
#             'winbacks': week_data.get('winbacks', 0),
#             'canceled_trial': week_data['new_canceled_during_trial'],
#             'canceled_refund': week_data['new_canceled_during_refund'],
#             'conversion_rate': round((week_data['new_trialers'] - week_data['new_canceled_during_trial']) / week_data['new_trialers'] * 100, 1) if week_data['new_trialers'] > 0 else 0,
#             'refund_rate': round(week_data['new_canceled_during_refund'] / week_data['new_subscriptions'] * 100, 1)
#         })
#
#     return pd.DataFrame(summary)
#
#
# def analyze_stripe_date_patterns(df):
#     """
#     Analyse les patterns de dates pour comprendre le comportement de Stripe
#     lors des renouvellements automatiques
#     """
#     print("🔍 ANALYSE DES PATTERNS DE DATES STRIPE")
#     print("=" * 50)
#
#     # Filtrer les subscriptions sans trial et non-gifted
#     no_trial_subs = df[(df['trial_duration'] == 0) & (~df['is_gifted_member'])].copy()
#
#     # Calculer les différences entre les dates clés
#     no_trial_subs['created_to_start_hours'] = (
#         no_trial_subs['start_utc'] - no_trial_subs['created_utc']
#     ).dt.total_seconds() / 3600
#
#     no_trial_subs['created_to_period_start_hours'] = (
#         no_trial_subs['current_period_start_utc'] - no_trial_subs['created_utc']
#     ).dt.total_seconds() / 3600
#
#     no_trial_subs['start_to_period_start_hours'] = (
#         no_trial_subs['current_period_start_utc'] - no_trial_subs['start_utc']
#     ).dt.total_seconds() / 3600
#
#     print(f"📊 ANALYSE SUR {len(no_trial_subs)} SUBSCRIPTIONS SANS TRIAL")
#     print(f"\n🕐 DIFFÉRENCES TEMPORELLES:")
#
#     # Analyse created_utc vs start_utc
#     print(f"Created → Start:")
#     print(f"  - Même moment (0h): {(no_trial_subs['created_to_start_hours'] == 0).sum()}")
#     print(f"  - < 1h: {(no_trial_subs['created_to_start_hours'].abs() < 1).sum()}")
#     print(f"  - < 24h: {(no_trial_subs['created_to_start_hours'].abs() < 24).sum()}")
#     print(f"  - Moyenne: {no_trial_subs['created_to_start_hours'].mean():.2f}h")
#
#     # Analyse created_utc vs current_period_start_utc
#     print(f"\nCreated → Period Start:")
#     print(f"  - Même moment (0h): {(no_trial_subs['created_to_period_start_hours'] == 0).sum()}")
#     print(f"  - < 1h: {(no_trial_subs['created_to_period_start_hours'].abs() < 1).sum()}")
#     print(f"  - < 24h: {(no_trial_subs['created_to_period_start_hours'].abs() < 24).sum()}")
#     print(f"  - Moyenne: {no_trial_subs['created_to_period_start_hours'].mean():.2f}h")
#
#     # Analyse start_utc vs current_period_start_utc
#     print(f"\nStart → Period Start:")
#     print(f"  - Même moment (0h): {(no_trial_subs['start_to_period_start_hours'] == 0).sum()}")
#     print(f"  - < 1h: {(no_trial_subs['start_to_period_start_hours'].abs() < 1).sum()}")
#     print(f"  - < 24h: {(no_trial_subs['start_to_period_start_hours'].abs() < 24).sum()}")
#     print(f"  - Moyenne: {no_trial_subs['start_to_period_start_hours'].mean():.2f}h")
#
#     # Chercher des patterns suspects pour les renouvellements
#     df_sorted = df.sort_values(['customer_id', 'created_utc'])
#
#     potential_renewals = []
#     for idx, row in no_trial_subs.iterrows():
#         customer_id = row['customer_id']
#         current_created = row['created_utc']
#
#         # Historique du client
#         customer_history = df_sorted[
#             (df_sorted['customer_id'] == customer_id) & 
#             (df_sorted['created_utc'] < current_created)
#         ]
#
#         if len(customer_history) > 0:
#             # Pattern suspect : created_utc très proche de current_period_start_utc
#             # mais le client a un historique
#             created_to_period_hours = abs(row['created_to_period_start_hours'])
#
#             if created_to_period_hours < 24:  # Moins de 24h de différence
#                 last_sub = customer_history.iloc[-1]
#
#                 potential_renewals.append({
#                     'customer_id': customer_id,
#                     'current_created': current_created,
#                     'current_period_start': row['current_period_start_utc'],
#                     'hours_diff': created_to_period_hours,
#                     'last_sub_created': last_sub['created_utc'],
#                     'last_sub_canceled': last_sub['canceled_at_utc'],
#                     'last_sub_ended': last_sub['ended_at_utc']
#                 })
#
#     print(f"\n🚨 RENOUVELLEMENTS POTENTIELS DÉTECTÉS: {len(potential_renewals)}")
#
#     if len(potential_renewals) > 0:
#         renewal_df = pd.DataFrame(potential_renewals)
#         print(f"Répartition par écart temporel:")
#         print(f"  - < 1h: {(renewal_df['hours_diff'] < 1).sum()}")
#         print(f"  - 1-6h: {((renewal_df['hours_diff'] >= 1) & (renewal_df['hours_diff'] < 6)).sum()}")
#         print(f"  - 6-24h: {((renewal_df['hours_diff'] >= 6) & (renewal_df['hours_diff'] < 24)).sum()}")
#
#         # Afficher quelques exemples
#         print(f"\n📋 EXEMPLES (5 premiers):")
#         for i, row in renewal_df.head().iterrows():
#             print(f"  Client {row['customer_id']}: {row['hours_diff']:.1f}h d'écart")
#
#     return no_trial_subs, potential_renewals
# analyze_stripe_date_patterns(df)
#
#
#
# def analyze_stripe_date_patterns(df):
#     """
#     Analyse les patterns de dates pour comprendre le comportement de Stripe
#     lors des renouvellements automatiques
#     """
#     print("🔍 ANALYSE DES PATTERNS DE DATES STRIPE")
#     print("=" * 50)
#
#     # Filtrer les subscriptions sans trial et non-gifted
#     no_trial_subs = df[(df['trial_duration'] == 0) & (~df['is_gifted_member'])].copy()
#
#     # Calculer les différences entre les dates clés
#     no_trial_subs['created_to_start_hours'] = (
#         no_trial_subs['start_utc'] - no_trial_subs['created_utc']
#     ).dt.total_seconds() / 3600
#
#     no_trial_subs['created_to_period_start_hours'] = (
#         no_trial_subs['current_period_start_utc'] - no_trial_subs['created_utc']
#     ).dt.total_seconds() / 3600
#
#     no_trial_subs['start_to_period_start_hours'] = (
#         no_trial_subs['current_period_start_utc'] - no_trial_subs['start_utc']
#     ).dt.total_seconds() / 3600
#
#     print(f"📊 ANALYSE SUR {len(no_trial_subs)} SUBSCRIPTIONS SANS TRIAL")
#     print(f"\n🕐 DIFFÉRENCES TEMPORELLES:")
#
#     # Analyse created_utc vs start_utc
#     print(f"Created → Start:")
#     print(f"  - Même moment (0h): {(no_trial_subs['created_to_start_hours'] == 0).sum()}")
#     print(f"  - < 1h: {(no_trial_subs['created_to_start_hours'].abs() < 1).sum()}")
#     print(f"  - < 8h: {(no_trial_subs['created_to_start_hours'].abs() < 8).sum()}")
#     print(f"  - Moyenne: {no_trial_subs['created_to_start_hours'].mean():.2f}h")
#
#     # Analyse created_utc vs current_period_start_utc
#     print(f"\nCreated → Period Start:")
#     print(f"  - Même moment (0h): {(no_trial_subs['created_to_period_start_hours'] == 0).sum()}")
#     print(f"  - < 1h: {(no_trial_subs['created_to_period_start_hours'].abs() < 1).sum()}")
#     print(f"  - < 8h: {(no_trial_subs['created_to_period_start_hours'].abs() < 8).sum()}")
#     print(f"  - Moyenne: {no_trial_subs['created_to_period_start_hours'].mean():.2f}h")
#
#     # Analyse start_utc vs current_period_start_utc
#     print(f"\nStart → Period Start:")
#     print(f"  - Même moment (0h): {(no_trial_subs['start_to_period_start_hours'] == 0).sum()}")
#     print(f"  - < 1h: {(no_trial_subs['start_to_period_start_hours'].abs() < 1).sum()}")
#     print(f"  - < 8h: {(no_trial_subs['start_to_period_start_hours'].abs() < 8).sum()}")
#     print(f"  - Moyenne: {no_trial_subs['start_to_period_start_hours'].mean():.2f}h")
#
#     # Chercher des patterns suspects pour les renouvellements
#     df_sorted = df.sort_values(['customer_id', 'created_utc'])
#
#     potential_renewals = []
#     for idx, row in no_trial_subs.iterrows():
#         customer_id = row['customer_id']
#         current_created = row['created_utc']
#
#         # Historique du client
#         customer_history = df_sorted[
#             (df_sorted['customer_id'] == customer_id) & 
#             (df_sorted['created_utc'] < current_created)
#         ]
#
#         if len(customer_history) > 0:
#             # Pattern suspect : created_utc très proche de current_period_start_utc
#             # mais le client a un historique
#             created_to_period_hours = abs(row['created_to_period_start_hours'])
#
#             if created_to_period_hours < 8:  # Moins de 8h de différence
#                 last_sub = customer_history.iloc[-1]
#
#                 potential_renewals.append({
#                     'customer_id': customer_id,
#                     'current_created': current_created,
#                     'current_period_start': row['current_period_start_utc'],
#                     'hours_diff': created_to_period_hours,
#                     'last_sub_created': last_sub['created_utc'],
#                     'last_sub_canceled': last_sub['canceled_at_utc'],
#                     'last_sub_ended': last_sub['ended_at_utc']
#                 })
#
#     print(f"\n🚨 RENOUVELLEMENTS POTENTIELS DÉTECTÉS: {len(potential_renewals)}")
#
#     if len(potential_renewals) > 0:
#         renewal_df = pd.DataFrame(potential_renewals)
#         print(f"Répartition par écart temporel:")
#         print(f"  - < 1h: {(renewal_df['hours_diff'] < 1).sum()}")
#         print(f"  - 1-8h: {((renewal_df['hours_diff'] >= 1) & (renewal_df['hours_diff'] < 8)).sum()}")
#         print(f"  - 8h+: {(renewal_df['hours_diff'] >= 8).sum()}")
#
#         # Afficher quelques exemples
#         print(f"\n📋 EXEMPLES (5 premiers):")
#         for i, row in renewal_df.head().iterrows():
#             print(f"  Client {row['customer_id']}: {row['hours_diff']:.1f}h d'écart")
#
#     return no_trial_subs, potential_renewals
#
#
# def classify_no_trial_subscriptions(df):
#     """
#     Nouvelle logique de classification avec les règles mises à jour:
#     1. Gift-to-Pay: A eu au moins 1 gift + subscription actuelle sans trial + pas gifted actuellement
#     2. Renewal: Gap ≤ 1 jour OU détection via patterns Stripe
#     3. Winback: Gap > 1 jour (peu importe si précédent était trial-only)
#     4. Edge Case/Unknown: Tout le reste
#     """
#
#     df_sorted = df.sort_values(['customer_id', 'created_utc'])
#     no_trial_mask = (df['trial_duration'] == 0) & (~df['is_gifted_member'])
#
#     results = []
#
#     for idx, row in df[no_trial_mask].iterrows():
#         customer_id = row['customer_id']
#         current_created = row['created_utc']
#         current_period_start = row['current_period_start_utc']
#
#         # Historique du client
#         customer_history = df_sorted[
#             (df_sorted['customer_id'] == customer_id) & 
#             (df_sorted['created_utc'] < current_created)
#         ]
#
#         # Détection Stripe renewal pattern
#         created_to_period_hours = abs((current_period_start - current_created).total_seconds() / 3600)
#         potential_stripe_renewal = (len(customer_history) > 0) and (created_to_period_hours <= 8)
#
#         if len(customer_history) == 0:
#             # Aucun historique = première subscription
#             classification = 'unknown'
#             previous_status = None
#             previous_created = None
#             previous_canceled = None
#             gap_days = None
#             stripe_renewal_detected = False
#
#         else:
#             last_subscription = customer_history.iloc[-1]
#             previous_status = last_subscription['status']
#             previous_created = last_subscription['created_utc']
#             previous_canceled = last_subscription['canceled_at_utc']
#             stripe_renewal_detected = potential_stripe_renewal
#
#             # 1. GIFT-TO-PAY : A eu au moins 1 gift dans l'historique
#             had_gift = customer_history['is_gifted_member'].any()
#
#             if had_gift:
#                 classification = 'gift_to_paid_winback'
#                 gap_days = None if pd.isna(previous_canceled) else (current_created - previous_canceled).days
#
#             # 2. RENEWAL : Gap ≤ 1 jour OU pattern Stripe détecté
#             elif (pd.notna(previous_canceled) and 
#                   (current_created - previous_canceled).days <= 1) or potential_stripe_renewal:
#                 classification = 'renewal'
#                 gap_days = None if pd.isna(previous_canceled) else (current_created - previous_canceled).days
#
#             # 3. WINBACK : Gap > 1 jour
#             elif (pd.notna(previous_canceled) and 
#                   (current_created - previous_canceled).days > 1):
#                 classification = 'winback'
#                 gap_days = (current_created - previous_canceled).days
#
#             # 4. EDGE CASE/UNKNOWN
#             else:
#                 classification = 'unknown'
#                 gap_days = None
#
#         results.append({
#             'index': idx,
#             'customer_id': customer_id,
#             'classification': classification,
#             'previous_status': previous_status,
#             'previous_created': previous_created,
#             'previous_canceled': previous_canceled,
#             'gap_days': gap_days,
#             'stripe_renewal_detected': stripe_renewal_detected,
#             'created_to_period_hours': created_to_period_hours
#         })
#
#     return pd.DataFrame(results)
#
#
# def detailed_weekly_analysis(df, weeks_back=1, reference_date=None):
#     """
#     Version mise à jour de l'analyse hebdomadaire avec la nouvelle logique
#     """
#     if reference_date is None:
#         reference_date = pd.Timestamp.now(tz='UTC')
#
#     week_info = get_specific_past_week(weeks_back=weeks_back, reference_date=reference_date)
#     week_start = week_info['week_start']
#     week_end = week_info['week_end']
#
#     print(f'🗓️  SEMAINE {week_info["year_week"]} (il y a {weeks_back} semaine{"s" if weeks_back > 1 else ""})')
#     print(f'Du lundi {week_start.strftime("%d/%m")} au dimanche {week_end.strftime("%d/%m/%Y")}')
#     print('=' * 60)
#
#     # NOUVELLES SUBSCRIPTIONS DE LA SEMAINE
#     new_subscriptions = df[
#         (df['created_utc'] >= week_start) &
#         (df['created_utc'] <= week_end)
#     ]
#     len_new_subscriptions = len(new_subscriptions)
#
#     # TRIALERS
#     new_trialers = new_subscriptions[new_subscriptions['trial_duration'] > 0]
#     len_new_trialers = len(new_trialers)
#
#     # GIFTED
#     new_gifted = new_subscriptions[new_subscriptions['is_gifted_member'] == True]
#     len_new_gifted = len(new_gifted)
#
#     # CLASSIFICATION DES SUBSCRIPTIONS SANS TRIAL
#     classification_results = classify_no_trial_subscriptions(df)
#     weekly_no_trial = new_subscriptions[
#         (new_subscriptions['trial_duration'] == 0) & 
#         (~new_subscriptions['is_gifted_member'])
#     ].copy()
#
#     if len(weekly_no_trial) == 0:
#         len_renewals = len_winbacks = len_gift_to_paid = len_unknowns = 0
#         len_stripe_renewals = 0
#         renewal_subscriptions = winback_subscriptions = pd.DataFrame()
#         gift_to_paid_subscriptions = unknown_subscriptions = pd.DataFrame()
#     else:
#         weekly_classified = weekly_no_trial.merge(
#             classification_results, 
#             left_index=True, 
#             right_on='index', 
#             how='left'
#         )
#         weekly_classified['classification'] = weekly_classified['classification'].fillna('unknown')
#
#         # Compter chaque catégorie
#         len_renewals = (weekly_classified['classification'] == 'renewal').sum()
#         len_winbacks = (weekly_classified['classification'] == 'winback').sum()
#         len_gift_to_paid = (weekly_classified['classification'] == 'gift_to_paid_winback').sum()
#         len_unknowns = (weekly_classified['classification'] == 'unknown').sum()
#         len_stripe_renewals = (weekly_classified['stripe_renewal_detected'] == True).sum()
#
#         # Créer les DataFrames
#         renewal_subscriptions = weekly_classified[weekly_classified['classification'] == 'renewal']
#         winback_subscriptions = weekly_classified[weekly_classified['classification'] == 'winback']
#         gift_to_paid_subscriptions = weekly_classified[weekly_classified['classification'] == 'gift_to_paid_winback']
#         unknown_subscriptions = weekly_classified[weekly_classified['classification'] == 'unknown']
#
#     # AUTRES MÉTRIQUES
#     new_canceled_during_trial = df[
#         (df['canceled_during_trial'] == True) &
#         (df['canceled_at_utc'].notna()) &
#         (df['canceled_at_utc'] >= week_start) &
#         (df['canceled_at_utc'] <= week_end)
#     ]
#     len_new_canceled_during_trial = len(new_canceled_during_trial)
#
#     new_canceled_during_refund = df[
#         (df['canceled_during_refund_period'] == True) &
#         (df['canceled_at_utc'].notna()) &
#         (df['canceled_at_utc'] >= week_start) &
#         (df['canceled_at_utc'] <= week_end)
#     ]
#     len_new_canceled_during_refund = len(new_canceled_during_refund)
#
#     new_full_members = df[
#         (df['full_member'] == True) &
#         (df['refund_period_end_utc'] >= week_start) &
#         (df['refund_period_end_utc'] <= week_end)
#     ]
#     len_new_full_members = len(new_full_members)
#
#     # AFFICHAGE
#     print(f"📈 {len_new_subscriptions} NOUVELLES SUBSCRIPTIONS")
#     print(f"  ├─ 🎯 {len_new_trialers} Nouveaux Trialers")
#     print(f"  ├─ 🎁 {len_new_gifted} Membres Offerts")
#     print(f"  ├─ 🔄 {len_renewals} RENEWALS (renouvellements automatiques)")
#     if len_stripe_renewals > 0:
#         print(f"  │   └─ 🤖 {len_stripe_renewals} détectés via pattern Stripe")
#     print(f"  ├─ 🔙 {len_winbacks} WINBACKS (clients de retour)")
#     print(f"  ├─ 💝 {len_gift_to_paid} Gift-to-Paid (ex-cadeaux → payants)")
#     print(f"  └─ ❓ {len_unknowns} INCONNUS")
#     print()
#     print(f"✅ {len_new_full_members} Devenus Membres Complets")
#     print(f"❌ {len_new_canceled_during_trial} Annulations pendant trial")
#     print(f"💔 {len_new_canceled_during_refund} Annulations pendant remboursement")
#     print()
#
#     # Nettoyage des colonnes pour les DataFrames retournés
#     def clean_merge_columns(df):
#         if len(df) > 0 and 'customer_id_x' in df.columns:
#             df = df.drop(['customer_id_y', 'index'], axis=1, errors='ignore')
#             df = df.rename(columns={'customer_id_x': 'customer_id'})
#         return df
#
#     renewal_subscriptions = clean_merge_columns(renewal_subscriptions)
#     winback_subscriptions = clean_merge_columns(winback_subscriptions) 
#     gift_to_paid_subscriptions = clean_merge_columns(gift_to_paid_subscriptions)
#     unknown_subscriptions = clean_merge_columns(unknown_subscriptions)
#
#     # DICTIONNAIRE RÉSUMÉ
#     week_dict = {
#         'week_start': week_start, 'week_end': week_end, 'year_week': week_info['year_week'],
#         'weeks_ago': week_info['weeks_ago'], 'new_subscriptions': len_new_subscriptions,
#         'new_trialers': len_new_trialers, 'new_gifted': len_new_gifted,
#         'renewals': len_renewals, 'winbacks': len_winbacks,
#         'gift_to_paid_converters': len_gift_to_paid, 'unknowns': len_unknowns,
#         'stripe_renewals_detected': len_stripe_renewals,
#         'new_full_members': len_new_full_members,
#         'new_canceled_during_trial': len_new_canceled_during_trial,
#         'new_canceled_during_refund': len_new_canceled_during_refund
#     }
#
#     # DONNÉES DÉTAILLÉES
#     detailed_data = {
#         'renewal_subscriptions': renewal_subscriptions,
#         'winback_subscriptions': winback_subscriptions,
#         'gift_to_paid_subscriptions': gift_to_paid_subscriptions,
#         'unknown_subscriptions': unknown_subscriptions,
#         'new_gifted': new_gifted,
#         'new_trialers': new_trialers
#     }
#
#     return week_dict, detailed_data
#
#
# # FONCTION D'UTILISATION
# def test_new_classification(df, reference_date):
#     """
#     Teste la nouvelle classification et compare avec l'ancienne si nécessaire
#     """
#     print("🧪 TEST DE LA NOUVELLE CLASSIFICATION")
#     print("=" * 50)
#
#     # 1. Analyse des patterns de dates
#     date_analysis, potential_renewals = analyze_stripe_date_patterns(df)
#
#     print("\n" + "=" * 50)
#
#     # 2. Test sur les 4 dernières semaines
#     for week in range(1, 5):
#         week_dict, detailed_data = detailed_weekly_analysis(df, weeks_back=week, reference_date=reference_date)
#
#         # Analyse rapide des winbacks s'il y en a
#         if len(detailed_data['winback_subscriptions']) > 0:
#             winback_timing = analyze_winback_timing(detailed_data['winback_subscriptions'], df)
#
#     return date_analysis, potential_renewals
