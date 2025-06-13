# %# %%
######################################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from datetime import datetime

# %%
######################################################################################
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
######################################################################################
# Set TODAY DATE
# today_date = pd.Timestamp.now(tz='UTC')
today_date = pd.Timestamp('2025-05-23', tz='UTC')  # For testing purposesv


# Set REFUND PERDIOD DURATION
REFUND_PERIOD_DAYS = 14  # Duration of the refund period in days

# Set thresholds for cleaning
HIGH_VOLUME_THRESHOLD = 5
DUPLICATE_THRESHOLD_MINUTES = 15


# Set DIRECTORIES
data_dir = 'both_csv_go_here'
archive_csv_dir = 'archive/csv'
archive_png_dir = 'archive/analysis'
analysis_dir = 'analysis'


# %%
######################################################################################
# LOADING CSV

# Toggle this flag to True in production
RENAME_FILES = False 
MOVE_FILES = False

# Ensure archive directory exists
os.makedirs(archive_csv_dir, exist_ok=True)


# List and sort files by creation time
files = [
    os.path.join(data_dir, f)
    for f in os.listdir(data_dir)
    if os.path.isfile(os.path.join(data_dir, f)) and f.endswith('.csv')]
sorted_files = sorted(files, key=os.path.getctime, reverse=True)

# Check if we have exactly 2 CSV files
if len(sorted_files) != 2:
    print(f"Error: Expected 2 CSV files, found {len(sorted_files)}")
    print("Files found:", [os.path.basename(f) for f in sorted_files])
    exit(1)

for i, file_path in enumerate(sorted_files, 1):
    print(f"  File {i}:\n {os.path.basename(file_path)}")

# Loop over files
processed_files = []
for file_path in sorted_files:
    created_at = datetime.fromtimestamp(os.path.getctime(file_path))
    timestamp_str = created_at.strftime('%Y-%m-%d_%H-%M')
    original_name = os.path.basename(file_path)
    new_name = f"{timestamp_str}_{original_name}"
    
    if RENAME_FILES:
        if not original_name.startswith(timestamp_str):
            new_path = os.path.join(data_dir, new_name)
            os.rename(file_path, new_path)
            print(f"Renamed:\n {original_name} →\n {new_name}\n")
            processed_files.append(new_path)
        else:
            processed_files.append(file_path)
    else:
        processed_files.append(file_path)

# Load both CSV files into pandas DataFrames
file1_path, file2_path = processed_files[0], processed_files[1]
print(f"\nLoading CSV files:")
print(f"  File 1: {os.path.basename(file1_path)}")
print(f"  File 2: {os.path.basename(file2_path)}")

try:
    df1_raw = pd.read_csv(file1_path, low_memory=False)
    df2_raw = pd.read_csv(file2_path, low_memory=False)
    print(f"\nSuccessfully loaded:")
    print(f"  df1_raw: {df1_raw.shape[0]} rows, {df1_raw.shape[1]} columns")
    print(f"  df2_raw: {df2_raw.shape[0]} rows, {df2_raw.shape[1]} columns")
except Exception as e:
    print(f"Error loading CSV files: {e}")
    exit(1)

# Move files to archive
if MOVE_FILES:
    for file_path in processed_files:
        file_name = os.path.basename(file_path)
        archive_path = os.path.join(archive_csv_dir, file_name)
        
        if not os.path.exists(archive_path):
            os.rename(file_path, archive_path)
            print(f"Moved: {file_name} to archive")
        else:
            print(f"Already archived: {file_name}")
else:
    for file_path in processed_files:
        file_name = os.path.basename(file_path)

print("\nDataFrames available as: df1_raw, df2_raw")
print("\nProcessing complete!")
print('***************************************************')


# %%
######################################################################################
# DATA PREPROCESSING (customer df)
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
        'id': 'subscription_id',
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


    # Reference date for analysis
    print(f"📅 Reference date (TODAY) : {today_date.strftime('%d-%m-%Y')}")
    print(f"{len(df)} entries loaded from {file_path}")
    print('***************************************************')

    return df

df1 = preprocess_data(df1_raw)


# %%
######################################################################################
# DATA PREPROCESSING (invoices df)

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
        'id', 'Customer Name', 'Customer ID',
    ]
    
    df = df[columns_to_keep]

    df.rename(columns={
        'id': 'subscription_id',
        'Customer ID': 'customer_id',
        'Customer Name': 'customer_name',
    }, inplace=True)

    return df



df2 = preprocess_data(df2_raw)

# %%
######################################################################################
# MERGING DATAFRAMES
def merge_dataframes(df1, df2):
    """Merge two DataFrames on 'subscription_id' and 'customer_id'"""
    # Ensure both DataFrames have the same columns for merging

    merged_df = pd.merge(df1, df2, on=['id', 'customer_id'], how='outer')


    return merged_df

df = merge_dataframes(df1, df2)


# %%
######################################################################################
# Removing customers with more than 5 subscriptions (Probably testing accounts)
def remove_high_volume_customers(df, threshold=HIGH_VOLUME_THRESHOLD):
    """Remove customers with more than a specified number of subscriptions"""
    
    original_count = len(df)

    customer_counts = df['customer_id'].value_counts()
    high_volume_customers = customer_counts[customer_counts > threshold].index
    
    df = df[~df['customer_id'].isin(high_volume_customers)]
    
    print(f'{original_count - len(df)} subscriptions removed from \
{len(high_volume_customers)} customers with more than {threshold} subscriptions')
    print('***************************************************')

    return df


df = remove_high_volume_customers(df)


# %%
######################################################################################
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


# %%
######################################################################################
# SETTING REFUND PERIOD END UTC
def refund_period_end_utc(df, REFUND_PERIOD_DAYS):
    df['refund_period_end_utc'] = np.where(
        df['trial_end_utc'].notna(), df['trial_end_utc'] + pd.Timedelta(days=REFUND_PERIOD_DAYS),
        df['current_period_start_utc'] + pd.Timedelta(days=REFUND_PERIOD_DAYS))

    return df

df = refund_period_end_utc(df, REFUND_PERIOD_DAYS)


# %%
######################################################################################
# CANCEL DURRING REFUND PERIOD
def canceled_during_refund_period(df):
    """Check if a member canceled during their refund period"""
    # Ensure columns are in datetime format
    df['canceled_during_refund_period'] = (
        (df['canceled_during_trial'] == False) &
        (df['canceled_at_utc'].notna()) & 
        (df['refund_period_end_utc'] > df['canceled_at_utc']) 
    )
    return df

df = canceled_during_refund_period(df)


# %%
#####################################################################################
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


    df['is_full_member'] = (
        no_early_cancellation & 
        not_gifted & 
        refund_period_passed
    )
    
    return df

df = full_member_status(df)


# %%
######################################################################################
# PAYINF MEMBERS
def paying_members(df):
    """Determine if a customer is a paying member"""
    # Paying member if:
    # 1. Not canceled
    # 2. Not gifted

    no_early_cancellation = (
        (~df['canceled_during_trial']) & 
        (~df['canceled_during_refund_period'])
    )

    not_gifted = (~df['is_gifted_member'])


    df['is_paying_member'] = (
        no_early_cancellation & 
        not_gifted
    )
    
    return df

df = paying_members(df)


# %%
######################################################################################
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


