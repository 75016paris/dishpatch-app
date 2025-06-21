# %%
######################################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from datetime import datetime
import shutil
import glob


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
archive_png_dir = 'archive/analysis/png'
archive_pdf_dir = 'archive/analysis/pdf'
analysis_dir = 'analysis'

# Transfer pdf from analysis_dir to archive_pdf_dir

# Transfer png from analysis_dir to archive_png_dir



# %%
######################################################################################
def get_file_creation_date(file_path):
    """
    Get the creation date of a file and return it as a formatted string
    Returns format: YYYY-MM-DD
    """
    try:
        # Get file creation time (or modification time if creation not available)
        if os.name == 'nt':  # Windows
            creation_time = os.path.getctime(file_path)
        else:  # Unix/Linux/Mac
            creation_time = os.path.getmtime(file_path)
        
        # Convert to datetime and format
        creation_date = datetime.fromtimestamp(creation_time)
        return creation_date.strftime('%Y-%m-%d')
    
    except Exception as e:
        print(f"❌ Error getting creation date for {file_path}: {e}")
        # Fallback to today's date
        return datetime.now().strftime('%Y-%m-%d')


def transfer_files_to_archive():
    """
    Enhanced version with date-based organization
    Transfer PNG files from analysis_dir to archive_png_dir/YYYY-MM-DD/
    Transfer PDF files from analysis_dir to archive_pdf_dir/YYYY-MM-DD/
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # === TRANSFER PNG FILES ===
    png_files = glob.glob(os.path.join(analysis_dir, "*.png"))
    png_transferred = 0
    
    for png_file in png_files:
        filename = os.path.basename(png_file)
        
        # Get creation date for organization
        creation_date = get_file_creation_date(png_file)
        
        # Create date-based directory in archive
        date_archive_dir = os.path.join(archive_png_dir, creation_date)
        os.makedirs(date_archive_dir, exist_ok=True)
        
        # Set destination with date organization
        destination = os.path.join(date_archive_dir, filename)
        
        try:
            # Copy file to archive (keep original in analysis_dir)
            shutil.copy2(png_file, destination)
            print(f"📊 PNG archived: {creation_date}/{filename}")
            png_transferred += 1
        except Exception as e:
            print(f"❌ Error archiving PNG {filename}: {e}")
    
    # === TRANSFER PDF FILES ===
    pdf_files = glob.glob(os.path.join(analysis_dir, "*.pdf"))
    pdf_transferred = 0
    
    for pdf_file in pdf_files:
        filename = os.path.basename(pdf_file)
        
        # Get creation date for organization
        creation_date = get_file_creation_date(pdf_file)
        
        # Create date-based directory in archive
        date_archive_dir = os.path.join(archive_pdf_dir, creation_date)
        os.makedirs(date_archive_dir, exist_ok=True)
        
        # Set destination with date organization
        destination = os.path.join(date_archive_dir, filename)
        
        try:
            # Copy file to archive (keep original in analysis_dir)
            shutil.copy2(pdf_file, destination)
            print(f"📄 PDF archived: {creation_date}/{filename}")
            pdf_transferred += 1
        except Exception as e:
            print(f"❌ Error archiving PDF {filename}: {e}")
    
    # === SUMMARY ===
    print(f"\n📦 ARCHIVING SUMMARY ({timestamp}):")
    print(f"   PNG files transferred: {png_transferred}")
    print(f"   PDF files transferred: {pdf_transferred}")
    print(f"   Total files archived: {png_transferred + pdf_transferred}")
    
    return png_transferred, pdf_transferred


def clean_analysis_dir_after_archive():
    """
    OPTIONAL: Remove files from analysis_dir after successful archiving
    USE WITH CAUTION - This will delete the original files!
    Enhanced with better logging and date information
    """
    # Get all PNG and PDF files in analysis_dir
    png_files = glob.glob(os.path.join(analysis_dir, "*.png"))
    pdf_files = glob.glob(os.path.join(analysis_dir, "*.pdf"))
    all_files = png_files + pdf_files
    
    if not all_files:
        print("🗑️  No files to clean in analysis directory")
        return 0
    
    print(f"🗑️  Cleaning {len(all_files)} files from {analysis_dir}...")
    
    cleaned_files = 0
    
    for file_path in all_files:
        try:
            filename = os.path.basename(file_path)
            creation_date = get_file_creation_date(file_path)
            
            os.remove(file_path)
            print(f"🗑️  Cleaned: {filename} (was from {creation_date})")
            cleaned_files += 1
            
        except Exception as e:
            print(f"❌ Error cleaning {file_path}: {e}")
    
    print(f"🧹 Cleanup complete: {cleaned_files} files removed from {analysis_dir}")
    return cleaned_files


transfer_files_to_archive()
clean_analysis_dir_after_archive()


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

def preprocess_data_invoice(input_df):
    """Clean and preprocess the subscription data"""
    df = input_df.copy()

    # Date conversion
    date_cols = [col for col in df.columns if '(UTC)' in col]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)

    df = df.sort_values(by='Created (UTC)')

    # Column selection and renaming
    columns_to_keep = [
        'id', 'Customer Name', 'Customer', 'Amount Due', 'Amount Paid', 'Paid', 'Billing', 'Charge', 'Closed',
        'Date (UTC)', 'Description', 'Number', 'Finalized At (UTC)',
        'Paid At (UTC)', 'Minimum Line Item Period Start (UTC)', 'Maximum Line Item Period End (UTC)',
        'Period End (UTC)', 'Subscription', 'Total Discount Amount', 'Applied Coupons', 'Status' 
        ]
    
    df = df[columns_to_keep]

    df.rename(columns={
        'id': 'invoice_id',
        'Customer': 'customer_id',
        'Customer Name': 'customer_name',

    }, inplace=True)

    return df



#df2 = preprocess_data_invoice(df2_raw)
df2_raw.info()
df2_raw['Status'].value_counts()
# df2_raw['Total Discount Amount'].value_counts()

#
# time_dis = df2_raw[df2_raw['Minimum Line Item Period Start (UTC)'] != df2_raw['Maximum Line Item Period End (UTC)']]
# len(time_dis)

# # %%
# ######################################################################################
# # MERGING DATAFRAMES
# def merge_dataframes(df1, df2):
#     """Merge two DataFrames on 'subscription_id' and 'customer_id'"""
#     # Ensure both DataFrames have the same columns for merging
#
#     merged_df = pd.merge(df1, df2, on=['id', 'customer_id'], how='outer')
#
#
#     return merged_df
#
# df = merge_dataframes(df1, df2)
#
# # %%
# ######################################################################################
# # Removing customers with more than 5 subscriptions (Probably testing accounts)
# def remove_high_volume_customers(df, threshold=HIGH_VOLUME_THRESHOLD):
#     """Remove customers with more than a specified number of subscriptions"""
#     df = df.copy()
#
#     original_count = len(df)
#
#     customer_counts = df['customer_id'].value_counts()
#     high_volume_customers = customer_counts[customer_counts > threshold].index
#
#     df = df[~df['customer_id'].isin(high_volume_customers)]
#
#     print(f'{original_count - len(df)} subscriptions removed from \
# {len(high_volume_customers)} customers with more than {threshold} subscriptions')
#     print('***************************************************')
#
#     return df
#
#
# df = remove_high_volume_customers(df)
#
#
# # %%
# ######################################################################################
# # CANCEL DURING TRIAL PERIOD
# def cancel_during_trial(df):
#     """Check if a member canceled during their trial period"""
#     df =df.copy()
#
#     df['canceled_during_trial'] = (
#         (df['canceled_at_utc'].notna()) & 
#         (df['trial_end_utc'] > df['canceled_at_utc']) 
#     )
#     return df
#
# df = cancel_during_trial(df) 
#
#
# # %%
# ######################################################################################
# # SETTING REFUND PERIOD END UTC
# def refund_period_end_utc(df, REFUND_PERIOD_DAYS):
#     df = df.copy()
#
#     df['refund_period_end_utc'] = np.where(
#         df['trial_end_utc'].notna(), df['trial_end_utc'] + pd.Timedelta(days=REFUND_PERIOD_DAYS),
#         df['current_period_start_utc'] + pd.Timedelta(days=REFUND_PERIOD_DAYS))
#
#     return df
#
# df = refund_period_end_utc(df, REFUND_PERIOD_DAYS)
#
#
# # %%
# ######################################################################################
# # CANCEL DURRING REFUND PERIOD
# def canceled_during_refund_period(df):
#     """Check if a member canceled during their refund period"""
#     df = df.copy()
#
#     df['canceled_during_refund_period'] = (
#         (df['canceled_during_trial'] == False) &
#         (df['canceled_at_utc'].notna()) & 
#         (df['refund_period_end_utc'] > df['canceled_at_utc']) 
#     )
#     return df
#
# df = canceled_during_refund_period(df)
#
#
# # %%
# #####################################################################################
# # FULL MEMBER STATUS
# def full_member_status(df):
#     """Determine if a customer is a full member based on business logic"""
#     df = df.copy()
#
#     # Full member if:
#     # 1. Not canceled during trial
#     # 2. Not canceled during refund period
#     # 3. Not gifted
#     # 4. Trial ended more than 14 days ago (if no trial, current_period_start_utc > 14 days ago)
#
#     no_early_cancellation = (
#         (~df['canceled_during_trial']) & 
#         (~df['canceled_during_refund_period'])
#     )
#
#     not_gifted = (~df['is_gifted_member'])
#
#     refund_period_passed = (
#             (today_date > df['refund_period_end_utc']) 
#             )
#
#     df['is_full_member'] = (
#         no_early_cancellation & 
#         not_gifted & 
#         refund_period_passed
#     )
#
#     return df
#
# df = full_member_status(df)
#
#
# # %%
# ######################################################################################
# # PAYING MEMBERS
# def paying_members(df):
#     """Determine if a customer is a paying member"""
#     df = df.copy()
#
#     # Paying member if:
#     # 1. Not canceled
#     # 2. Not gifted
#
#     no_early_cancellation = (
#         (~df['canceled_during_trial']) & 
#         (~df['canceled_during_refund_period'])
#     )
#
#     not_gifted = (~df['is_gifted_member'])
#
#
#     df['is_paying_member'] = (
#         no_early_cancellation & 
#         not_gifted
#     )
#
#     return df
#
# df = paying_members(df)
#
# # %%
# ######################################################################################
# # add ended_at_utc when needed
# def add_ended_at_utc(df):
#     """add ended_at_utc when needed"""
#     df = df.copy()
#     # if canceled during trial, set ended_at_utc to trial_end_utc
#     df['ended_at_utc'] = np.where(
#         df['canceled_during_trial'], 
#         df['trial_end_utc'],  # Ils gardent accès jusqu'à fin trial
#         df['ended_at_utc']
#     )
#
#     # if canceled during refund period, set ended_at_utc to canceled_at_utc
#     df['ended_at_utc'] = np.where(
#         (df['canceled_during_refund_period']) &
#         (~df['canceled_during_trial']),  # Ajout de cette condition
#         df['canceled_at_utc'],
#         df['ended_at_utc']
#     )
#
#     # if canceled after refund period, set ended_at_utc to current_period_end_utc
#     df['ended_at_utc'] = np.where(
#         (df['canceled_at_utc'].notna()) & 
#         (~df['canceled_during_refund_period']) &
#         (~df['canceled_during_trial']), 
#         df['current_period_end_utc'],
#         df['ended_at_utc']
#     )
#
#     # If status is not 'Active' or 'Trialing' but ended_at_utc is still NaT, set it to current_period_end_utc
#     df['ended_at_utc'] = np.where(
#         (df['status'] != 'active') & 
#         (df['status'] != 'trialing') &
#         (df['canceled_at_utc'].isna()) &
#         (df['ended_at_utc'].isna()),
#         df['current_period_end_utc'],
#         df['ended_at_utc']
#     )
#
#     return df
#
#
# df = add_ended_at_utc(df)
#
#
# # %%
# ######################################################################################
# # CALCULATING DURATIONS
# def calculate_duration(df):
#     """Calculate various durations in days with proper business logic"""
#
#     # Trial duration (if trial exists)
#     df['trial_duration'] = (df['trial_end_utc'] - df['trial_start_utc']).dt.days.fillna(0)
#
#     # Current period duration
#     df['current_period_duration'] = (df['current_period_end_utc'] - df['current_period_start_utc']).dt.days
#
#     # Trial-only subscription
#     df['trial_only_subscription'] = (
#         df['trial_start_utc'].notna() & 
#         df['trial_end_utc'].notna() & 
#         (df['trial_duration'] == df['current_period_duration'])
#     )
#
#     # Gift duration (only for gifted members)
#     df['gift_duration'] = df['current_period_duration'].where(df['is_gifted_member'], 0)
#
#     # Days until end for active subscriptions
#     df['end_in'] = ((df['current_period_end_utc'] - today_date).dt.days).where(df['status'] == 'active', np.nan)
#
#     # For active subscriptions: from created_utc to current_period_end_utc (projected)
#     # For ended subscriptions: from created_utc to ended_at_utc (actual)
#     df['expected_duration'] = np.where(
#         (df['ended_at_utc'].isna()),
#         (df['current_period_end_utc'] - df['created_utc']).dt.days,  # Active: projected duration
#         (df['ended_at_utc'] - df['created_utc']).dt.days             # Ended: actual duration
#     )
#
#     df['real_duration'] = np.where(
#             df['ended_at_utc'].notna(),
#             (df['ended_at_utc'] - df['created_utc']).dt.days,  # Ended: actual duration
#             (today_date - df['created_utc']).dt.days  # Active: duration until now
#     )
#
#
#     # Void duration (time between creation and start - should be minimal)
#     df['void_duration'] = (df['start_utc'] - df['created_utc']).dt.days
#
#     df['days_since_creation'] = (today_date - df['created_utc']).dt.days
#
#     return df
#
# df = calculate_duration(df)
#
#
# # %%
# ######################################################################################
# # WEEKS ARE FROM MONDAY TO SUNDAY
# def get_specific_past_week(weeks_back=1, reference_date=None):
#     """
#     Get specific date for a specific week.
#     weeks_back=1 : last week (from Monday to Sunday)
#     weeks_back=2 : previous week (from Monday to Sunday)
#     weeks_back=3 : three weeks ago (from Monday to Sunday)
#     """
#
#     if reference_date is None:
#         today = pd.Timestamp.now(tz='UTC')
#     else:
#         if hasattr(reference_date, 'tz') and reference_date.tz is not None:
#             today = pd.to_datetime(reference_date).tz_convert('UTC')
#         else:
#             today = pd.to_datetime(reference_date).tz_localize('UTC')
#
#
#     # Finding the Monday of the target week
#     days_since_monday = today.weekday()
#     this_monday = today - pd.Timedelta(days=days_since_monday)
#     target_monday = this_monday - pd.Timedelta(days=7 * weeks_back)
#     target_sunday = target_monday + pd.Timedelta(days=6)
#
#     week_start = target_monday.normalize()  # 00:00:00
#     week_end = target_sunday.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)  # 23:59:59
#
#     monday = target_monday.strftime('%d-%m-%y')
#     sunday = target_sunday.strftime('%d-%m-%y')
#
#     # Las week info
#     week_info = {
#         'weeks_ago': weeks_back,
#         'week_start': week_start,
#         'week_end': week_end, 
#         'year': target_monday.year,
#         'week_number': target_monday.isocalendar().week,
#         'year_week': f"{target_monday.year}-W{target_monday.isocalendar().week:02d}",
#         'monday': monday,
#         'sunday': sunday,
#     }
#
#     return week_info
#
# # %%
# ######################################################################################
# def get_full_members_count(df):
#     """Count the number of full members"""
#     df = df.copy()
#
#     df = df[df['is_full_member'] == True]
#     df_active = df[df['status'] == 'active']
#
#     active = len(df_active)
#     print(f"Total Active full member: {active}")
#
#
#     dict_full_members = {'active': active
#                          }
#
#     return dict_full_members
#
#
# dict_full_member = get_full_members_count(df)
#
#
# # %%
# #######################################################################################
# # find the maximun count of full menbers at any time
# def get_max_concurrent_full_members(df, dict_full_member):
#     """Find maximum concurrent full members using membership periods"""
#     events = []
#     full_members = df[df['is_full_member']].copy()
#
#     start_events = 0
#     end_events = 0
#     invalid_starts = 0    
#
#     for _, row in full_members.iterrows():  # ✅ CORRIGÉ ICI !
#         start_date = row['refund_period_end_utc']
#
#         if pd.notna(start_date):
#             events.append((start_date, 1))
#             start_events += 1
#         else:
#             invalid_starts += 1
#
#         if pd.notna(row['ended_at_utc']):
#             events.append((row['ended_at_utc'], -1))
#             end_events += 1
#
#     print(f"Events créés: start={start_events}, end={end_events}")
#     print(f"Starts invalides: {invalid_starts}")
#
#     if len(events) == 0:
#         print("❌ Aucun event valide !")
#         dict_full_member['max_full_members'] = 0
#         dict_full_member['max_date'] = None
#         return dict_full_member
#
#     events.sort()
#     current_count = 0
#     max_count = 0
#     max_date = None
#
#     for date, change in events:
#         current_count += change
#         if current_count > max_count:
#             max_count = current_count
#             max_date = date
#
#     # Validation
#     current_active = len(df[
#         (df['is_full_member'] == True) & 
#         (df['status'] == 'active')
#     ])
#
#     print(f"\n📊 RÉSULTATS:")
#     print(f"Maximum concurrent: {max_count}")
#     print(f"Date du maximum: {max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else 'N/A'}")
#     print(f"Count final simulation: {current_count}")
#     print(f"Actifs actuels: {current_active}")
#
#     if current_count != current_active:
#         print(f"⚠️  ATTENTION: Simulation ({current_count}) ≠ Actifs ({current_active})")
#         print("   Vérifiez vos données ended_at_utc")
#     else:
#         print("✅ Cohérence vérifiée")
#
#     dict_full_member['max_full_members'] = max_count
#     dict_full_member['max_date'] = max_date
#
#     return dict_full_member
#
# dict_full_members = get_max_concurrent_full_members(df, dict_full_member)
#
#
# # %%
# ######################################################################################
# # how many trial this week
# def get_new_trial_last_week(df, weeks_back=1):
#     """Count new trials started this week"""
#     week_info = get_specific_past_week(weeks_back=weeks_back, reference_date=today_date)
#
#     # Filter for the current week
#     df_week = df[(df['trial_start_utc'] >= week_info['week_start']) & 
#                  (df['trial_start_utc'] < week_info['week_end'])]
#
#     # Count new trials
#     new_trials = df_week.shape[0]
#
#     print(f"New trials last week ({week_info['year_week']}): {new_trials}")
#
#     return new_trials
#
#
# new_trial_last_week = get_new_trial_last_week(df, weeks_back=1)
# new_trial_prev_week = get_new_trial_last_week(df, weeks_back=2)
#
# # %%
# ######################################################################################
# # Count trials that converted to full members        
# def get_conversion_rate(df):
#     """Calculate conversion rate from trial to full member"""
#     df = df.copy()
#
#     new_customers = df[df['trial_start_utc'].notna()].copy()
#
#     mature_trials = new_customers[new_customers['refund_period_end_utc'] < today_date]
#
#     total_trials = len(mature_trials)
#     conversions = len(mature_trials[mature_trials['is_full_member'] == True])
#
#     conversion_rate = (conversions / total_trials * 100)
#
#
#     conversion_rate_dict = {
#         'total_trials': total_trials,
#         'mature_trials': mature_trials,
#         'conversion_rate': round(conversion_rate, 2)
#     }
#
#     return conversion_rate_dict
#
#
# conversion_rate_dict = get_conversion_rate(df)
#
#
# # %%
# ######################################################################################
# # Count trials that converted to full members        
# def get_conversion_rate_last_week(df, weeks_back=1):
#     """Calculate conversion rate from trial to full member"""
#     df = df.copy()
#
#     week_info = get_specific_past_week(weeks_back=weeks_back, reference_date=today_date)
#     # Filter for the current week
#     mature_customers = df[(df['refund_period_end_utc'] >= week_info['week_start']) &
#                           (df['refund_period_end_utc'] < week_info['week_end']) &
#                           (df['trial_start_utc'].notna())].copy()
#
#
#     total_trials = len(mature_customers)
#     conversions = len(mature_customers[mature_customers['is_full_member'] == True])
#
#     conversion_rate = (conversions / total_trials * 100)
#
#
#     return conversion_rate
#
#
# last_week_conversion_rate = get_conversion_rate_last_week(df, weeks_back=1)
# prev_week_conversion_rate = get_conversion_rate_last_week(df, weeks_back=2)
#
# # %%
# ######################################################################################
# # Count renwals rate
# def get_renewal_rate(df):
#     """Calculate renewal rate from full members + cancellation rate Y1"""
#
#     renewals = df[(df['trial_start_utc'].isna()) & (~df['is_gifted_member'])].copy()
#
#     active_renewals = renewals[renewals['status'] == 'active']
#     successful_renewals = renewals[renewals['is_full_member'] == True]
#
#
#
#     renewed = eligible[eligible['real_duration'] > (364 + REFUND_PERIOD_DAYS)]
#
#     renewal_rate = (len(renewed) / len(eligible)) * 100
#
#     # === AJOUT : CALCUL DU TAUX D'ANNULATION PREMIÈRE ANNÉE ===
#     # Members who have been around long enough to potentially complete first year
#     eligible_first_year = df[
#         (df['real_duration'] >= 365) &  # At least 1 year since creation
#         (~df['is_gifted_member']) &     # Not gifted
#         (df['trial_start_utc'].notna()) # Had a trial (not renewal)
#     ]
#
#     # Members who canceled during their first year
#     canceled_first_year = eligible_first_year[
#         (eligible_first_year['canceled_at_utc'].notna()) &
#         (eligible_first_year['real_duration'] <= 365)
#     ]
#
#     # Calculate cancellation rate for first year
#     if len(eligible_first_year) > 0:
#         cancellation_rate_y1 = (len(canceled_first_year) / len(eligible_first_year)) * 100
#     else:
#         cancellation_rate_y1 = 0
#
#     renewal_rate_dict = {
#         'eligible': len(eligible),
#         'renewed': len(renewed),
#         'renewal_rate': round(renewal_rate, 2),
#         # NOUVEAUX CHAMPS AJOUTÉS
#         'cancellation_rate': round(cancellation_rate_y1, 1),
#         'eligible_first_year': len(eligible_first_year),
#         'canceled_first_year': len(canceled_first_year)
#     }
#
#     print(f"Renewal rate: {renewal_rate:.1f}%, Cancellation rate Y1: {cancellation_rate_y1:.1f}%")
#
#     return renewal_rate_dict
#
#
# # %%
# ######################################################################################
# def get_renewal_rates_by_year(df):
#     """
#     Calculate renewal rates from Year 1 to Year 2, and Year 2 to Year 3
#
#     Returns:
#         dict: Contains renewal rates for different year transitions
#     """
#
#     # Year 1 to Year 2 renewal rate
#     eligible_for_y2 = df[
#         (df['real_duration'] >= 364) & 
#         (~df['is_gifted_member']) 
#     ]
#
#     renewed_to_y2 = eligible_for_y2[
#         (eligible_for_y2['real_duration'] > 380) &
#         (~eligible_for_y2['canceled_during_trial']) &
#         (~eligible_for_y2['canceled_during_refund_period'])]
#
#     y1_to_y2_rate = (len(renewed_to_y2) / len(eligible_for_y2) * 100) if len(eligible_for_y2) > 0 else 0
#
#     # Year 2 to Year 3 renewal rate  
#     eligible_for_y3 = df[
#         (df['real_duration'] > 729) &
#         (~df['is_gifted_member']) 
#     ]
#
#     renewed_to_y3 = eligible_for_y3[
#         (eligible_for_y3['real_duration'] > 731) &
#         (~eligible_for_y3['canceled_during_trial']) &
#         (~eligible_for_y3['canceled_during_refund_period'])] 
#
#     y2_to_y3_rate = (len(renewed_to_y3) / len(eligible_for_y3) * 100) if len(eligible_for_y3) > 0 else 0
#
#     print(f"📊 RENEWAL RATES BY YEAR:")
#     print(f"Year 1 → Year 2: {y1_to_y2_rate:.1f}% ({len(renewed_to_y2)}/{len(eligible_for_y2)})")
#     print(f"Year 2 → Year 3: {y2_to_y3_rate:.1f}% ({len(renewed_to_y3)}/{len(eligible_for_y3)})")
#
#     renewal_dict = {
#         'y1_to_y2_rate': round(y1_to_y2_rate, 1),
#         'y2_to_y3_rate': round(y2_to_y3_rate, 1),
#         'eligible_for_y2': len(eligible_for_y2),
#         'renewed_to_y2': len(renewed_to_y2),
#         'eligible_for_y3': len(eligible_for_y3),
#         'renewed_to_y3': len(renewed_to_y3)
#     }
#
#     return renewal_dict 
#
# renewal_dict = get_renewal_rates_by_year(df)
#
# # %%
# ################################################################################################
# def get_active_members_by_year(df, renewal_dict):
#     """
#     Calculate active members in 1st, 2nd and 3rd year + active second year count
#
#     Returns:
#         dict: Contains counts of active members by year
#     """
#
#     # Active 1st year members (0-366 days since creation)
#     active_1st_year = df[
#         (df['real_duration'] <= 366) & 
#         (df['status'] == 'active') &
#         (df['is_full_member'] == True) 
#     ]
#
#     # Active 2nd year members (366-730 days since creation)
#     active_2nd_year = df[
#         (df['real_duration'] > 366) & 
#         (df['real_duration'] <= 730) &
#         (df['status'] == 'active') &
#         (df['is_full_member'] == True) 
#     ]
#
#     # Active 3rd year members (730+ days since creation)
#     active_3rd_year = df[
#         (df['real_duration'] > 730) &
#         (df['status'] == 'active') &
#         (df['is_full_member'] == True) 
#     ]
#
#     print(f"📊 ACTIVE MEMBERS BY YEAR:")
#     print(f"Active 1st Year: {len(active_1st_year)}")
#     print(f"Active 2nd Year: {len(active_2nd_year)}")
#     print(f"Active 3rd Year+: {len(active_3rd_year)}")
#
#     renewal_dict['active_1st_year'] = len(active_1st_year)
#     renewal_dict['active_2nd_year'] = len(active_2nd_year)
#     renewal_dict['active_3rd_year'] = len(active_3rd_year)
#
#     # === AJOUT : ACTIVE SECOND YEAR SUM (pour weekly_flow) ===
#     renewal_dict['active_second_year_sum'] = len(active_2nd_year)
#
#     return renewal_dict
#
#
# renewal_dict = get_active_members_by_year(df, renewal_dict)
#
# # %%
# ######################################################################################
# def get_new_full_members_last_week(df, today_date, weeks_back=1):
#     """
#     Get new full members from last week
#     """
#     week_info = get_specific_past_week(weeks_back=weeks_back, reference_date=today_date)
#
#     # New full members = those whose refund period ended last week
#     last_week_full_members = df[
#         (df['refund_period_end_utc'] >= week_info['week_start']) & 
#         (df['refund_period_end_utc'] <= week_info['week_end']) &
#         (df['is_full_member'] == True)
#     ]
#
#     last_week_full_members = len(last_week_full_members)
#
#     return last_week_full_members
#
# last_week_full_members = get_new_full_members_last_week(df, today_date, weeks_back=1)
# prev_week_full_members = get_new_full_members_last_week(df, today_date, weeks_back=2)
#
#
#
# # %%
# ######################################################################################
# def get_churn_members_last_week(df, today_date, weeks_back=1):
#     """
#     Get churned members from last week (using the function from previous artifact)
#     """
#     week_info = get_specific_past_week(weeks_back=weeks_back, reference_date=today_date)
#
#     # Churned members = those who canceled last week
#     churned_members = df[
#         (df['canceled_at_utc'] >= week_info['week_start']) & 
#         (df['canceled_at_utc'] <= week_info['week_end']) &
#         (df['is_full_member'] == True)
#     ]
#
#     last_week_churned_members = len(churned_members)
#
#     return last_week_churned_members
#
# last_week_churned_members = get_churn_members_last_week(df, today_date, weeks_back=1)
# prev_week_churned_members = get_churn_members_last_week(df, today_date, weeks_back=2)
#
#
# # %%
# ######################################################################################
# print('***************************************************')  
#
# def renewal_rate_2(df):
#     df = df.copy()
#     df = df[~df['is_gifted_member']]  # Exclude gifted members
#
#     # Number of Trial Subscriptions
#     trial_sub = df[df['trial_start_utc'].notna()]
#     print(f"trial_sub: {len(trial_sub)}")
#
#     # Number of Full Members
#     first_year_sub = df[(df['trial_start_utc'].notna() & (df['is_full_member']))]
#     print(f"first_year_sub: {len(first_year_sub)}")
#
#     # Trail > Full Member Conversion Rate
#     conversion_rate = ((len(first_year_sub) / len(trial_sub)) * 100)
#     print(f"Conversion rate: {conversion_rate:.2f}%")
#
#     print('------------------------')
#
#
#     # Full Members who Canceled during the First Year
#     canceled_during_first_year = first_year_sub[first_year_sub['canceled_at_utc'].notna()]
#     print(f"canceled_during_first_year: {len(canceled_during_first_year)}")
#
#     # Full Members who Canceled during the First Year Rate
#     full_members_canceled_first_year_rate = ((len(canceled_during_first_year) / len(first_year_sub)) * 100)
#     print(f"Full members canceled during first year: {full_members_canceled_first_year_rate:.2f}%")
#
#     # Full Members who didn't Cancel during the First Year
#     not_canceled_first_year = first_year_sub[first_year_sub['canceled_at_utc'].isna()]
#     print(f"not_canceled_first_year : {len(not_canceled_first_year)}")
#
#     # Full Members who didn't Cancel during the First Year Rate
#     full_members_not_canceled_first_year_rate = ((len(not_canceled_first_year) / len(first_year_sub)) * 100)
#     print(f"Full members not canceled during first year: {full_members_not_canceled_first_year_rate:.2f}%")
#
#     # List of customers not cancel during first year
#     list_not_canceled_first_year = not_canceled_first_year['customer_id'].tolist()
#     print(f"List of customers not cancel during first year: {len(list_not_canceled_first_year)}")
#
#
#     print('------------------------')
#
#
#     # Full Members who Renewed for the Second Year
#     renewed_for_2nd_year = df[(df['trial_start_utc'].isna()) & (df['real_duration'] <= 366)]
#     # renewed_for_2nd_year = df[df['trial_start_utc'].isna()]
#     print(f"renewed_for_2nd_year: {len(renewed_for_2nd_year)}")
#
#     # Full Members who Renewed for the Second Year
#     full_renewed_for_2nd_year = renewed_for_2nd_year[renewed_for_2nd_year['is_full_member']]
#     print(f"full_renewed_for_2nd_year: {len(full_renewed_for_2nd_year)}")
#
#     # Full Members who Renewed for the Second Year Rate
#     renewal_rate_2nd_year = ((len(full_renewed_for_2nd_year) / len(renewed_for_2nd_year)) * 100)
#     print(f"Renewal rate for 2nd year: {renewal_rate_2nd_year:.2f}%")
#
#     canceled_during_second_year = full_renewed_for_2nd_year[full_renewed_for_2nd_year['canceled_at_utc'].notna()]
#     print(f"canceled_during_second_year: {len(canceled_during_second_year)}")
#
#
#
#     print('------------------------')
#
#
#     # Full Members who Renewed for the Third Year
#     renewed_for_3rd_year = df[(df['trial_start_utc'].isna()) & (df['real_duration'] > 366)]
#     print(f"renewed_for_3rd_year: {len(renewed_for_3rd_year)}")
#
#     # Full Members who Renewed for the 3rd Year
#     full_renewed_for_3rd_year = renewed_for_3rd_year[renewed_for_3rd_year['is_full_member']]
#     print(f"full_renewed_for_3rd_year: {len(full_renewed_for_3rd_year)}")
#
#     # Full Members who Renewed for the 3rd Year Rate
#     renewal_rate_3rd_year = ((len(full_renewed_for_3rd_year) / len(renewed_for_3rd_year)) * 100)
#     print(f"Renewal rate for 3rd year: {renewal_rate_3rd_year:.2f}%")
#
#
#     correct_renewal_dict = {'trial_sub': len(trial_sub),
#                             'first_year_sub': len(first_year_sub),
#                             'conversion_rate': round(conversion_rate, 2),
#                             'canceled_during_first_year': len(canceled_during_first_year),
#                             'full_members_canceled_first_year_rate': round(full_members_canceled_first_year_rate, 2),
#                             'not_canceled_first_year': len(not_canceled_first_year),
#                             'full_members_not_canceled_first_year_rate': round(full_members_not_canceled_first_year_rate, 2),
#                             'list_not_canceled_first_year': list_not_canceled_first_year,
#                             'renewed_for_2nd_year': len(renewed_for_2nd_year),
#                             'full_renewed_for_2nd_year': len(full_renewed_for_2nd_year),
#                             'renewal_rate_2nd_year': round(renewal_rate_2nd_year, 2),
#                             'canceled_during_second_year': len(canceled_during_second_year),
#                             'renewed_for_3rd_year': len(renewed_for_3rd_year),
#                             'full_renewed_for_3rd_year': len(full_renewed_for_3rd_year),
#                             'renewal_rate_3rd_year': round(renewal_rate_3rd_year, 2)
#                            }
#
#     return correct_renewal_dict
#
# correct_renewal_dict = renewal_rate_2(df)
# print('***************************************************')  
#
# # %%
# ######################################################################################
# def weekly_flow_all_time(df, today_date):
#     """
#     Create a dual-axis chart with weekly metrics for ALL TIME
#     North: Conversions + Renewals (stacked)
#     South: Churn full members
#     + Cumulative line plot
#     """
#
#     # Use all data since first date
#     first_date = df['created_utc'].min()
#     num_weeks = int((today_date - first_date).days / 7) + 1
#     print(f"Analysis since first date: {first_date.strftime('%d-%m-%Y')} ({num_weeks} weeks)")
#
#     # Create list of weeks
#     week_data = []
#     week_range = []
#
#     for weeks_back in range(num_weeks, 0, -1):  # From oldest to most recent
#         week_info = get_specific_past_week(weeks_back=weeks_back, reference_date=today_date)
#         week_data.append(week_info)
#         week_range.append(week_info['week_start'])
#
#     # Initialize lists to store weekly data
#     conversions_data = []
#     renewals_y1_data = []
#     renewals_y2_data = []
#     churn_data = []
#     new_trials_data = []
#
#     # === CALCULATE METRICS FOR EACH WEEK ===
#     for i, week_info in enumerate(week_data):
#         # === CONVERSIONS ===
#         week_conversions = df[
#             (df['created_utc'] >= week_info['week_start']) &
#             (df['created_utc'] <= week_info['week_end']) &
#             (df['trial_start_utc'].notna()) &  # Had a trial
#             (df['is_full_member'] == True) &   # Became full member
#             (df['refund_period_end_utc'] < today_date)]
#         conversions_data.append(len(week_conversions))
#
#         # === NEW TRIALS ===
#         week_trials = df[
#             (df['trial_start_utc'] >= week_info['week_start']) &
#             (df['trial_start_utc'] <= week_info['week_end'])]
#         new_trials_data.append(len(week_trials))
#
#         # === RENEWALS Y1 ===
#         week_renewals_y1 = df[
#             (df['created_utc'] >= week_info['week_start']) &
#             (df['created_utc'] <= week_info['week_end']) &
#             (df['trial_start_utc'].isna()) &      # No trial = renewal
#             (~df['is_gifted_member']) &           # No gift
#             (df['is_full_member'] == True) &      # Full member
#             (df['real_duration'] <= 364)]
#         renewals_y1_data.append(len(week_renewals_y1))
#
#         # === RENEWALS Y2+ ===
#         week_renewals_y2 = df[
#             (df['created_utc'] >= week_info['week_start']) &
#             (df['created_utc'] <= week_info['week_end']) &
#             (df['trial_start_utc'].isna()) &      # No trial = renewal
#             (~df['is_gifted_member']) &           # No gift
#             (df['is_full_member'] == True) &      # Full member
#             (df['real_duration'] > 366)]
#         renewals_y2_data.append(len(week_renewals_y2))
#
#         # === CHURN FULL MEMBERS ===
#         week_churn = df[
#             (df['canceled_at_utc'] >= week_info['week_start']) &
#             (df['canceled_at_utc'] <= week_info['week_end']) &
#             (df['is_full_member'] == True)]
#         churn_data.append(len(week_churn))
#
#     # Convert to pandas series
#     conversions_weekly = pd.Series(conversions_data, index=week_range)
#     renewals_y1_weekly = pd.Series(renewals_y1_data, index=week_range)
#     renewals_y2_weekly = pd.Series(renewals_y2_data, index=week_range)
#     churn_weekly = pd.Series(churn_data, index=week_range)
#     trials_weekly = pd.Series(new_trials_data, index=week_range)
#
#     # Calculate cumulative values
#     net_weekly = conversions_weekly + renewals_y1_weekly + renewals_y2_weekly - churn_weekly
#     net_cumul = net_weekly.cumsum()
#
#     # === CREATE CHART - ALL TIME ===
#     fig, ax = plt.subplots(1, 1, figsize=(22, 8))
#
#     # Format dates for X axis
#     weeks_labels = [week_info['monday'] + ' > ' + week_info['sunday'] for week_info in week_data]
#     x_pos = range(len(week_range))
#
#     # === POSITIVE BARPLOT (NORTH) ===
#     ax.bar(x_pos, conversions_weekly, label='Conversions (Trial→Full)', color='green')
#     ax.bar(x_pos, renewals_y1_weekly, bottom=conversions_weekly,label='Renewals Y1', color='lightgreen')
#     ax.bar(x_pos, renewals_y2_weekly, \
#            bottom=conversions_weekly + renewals_y1_weekly, \
#            label='Renewals Y2+', color='orange')
#
#     # === NEGATIVE BARPLOT (SOUTH) ===
#     ax.bar(x_pos, -churn_weekly, label='Churn Full Members', color='red')
#
#     # === CUMULATIVE LINE PLOT ===
#     ax_twin = ax.twinx()
#     ax_twin.plot(x_pos, net_cumul, color='darkblue', linewidth=1, \
#                  label='Net Cumulative (Gains - Losses)')
#
#     # === AXIS CONFIGURATION ===
#     ax.set_ylabel('Full Members per week\n(Positive: Gains | Negative: Losses)', 
#                   fontsize=12, fontweight='bold')
#     ax.set_xlabel('Weeks (Monday - Sunday)', fontsize=12, fontweight='bold')
#
#     # Adding numbers on bars (only if reasonable number)
#     for i, (conv, ren1, ren2) in enumerate(zip(conversions_weekly, renewals_y1_weekly, renewals_y2_weekly)):
#         total_positive = conv + ren1 + ren2
#         if total_positive > 0:
#             ax.text(i, total_positive + 1, str(int(total_positive)), 
#                    ha='center', va='bottom', fontsize=8, color='green')
#
#     for i, v in enumerate(churn_weekly):
#         if v > 0:
#             ax.text(i, -v - 1, str(int(v)), 
#                    ha='center', va='top', fontsize=8, color='red')
#
#     ax_twin.set_ylabel('Net Cumulative Total', fontsize=12, fontweight='bold', color='darkblue')
#     ax_twin.tick_params(axis='y', labelcolor='darkblue')
#
#     # === VISUAL CONFIGURATION ===
#     ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
#     ax.grid(True, alpha=0.3)
#
#     ax.set_xlim(-0.3, len(x_pos) - 0.5)
#
#     # Adjust Y limits
#     y_max = max(conversions_weekly + renewals_y1_weekly + renewals_y2_weekly) * 1.2
#     y_min = -max(churn_weekly) * 1.2
#     ax.set_ylim(y_min, y_max)
#
#     # X axis configuration - reduce labels for long periods
#     ax.set_xticks(x_pos[::max(1, len(x_pos)//15)])
#     ax.set_xticklabels([weeks_labels[i] for i in x_pos[::max(1, len(x_pos)//15)]], 
#                        rotation=45, ha='right')
#
#     # === GREY ZONE FOR IMMATURE PERIODS ===
#     immature_cutoff = today_date - pd.Timedelta(days=24)
#     immature_indices = []
#     for i, week_info in enumerate(week_data):
#         if week_info['week_start'] >= immature_cutoff:
#             immature_indices.append(i)
#
#     if immature_indices:
#         start_idx = min(immature_indices) - 0.5
#         end_idx = max(immature_indices) + 0.5
#         ax.axvspan(start_idx, end_idx, alpha=0.2, color='grey', 
#                    label='Pending conversion period (< 24 days)', zorder=0)
#         print(f"Grey zone covers {len(immature_indices)} recent weeks")
#
#     # === TITLES AND LEGENDS ===
#     period_text = f'(from {week_data[0]["monday"]} to {week_data[-1]["sunday"]})'
#     ax.set_title(f'WEEKLY FULL MEMBERS FLOW - ALL TIME\n{period_text}', fontsize=18, fontweight='bold', pad=30)
#
#     # Combine legends
#     lines1, labels1 = ax.get_legend_handles_labels()
#     lines2, labels2 = ax_twin.get_legend_handles_labels()
#     ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
#
#     # === SUMMARY METRICS ===
#     print("=== CALCULATING METRICS ===")
#
#     conversion_rate_dict = get_conversion_rate(df)
#     renewal_rate_dict = get_renewal_rate(df)  # Maintenant inclut cancellation_rate
#     renewal_dict_temp = get_renewal_rates_by_year(df)
#     renewal_dict = get_active_members_by_year(df, renewal_dict_temp)  # Maintenant inclut active_second_year_sum
#
#     total_conversions = conversions_weekly.sum()
#     total_renewals_y1 = renewals_y1_weekly.sum()
#     total_renewals_y2 = renewals_y2_weekly.sum()
#     total_churn = churn_weekly.sum()
#     total_trials = trials_weekly.sum()
#     net_growth = total_conversions + total_renewals_y1 + total_renewals_y2 - total_churn   
#
#     summary_text = f"""ALL TIME SUMMARY ({num_weeks} weeks):
# Total trials: {total_trials:,}
# Conversions: {total_conversions:,} ({conversion_rate_dict.get('conversion_rate', 0):.1f}%)
# Renewals Year 1: {total_renewals_y1:,}
# Renewals Year 2: {total_renewals_y2:,} 
# Renewal rate: {renewal_rate_dict.get('renewal_rate', 0):.1f}%
# Total Churn: {total_churn:,}
# ────────────────
# Net growth: {net_growth:,}"""
#
#     # fig.text(0.4, 0.2, summary_text, fontsize=11, 
#     #          bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgray", alpha=0.8))
#
#     plt.tight_layout()
#
#     # === SAVE ===
#     filename = f"weekly_flow_all_time_{today_date.strftime('%Y-%m-%d')}.png"
#     plt.savefig(os.path.join(analysis_dir, filename), dpi=300, bbox_inches='tight')
#     print(f"All time chart saved: {filename}")
#     plt.show()
#
#
#     metrics_all = {'conversions': total_conversions, \
#         'renewals_y1': total_renewals_y1, \
#         'renewals_y2': total_renewals_y2, \
#         'churn': total_churn, \
#         'trials': total_trials, \
#         'net_growth': net_growth, \
#         'conversion_rate': conversion_rate_dict.get('conversion_rate', 0), \
#         'renewal_rate': renewal_rate_dict.get('renewal_rate', 0), \
#         'active_second_year': renewal_dict.get('active_second_year_sum', 0), \
#         'cancellation_rate_y1': renewal_rate_dict.get('cancellation_rate', 0), \
#         'num_weeks': num_weeks}
#
#
#     return  metrics_all
#
#
# metrics_all = weekly_flow_all_time(df, today_date)
#
#
# # %%
# #######################################################################################
#
# def weekly_flow_8_weeks(df, today_date, num_weeks=8):
#     """
#     Create a dual-axis chart with weekly metrics for last N weeks (default 8)
#     North: Conversions + Renewals (stacked)
#     South: Churn full members
#     + Cumulative line plot
#     """
#
#     print(f"Analysis of last {num_weeks} weeks")
#
#     # Create list of weeks
#     week_data = []
#     week_range = []
#
#     for weeks_back in range(num_weeks, 0, -1):  # From oldest to most recent
#         week_info = get_specific_past_week(weeks_back=weeks_back, reference_date=today_date)
#         week_data.append(week_info)
#         week_range.append(week_info['week_start'])
#
#     # Initialize lists to store weekly data
#     conversions_data = []
#     renewals_y1_data = []
#     renewals_y2_data = []
#     churn_data = []
#     new_trials_data = []
#
#     # === CALCULATE METRICS FOR EACH WEEK ===
#     for i, week_info in enumerate(week_data):
#         # === CONVERSIONS ===
#         week_conversions = df[
#             (df['created_utc'] >= week_info['week_start']) &
#             (df['created_utc'] <= week_info['week_end']) &
#             (df['trial_start_utc'].notna()) &  # Had a trial
#             (df['is_full_member'] == True) &   # Became full member
#             (df['refund_period_end_utc'] < today_date)]
#         conversions_data.append(len(week_conversions))
#
#         # === NEW TRIALS ===
#         week_trials = df[
#             (df['trial_start_utc'] >= week_info['week_start']) &
#             (df['trial_start_utc'] <= week_info['week_end'])]
#         new_trials_data.append(len(week_trials))
#
#         # === RENEWALS Y1 ===
#         week_renewals_y1 = df[
#             (df['created_utc'] >= week_info['week_start']) &
#             (df['created_utc'] <= week_info['week_end']) &
#             (df['trial_start_utc'].isna()) &      # No trial = renewal
#             (~df['is_gifted_member']) &           # No gift
#             (df['is_full_member'] == True) &      # Full member
#             (df['days_since_creation'] <= 365)]
#         renewals_y1_data.append(len(week_renewals_y1))
#
#         # === RENEWALS Y2+ ===
#         week_renewals_y2 = df[
#             (df['created_utc'] >= week_info['week_start']) &
#             (df['created_utc'] <= week_info['week_end']) &
#             (df['trial_start_utc'].isna()) &      # No trial = renewal
#             (~df['is_gifted_member']) &           # No gift
#             (df['is_full_member'] == True) &      # Full member
#             (df['days_since_creation'] > 365)]
#         renewals_y2_data.append(len(week_renewals_y2))
#
#         # === CHURN FULL MEMBERS ===
#         week_churn = df[
#             (df['canceled_at_utc'] >= week_info['week_start']) &
#             (df['canceled_at_utc'] <= week_info['week_end']) &
#             (df['is_full_member'] == True)]
#         churn_data.append(len(week_churn))
#
#     # Convert to pandas series
#     conversions_weekly = pd.Series(conversions_data, index=week_range)
#     renewals_y1_weekly = pd.Series(renewals_y1_data, index=week_range)
#     renewals_y2_weekly = pd.Series(renewals_y2_data, index=week_range)
#     churn_weekly = pd.Series(churn_data, index=week_range)
#     trials_weekly = pd.Series(new_trials_data, index=week_range)
#
#     # Calculate cumulative values
#     net_weekly = conversions_weekly + renewals_y1_weekly + renewals_y2_weekly - churn_weekly
#     net_cumul = net_weekly.cumsum()
#
#     # === CREATE CHART - SHORT PERIOD ===
#     fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#
#     # Format dates for X axis
#     weeks_labels = [week_info['monday'] + ' > ' + week_info['sunday'] for week_info in week_data]
#     x_pos = range(len(week_range))
#
#     bar_width = 0.8 # Width of the bars
#
#     # === POSITIVE BARPLOT (NORTH) ===
#     ax.bar(x_pos, conversions_weekly, label='Conversions (Trial→Full)', color='green', width=bar_width)
#     ax.bar(x_pos, renewals_y1_weekly, bottom=conversions_weekly, width=bar_width, \
#            label='Renewals Y1', color='lightgreen')
#     ax.bar(x_pos, renewals_y2_weekly, width=bar_width, \
#            bottom=conversions_weekly + renewals_y1_weekly, \
#            label='Renewals Y2+', color='orange')
#
#     # === NEGATIVE BARPLOT (SOUTH) ===
#     ax.bar(x_pos, -churn_weekly, label='Churn Full Members', color='red', width=bar_width)
#
#     # === AXIS CONFIGURATION ===
#     ax.set_ylabel('Full Members per week\n(Positive: Gains | Negative: Losses)', 
#                   fontsize=12, fontweight='bold')
#     ax.set_xlabel('Weeks (Monday - Sunday)', fontsize=12, fontweight='bold')
#
#     # Adding numbers on all bars (since short period)
#     for i, (conv, ren1, ren2) in enumerate(zip(conversions_weekly, renewals_y1_weekly, renewals_y2_weekly)):
#         total_positive = conv + ren1 + ren2
#         if total_positive > 0:
#             ax.text(i, total_positive + 1, str(int(total_positive)), 
#                    ha='center', va='bottom', fontsize=9, color='green')
#
#     for i, v in enumerate(churn_weekly):
#         if v > 0:
#             ax.text(i, -v - 1, str(int(v)), 
#                    ha='center', va='top', fontsize=9, color='red')
#
#
#     # === VISUAL CONFIGURATION ===
#     ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
#     ax.grid(True, alpha=0.3)
#
#     ax.set_xlim(-0.5, len(x_pos) - 0.5)
#
#     # Adjust Y limits
#     y_max = max(conversions_weekly + renewals_y1_weekly + renewals_y2_weekly) * 1.2
#     y_min = -max(churn_weekly) * 1.2
#     ax.set_ylim(y_min, y_max)
#
#     # X axis configuration - show all labels for short periods
#     ax.set_xticks(x_pos)
#     ax.set_xticklabels(weeks_labels, rotation=45, ha='right')
#
#     # === GREY ZONE FOR IMMATURE PERIODS ===
#     immature_cutoff = today_date - pd.Timedelta(days=24)
#     immature_indices = []
#     for i, week_info in enumerate(week_data):
#         if week_info['week_start'] >= immature_cutoff:
#             immature_indices.append(i)
#
#     if immature_indices:
#         start_idx = min(immature_indices) - 0.5
#         end_idx = max(immature_indices) + 0.5
#         ax.axvspan(start_idx, end_idx, alpha=0.2, color='grey', 
#                    label='Pending conversion period (< 24 days)', zorder=0)
#         print(f"Grey zone covers {len(immature_indices)} recent weeks")
#
#     # === TITLES AND LEGENDS ===
#     period_text = f'{num_weeks} last weeks (from {week_data[0]["monday"]} to {week_data[-1]["sunday"]})'
#     ax.set_title(f'WEEKLY FULL MEMBERS FLOW\n{period_text}', \
#                  fontsize=18, fontweight='bold', pad=30)
#
#     # Combine legends
#     lines1, labels1 = ax.get_legend_handles_labels()
#     ax.legend(lines1, labels1, loc='upper right', fontsize=10)
#
#     # === SUMMARY METRICS ===
#     print("=== CALCULATING METRICS ===")
#
#     conversion_rate_dict = get_conversion_rate(df)
#     renewal_rate_dict = get_renewal_rate(df)  # Maintenant inclut cancellation_rate
#     renewal_dict_temp = get_renewal_rates_by_year(df)
#     renewal_dict = get_active_members_by_year(df, renewal_dict_temp)  # Maintenant inclut active_second_year_sum
#
#     total_conversions = conversions_weekly.sum()
#     total_renewals_y1 = renewals_y1_weekly.sum()
#     total_renewals_y2 = renewals_y2_weekly.sum()
#     total_churn = churn_weekly.sum()
#     total_trials = trials_weekly.sum()
#     net_growth = total_conversions + total_renewals_y1 + total_renewals_y2 - total_churn
#
#     summary_text = f"""SUMMARY {num_weeks} WEEKS:
# Total trials: {total_trials:,}
# Conversions: {total_conversions:,} ({conversion_rate_dict.get('conversion_rate', 0):.1f}%)
# Renewals Year 1: {total_renewals_y1:,}
# Renewals Year 2: {total_renewals_y2:,} 
# Renewal rate: {renewal_rate_dict.get('renewal_rate', 0):.1f}%
# Total Churn: {total_churn:,}
# ────────────────
# Net growth: {net_growth:,}"""
#
#     # fig.text(0.4, 0.2, summary_text, fontsize=11, 
#     #          bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgray", alpha=0.8))
#
#     plt.tight_layout()
#
#     # === SAVE ===
#     filename = f"weekly_flow_{num_weeks}_weeks_{today_date.strftime('%Y-%m-%d')}.png"
#     plt.savefig(os.path.join(analysis_dir, filename), dpi=300, bbox_inches='tight')
#     print(f"Short period chart saved: {filename}")
#     plt.show()
#
#     return {'conversions': total_conversions, \
#         'renewals_y1': total_renewals_y1, \
#         'renewals_y2': total_renewals_y2, \
#         'churn': total_churn, \
#         'trials': total_trials, \
#         'net_growth': net_growth, \
#         'conversion_rate': conversion_rate_dict.get('conversion_rate', 0), \
#         'renewal_rate': renewal_rate_dict.get('renewal_rate', 0), \
#         'active_second_year': renewal_dict.get('active_second_year_sum', 0), \
#         'cancellation_rate_y1': renewal_rate_dict.get('cancellation_rate', 0), \
#         'num_weeks': num_weeks}
#
#
#
# metrics_8w = weekly_flow_8_weeks(df, today_date, num_weeks=8)
#
#
# # %%
# ######################################################################################
# def plot_weekly_trials_all_time(df, today_date):
#     """
#     Plot the number of new trials each week since the beginning
#     """
#
#     # Use all data since first date
#     first_date = df['created_utc'].min()
#     num_weeks = int((today_date - first_date).days / 7) + 1
#     print(f"Analysis since first date: {first_date.strftime('%d-%m-%Y')} ({num_weeks} weeks)")
#
#     # Create list of weeks using your function
#     week_data = []
#     week_range = []
#
#     for weeks_back in range(num_weeks, 0, -1):  # From oldest to most recent
#         week_info = get_specific_past_week(weeks_back=weeks_back, reference_date=today_date)
#         week_data.append(week_info)
#         week_range.append(week_info['week_start'])
#
#     # Initialize list to store weekly trials data
#     trials_data = []
#
#     # === CALCULATE TRIALS FOR EACH WEEK ===
#     for i, week_info in enumerate(week_data):
#         week_trials = df[
#             (df['trial_start_utc'] >= week_info['week_start']) &
#             (df['trial_start_utc'] <= week_info['week_end'])]
#         trials_data.append(len(week_trials))
#
#     # Convert to pandas series
#     trials_weekly = pd.Series(trials_data, index=week_range)
#
#     # === CREATE CHART - ALL TIME ===
#     fig, ax = plt.subplots(1, 1, figsize=(22, 8))  # GRANDE taille pour all time
#
#     # Format dates for X axis
#     weeks_labels = [week_info['monday'] + ' > ' + week_info['sunday'] for week_info in week_data]
#     x_pos = range(len(week_range))
#
#     # === BARPLOT ===
#     bars = ax.bar(x_pos, trials_weekly, label='New Trials', 
#                   color='gray', alpha=0.8)
#
#     # Adding numbers on top of each bar (only for reasonable number of bars)
#     for i, v in enumerate(trials_weekly):
#         if v > 0:
#             ax.text(i, v + max(trials_weekly) * 0.01, str(int(v)), 
#                    ha='center', va='bottom', fontsize=8)
#
#     # === AXIS CONFIGURATION ===
#     ax.set_ylabel('Number of New Trials per Week', fontsize=12, fontweight='bold')
#     ax.set_xlabel('Weeks (Monday - Sunday)', fontsize=12, fontweight='bold')
#
#     # === TITLES ===
#     period_text = f'Since beginning ({week_data[0]["monday"]} - {week_data[-1]["sunday"]})'
#     ax.set_title(f'WEEKLY NEW TRIALS - ALL TIME\n{period_text}', \
#                  fontsize=18, fontweight='bold', pad=30)
#
#     ax.legend(loc='upper right', fontsize=10)
#
#     # === VISUAL CONFIGURATION ===
#     ax.grid(True, alpha=0.3, axis='y')
#     y_max = max(trials_weekly) * 1.1
#     ax.set_ylim(0, y_max)
#
#     ax.set_xlim(-0.3, len(x_pos))
#
#     # X axis configuration - adjust labels for long periods
#     ax.set_xticks(x_pos[::max(1, len(x_pos)//15)])
#     ax.set_xticklabels([weeks_labels[i] for i in x_pos[::max(1, len(x_pos)//15)]], 
#                        rotation=45, ha='right')
#
#     # === SUMMARY METRICS ===
#     total_trials = trials_weekly.sum()
#     avg_trials = trials_weekly.mean()
#     max_trials = trials_weekly.max()
#     min_trials = trials_weekly.min()
#
#     if len(trials_weekly) >= 2:
#         latest_week_text = f"Latest week: {trials_weekly.iloc[-1]:,} trials\nPrevious week: {trials_weekly.iloc[-2]:,} trials"
#     else:
#         latest_week_text = f"Latest week: {trials_weekly.iloc[-1]:,} trials\nPrevious week: N/A"
#
#     summary_text = f"""ALL TIME SUMMARY ({num_weeks} weeks):
# Total trials: {total_trials:,}
# Average per week: {avg_trials:.1f}
# Maximum week: {max_trials:,}
# Minimum week: {min_trials:,}
# ────────────────
# {latest_week_text}"""
#
#     # fig.text(0.02, 0.5, summary_text, fontsize=11, 
#     #          bbox=dict(boxstyle="round,pad=0.3", facecolor="gray", alpha=0.8))
#
#     plt.tight_layout()
#
#     # === SAVE ===
#     filename = f"weekly_trials_all_time_{today_date.strftime('%Y-%m-%d')}.png"
#     plt.savefig(os.path.join(analysis_dir, filename), dpi=300, bbox_inches='tight')
#
#     print(f"All time chart saved: {filename}")
#     plt.show()
#
#     return {'total_trials': total_trials, \
#         'average_per_week': avg_trials, \
#         'max_week': max_trials, \
#         'min_week': min_trials, \
#         'latest_week': trials_weekly.iloc[-1], \
#         'num_weeks': num_weeks, \
#         'weekly_data': trials_weekly.tolist()}
#
#
# trials_metrics_all = plot_weekly_trials_all_time(df, today_date)
#
#
# # %%
# #######################################################################################
#
# def plot_weekly_trials_8_weeks(df, today_date, num_weeks=8):
#     """
#     Plot the number of new trials each week for the last N weeks (default 8)
#     """
#
#     print(f"Analysis of last {num_weeks} weeks of trials")
#
#     # Create list of weeks
#     week_data = []
#     week_range = []
#
#     for weeks_back in range(num_weeks, 0, -1):  # From oldest to most recent
#         week_info = get_specific_past_week(weeks_back=weeks_back, reference_date=today_date)
#         week_data.append(week_info)
#         week_range.append(week_info['week_start'])
#
#     # Initialize list to store weekly trials data
#     trials_data = []
#
#     # === CALCULATE TRIALS FOR EACH WEEK ===
#     for i, week_info in enumerate(week_data):
#         week_trials = df[
#             (df['trial_start_utc'] >= week_info['week_start']) &
#             (df['trial_start_utc'] <= week_info['week_end'])
#     ]
#         trials_data.append(len(week_trials))
#
#     # Convert to pandas series
#     trials_weekly = pd.Series(trials_data, index=week_range)
#
#     # === CREATE CHART - SHORT PERIOD ===
#     fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#
#     # Format dates for X axis
#     weeks_labels = [week_info['monday'] + ' > ' + week_info['sunday'] for week_info in week_data]
#     x_pos = range(len(week_range))
#
#     # === BARPLOT ===
#     bars = ax.bar(x_pos, trials_weekly, label='New Trials', 
#                   color='gray', alpha=0.8)
#
#     # Adding numbers on top of each bar
#     for i, v in enumerate(trials_weekly):
#         if v > 0:
#             ax.text(i, v + max(trials_weekly) * 0.01, str(int(v)), 
#                    ha='center', va='bottom', fontsize=10)
#
#     # === AXIS CONFIGURATION ===
#     ax.set_ylabel('Number of New Trials per Week', fontsize=12, fontweight='bold')
#     ax.set_xlabel('Weeks (Monday - Sunday)', fontsize=12, fontweight='bold')
#
#     # === TITLES ===
#     period_text = f'Last {num_weeks} weeks ({week_data[0]["monday"]} - {week_data[-1]["sunday"]})'
#     ax.set_title(f'WEEKLY NEW TRIALS\n{period_text}', fontsize=18, fontweight='bold', pad=30)
#
#     ax.legend(loc='upper right', fontsize=10)
#
#     # === VISUAL CONFIGURATION ===
#     ax.grid(True, alpha=0.3, axis='y')
#     y_max = max(trials_weekly) * 1.1
#     ax.set_ylim(0, y_max)
#
#     # X axis configuration - show all labels for short periods
#     ax.set_xticks(x_pos)
#     ax.set_xticklabels(weeks_labels, rotation=45, ha='right')
#
#     # === SUMMARY METRICS ===
#     total_trials = trials_weekly.sum()
#     avg_trials = trials_weekly.mean()
#     max_trials = trials_weekly.max()
#     min_trials = trials_weekly.min()
#
#     if len(trials_weekly) >= 2:
#         latest_week_text = f"Latest week: {trials_weekly.iloc[-1]:,} trials\nPrevious week: {trials_weekly.iloc[-2]:,} trials"
#     else:
#         latest_week_text = f"Latest week: {trials_weekly.iloc[-1]:,} trials\nPrevious week: N/A"
#
#     summary_text = f"""SUMMARY {num_weeks} WEEKS:
# Total trials: {total_trials:,}
# Average per week: {avg_trials:.1f}
# Maximum week: {max_trials:,}
# Minimum week: {min_trials:,}
# ────────────────
# {latest_week_text}"""
#
#     # fig.text(0.02, 0.5, summary_text, fontsize=11, 
#     #          bbox=dict(boxstyle="round,pad=0.3", facecolor="gray", alpha=0.8))
#
#     plt.tight_layout()
#
#     # === SAVE ===
#     filename = f"weekly_trials_{num_weeks}_weeks_{today_date.strftime('%Y-%m-%d')}.png"
#     plt.savefig(os.path.join(analysis_dir, filename), dpi=300, bbox_inches='tight')
#
#     print(f"Short period chart saved: {filename}")
#     plt.show()
#
#     return {'total_trials': total_trials, \
#         'average_per_week': avg_trials, \
#         'max_week': max_trials, \
#         'min_week': min_trials, \
#         'latest_week': trials_weekly.iloc[-1], \
#         'num_weeks': num_weeks, \
#         'weekly_data': trials_weekly.tolist()}
#
#
# trials_metrics_8w = plot_weekly_trials_8_weeks(df, today_date, num_weeks=8)
#
# # %%
# #######################################################################################
# # COHORT CONVERSION FUNNEL ANALYSIS
#
# def plot_cohort_conversion_funnel(df, today_date):
#     """
#     Plot a conversion funnel for different cohorts with 3 bars:
#     1. Initial trials
#     2. Survivors after trial period (not canceled during trial)
#     3. Survivors after refund period (not canceled during refund)
#     """
#
#     complete_cohort_week = get_specific_past_week(weeks_back=4, reference_date=today_date)
#     complete_cohort_trials = df[(df['trial_start_utc'] >= complete_cohort_week['week_start']) & (df['trial_start_utc'] <= complete_cohort_week['week_end'])]
#
#     # Calculate funnel for complete cohort
#     total_trials = len(complete_cohort_trials)
#     survivors_trial = len(complete_cohort_trials[~complete_cohort_trials['canceled_during_trial']])
#     survivors_refund = len(complete_cohort_trials[
#         (~complete_cohort_trials['canceled_during_trial']) & 
#         (~complete_cohort_trials['canceled_during_refund_period'])
#     ])
#
#     fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#
#     categories = ['Initial Trials', 'Survived Trial Period', 'Full Members']
#     values = [total_trials, survivors_trial, survivors_refund]
#     colors = ['gray', 'red', 'darkgreen']
#
#     bars = ax.bar(categories, values, color=colors, alpha=0.8)
#
#     # Add numbers on top of bars
#     for i, (bar, value) in enumerate(zip(bars, values)):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01, \
#                 f'{value:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)
#
#         # Add percentage of previous stage
#         initial_trial = values[0]
#         if initial_trial > 0:
#             percentage = (value / initial_trial) * 100
#             ax.text(bar.get_x() + bar.get_width()/2., height/2,
#                     f'{percentage:.1f}%', ha='center', va='center', 
#                     fontweight='bold', fontsize=10, color='white')
#
#
#     ax.set_title(f'CONVERSION FUNNEL \n Last Complete Cohort Week {complete_cohort_week["year_week"]} \n(Trialers From {complete_cohort_week["monday"]} to {complete_cohort_week["sunday"]})', fontsize=16, fontweight='bold', pad=20)
#
#     ax.set_ylabel('# of users', fontsize=12)
#     ax.grid(axis='y', alpha=0.3)
#     ax.set_ylim(0, max(values) * 1.15)
#
#     # Add summary text
#     conversion_trial = (survivors_trial / total_trials * 100) if total_trials > 0 else 0
#     conversion_refund = (survivors_refund / total_trials * 100) if total_trials > 0 else 0
#
#     summary_text = f"""FUNNEL SUMMARY:
# Trial Survival Rate: {conversion_trial:.1f}%
# Full Conversion Rate: {conversion_refund:.1f}%
# Total Drop-off: {100 - conversion_refund:.1f}%"""
#
#     # fig.text(0.68, 0.77, summary_text, fontsize=11, 
#     #          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
#
#     plt.tight_layout()
#
#
#     filename = f"Conversion_funnel_{complete_cohort_week['year_week']}_{today_date.strftime('%Y-%m-%d')}.png"
#     plt.savefig(os.path.join(analysis_dir, filename), dpi=300, bbox_inches='tight')
#     print(f"Complete cohort conversion funel : {filename}")
#     plt.show()
#
#     last_cohort_dict = {'total_trials': total_trials, \
#         'survived_trial': survivors_trial, \
#         'survived_refund': survivors_refund, \
#         'conversion_trial_rate': conversion_trial, \
#         'conversion_refund_rate': conversion_refund, \
#         'total_drop_off': 100 - conversion_refund}
#
#     return last_cohort_dict
#
#
# last_cohort_dict = plot_cohort_conversion_funnel(df, today_date)
#
# # %%
# ########################################################################################
# def plot_cohort_conversion_funnel_comparison(df, today_date, last_cohort_dict):
#     """
#     Plot a conversion funnel comparing different cohorts with 3 bars:
#     1. Initial trials
#     2. Survivors after trial period (not canceled during trial)
#     3. Survivors after refund period (not canceled during refund)
#     """
#
#     last_total_trials = last_cohort_dict['total_trials']
#     last_survived_trial = last_cohort_dict['survived_trial']
#     last_survived_refund = last_cohort_dict['survived_refund']
#     last_conversion_trial_rate = last_cohort_dict['conversion_trial_rate']
#     last_conversion_refund_rate = last_cohort_dict['conversion_refund_rate']
#     last_total_drop_off = last_cohort_dict['total_drop_off']
#     complete_cohort_week = get_specific_past_week(weeks_back=4, reference_date=today_date)
#
#     # PREVIOUS WEEK cohorts to compare
#     prev_cohort_week = get_specific_past_week(weeks_back=5, reference_date=today_date)
#     prev_cohort_trials = df[(df['trial_start_utc'] >= prev_cohort_week['week_start']) & (df['trial_start_utc'] <= prev_cohort_week['week_end'])]
#
#     prev_total_trials = len(prev_cohort_trials)
#     prev_survivors_trial = len(prev_cohort_trials[~prev_cohort_trials['canceled_during_trial']])
#     prev_survivors_refund = len(prev_cohort_trials[
#         (~prev_cohort_trials['canceled_during_trial']) & 
#         (~prev_cohort_trials['canceled_during_refund_period'])
#     ])
#
#     # 6 Months average cohort
#     six_m_cohort_start = get_specific_past_week(weeks_back=24, reference_date=today_date)
#     six_m_cohort_end = get_specific_past_week(weeks_back=4, reference_date=today_date)
#
#     six_m_cohort_start = six_m_cohort_start['week_start']
#     six_m_cohort_end = six_m_cohort_end['week_end']
#     six_m_time_divider = (six_m_cohort_end - six_m_cohort_start).days / 7
#
#     six_m_cohort_trials  = df[(df['trial_start_utc'] >= six_m_cohort_start) & (df['trial_start_utc'] <= six_m_cohort_end)]
#
#     six_m_total_trials = \
#             len(six_m_cohort_trials) / six_m_time_divider
#     six_m_survivors_trial = \
#             len(six_m_cohort_trials[~six_m_cohort_trials['canceled_during_trial']]) / six_m_time_divider
#     six_m_survivors_refund = len(six_m_cohort_trials[
#         (~six_m_cohort_trials['canceled_during_trial']) & 
#         (~six_m_cohort_trials['canceled_during_refund_period'])
#     ]) / six_m_time_divider
#
#     # All time average cohort
#     all_time_cohort_start = df['trial_start_utc'].min()
#     all_time_cohort_end = six_m_cohort_end
#     all_time_divider = (six_m_cohort_end - all_time_cohort_start).days / 7
#
#     all_time_cohort_trials = df[(df['trial_start_utc'] >= all_time_cohort_start) & (df['trial_start_utc'] <= all_time_cohort_end)]
#
#     all_time_total_trials = \
#             len(all_time_cohort_trials) / all_time_divider
#     all_time_survivors_trial = \
#             len(all_time_cohort_trials[~all_time_cohort_trials['canceled_during_trial']]) / all_time_divider
#     all_time_survivors_refund = len(all_time_cohort_trials[
#         (~all_time_cohort_trials['canceled_during_trial']) &
#         (~all_time_cohort_trials['canceled_during_refund_period'])
#     ]) / all_time_divider
#
#     # PLOT SINGLE CHART WITH GROUPED BARS
#     fig, ax = plt.subplots(1, 1, figsize=(22, 11))
#
#     # Data preparation
#     periods = [f'Last complete cohort\n(Trial from {complete_cohort_week["year_week"]})', \
#                f'Previous complete cohort\n(Trial from {prev_cohort_week["year_week"]})', '6 Month\nAverage', 'All Time\nAverage']
#
#     initial_trials = [last_total_trials, prev_total_trials, six_m_total_trials, all_time_total_trials]
#     survived_trial = [last_survived_trial, prev_survivors_trial, six_m_survivors_trial, all_time_survivors_trial]
#     full_members = [last_survived_refund, prev_survivors_refund, six_m_survivors_refund, all_time_survivors_refund]
#
#     # Bar positioning
#     x = np.arange(len(periods))
#     bar_width = 0.25
#
#     # Colors for the three categories
#     colors = ['darkgrey', 'red', 'darkgreen']
#
#     # Create bars
#     bars1 = ax.bar(x - bar_width, initial_trials, bar_width, 
#                    label='Initial Trials', color=colors[0], alpha=0.8)
#     bars2 = ax.bar(x, survived_trial, bar_width, 
#                    label='Survived Trial', color=colors[1], alpha=0.8)
#     bars3 = ax.bar(x + bar_width, full_members, bar_width, 
#                    label='Full Members', color=colors[2], alpha=0.8)
#
#     # Add value labels on bars
#     def add_value_labels(bars, values):
#         for bar, value in zip(bars, values):
#             height = bar.get_height()
#             ax.text(bar.get_x() + bar.get_width()/2., height + max(max(initial_trials), max(survived_trial), max(full_members)) * 0.01,
#                     f'{value:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
#
#     add_value_labels(bars1, initial_trials)
#     add_value_labels(bars2, survived_trial)
#     add_value_labels(bars3, full_members)
#
#     # Add percentage labels inside bars (conversion rates)
#     def add_percentage_labels(bars, values, base_values):
#         for i, (bar, value, base) in enumerate(zip(bars, values, base_values)):
#             if base > 0:
#                 percentage = (value / base) * 100
#                 height = bar.get_height()
#                 ax.text(bar.get_x() + bar.get_width()/2., height/2,
#                         f'{percentage:.1f}%', ha='center', va='center', 
#                         fontweight='bold', fontsize=10, color='white')
#
#     # Add percentages (all relative to initial trials)
#     add_percentage_labels(bars1, initial_trials, initial_trials)  # 100% for initial trials
#     add_percentage_labels(bars2, survived_trial, initial_trials)
#     add_percentage_labels(bars3, full_members, initial_trials)
#
#     # Customize chart
#     ax.set_xlabel('Time Periods', fontsize=14, fontweight='bold')
#     ax.set_ylabel('Number of Users', fontsize=14, fontweight='bold')
#     ax.set_title('CONVERSION FUNNEL COMPARISON ACROSS PERIODS', fontsize=18, fontweight='bold', pad=20)
#     ax.set_xticks(x)
#     ax.set_xticklabels(periods, fontsize=12)
#     ax.legend(fontsize=12, loc='upper left')
#     ax.grid(axis='y', alpha=0.3)
#
#     # Set y-axis limit
#     max_value = max(max(initial_trials), max(survived_trial), max(full_members))
#     ax.set_ylim(0, max_value * 1.15)
#
#     # Add summary text box
#     last_conversion_rate = (last_survived_refund / last_total_trials * 100) if last_total_trials > 0 else 0
#     prev_conversion_rate = (prev_survivors_refund / prev_total_trials * 100) if prev_total_trials > 0 else 0
#     six_m_conversion_rate = (six_m_survivors_refund / six_m_total_trials * 100) if six_m_total_trials > 0 else 0
#     all_time_conversion_rate = (all_time_survivors_refund / all_time_total_trials * 100) if all_time_total_trials > 0 else 0
#
#     summary_text = f"""CONVERSION RATES SUMMARY:
# Last Week: {last_conversion_rate:.1f}%
# Previous Week: {prev_conversion_rate:.1f}%
# 6 Month Avg: {six_m_conversion_rate:.1f}%
# All Time Avg: {all_time_conversion_rate:.1f}%"""
#
#     # ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=11, 
#     #         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
#
#     plt.tight_layout()
#
#     # Save
#     filename = f"conversion_funnel_comparison_2by2_{complete_cohort_week['year_week']}_{today_date.strftime('%Y-%m-%d')}.png"
#     plt.savefig(os.path.join(analysis_dir, filename), dpi=300, bbox_inches='tight')
#     print(f"Conversion funnel comparison 2x2 saved: {filename}")
#     plt.show()
#
#     cohort_dict = {'last_cohort': {
#             'values': [last_total_trials, last_survived_trial, last_survived_refund],
#             'conversion_rate': last_conversion_rate,
#             'week': complete_cohort_week['year_week']
#         }, 'prev_cohort': {
#             'values': [prev_total_trials, prev_survivors_trial, prev_survivors_refund],
#             'week': prev_cohort_week['year_week']
#         }, 'six_month_avg': {
#             'values': [six_m_total_trials, six_m_survivors_trial, six_m_survivors_refund],
#             'conversion_rate': six_m_conversion_rate
#         }, 'all_time_avg': {
#             'values': [all_time_total_trials, all_time_survivors_trial, all_time_survivors_refund],
#          'conversion_rate': all_time_conversion_rate
#         }
#     }
#
#
#
#     return cohort_dict
#
# last_cohort_comparison = plot_cohort_conversion_funnel_comparison(df, today_date, last_cohort_dict)
#
#
#
# # %%
# ########################################################################################
#
# # def plot_cohort_conversion_funnel_comparison(df, today_date, last_cohort_dict):
# #     """
# #     Plot a conversion funnel comparing different cohorts with 3 bars:
# #     1. Initial trials
# #     2. Survivors after trial period (not canceled during trial)
# #     3. Survivors after refund period (not canceled during refund)
# #     """
# #
# #     last_total_trials = last_cohort_dict['total_trials']
# #     last_survived_trial = last_cohort_dict['survived_trial']
# #     last_survived_refund = last_cohort_dict['survived_refund']
# #     last_conversion_trial_rate = last_cohort_dict['conversion_trial_rate']
# #     last_conversion_refund_rate = last_cohort_dict['conversion_refund_rate']
# #     last_total_drop_off = last_cohort_dict['total_drop_off']
# #     complete_cohort_week = get_specific_past_week(weeks_back=4, reference_date=today_date)
# #
# #     # PREVIOUS WEEK cohorts to compare
# #     prev_cohort_week = get_specific_past_week(weeks_back=5, reference_date=today_date)
# #     prev_cohort_trials = df[
# #         (df['trial_start_utc'] >= prev_cohort_week['week_start']) &
# #         (df['trial_start_utc'] <= prev_cohort_week['week_end'])
# #         ]
# #
# #     prev_total_trials = len(prev_cohort_trials)
# #     prev_survivors_trial = len(prev_cohort_trials[~prev_cohort_trials['canceled_during_trial']])
# #     prev_survivors_refund = len(prev_cohort_trials[
# #         (~prev_cohort_trials['canceled_during_trial']) & 
# #         (~prev_cohort_trials['canceled_during_refund_period'])
# #     ])
# #
# #
# #
# #     # 6 Months average cohort
# #     six_m_cohort_start = get_specific_past_week(weeks_back=24, reference_date=today_date)
# #     six_m_cohort_end = get_specific_past_week(weeks_back=4, reference_date=today_date)
# #
# #     six_m_cohort_start = six_m_cohort_start['week_start']
# #     six_m_cohort_end = six_m_cohort_end['week_end']
# #     six_m_time_divider = (six_m_cohort_end - six_m_cohort_start).days / 7
# #
# #     six_m_cohort_trials  = df[ 
# #         (df['trial_start_utc'] >= six_m_cohort_start) &
# #         (df['trial_start_utc'] <= six_m_cohort_end)
# #     ]
# #
# #     six_m_total_trials = \
# #             len(six_m_cohort_trials) / six_m_time_divider
# #     six_m_survivors_trial = \
# #             len(six_m_cohort_trials[~six_m_cohort_trials['canceled_during_trial']]) / six_m_time_divider
# #     six_m_survivors_refund = len(six_m_cohort_trials[
# #         (~six_m_cohort_trials['canceled_during_trial']) & 
# #         (~six_m_cohort_trials['canceled_during_refund_period'])
# #     ]) / six_m_time_divider
# #
# #
# #
# #     # All time average cohort
# #     all_time_cohort_start = df['trial_start_utc'].min()
# #     all_time_cohort_end = six_m_cohort_end
# #     all_time_divider = (six_m_cohort_end - all_time_cohort_start).days / 7
# #
# #     all_time_cohort_trials = df[
# #         (df['trial_start_utc'] >= all_time_cohort_start) &
# #         (df['trial_start_utc'] <= all_time_cohort_end)
# #     ]
# #
# #     all_time_total_trials = \
# #             len(all_time_cohort_trials) / all_time_divider
# #     all_time_survivors_trial = \
# #             len(all_time_cohort_trials[~all_time_cohort_trials['canceled_during_trial']]) / all_time_divider
# #     all_time_survivors_refund = len(all_time_cohort_trials[
# #         (~all_time_cohort_trials['canceled_during_trial']) &
# #         (~all_time_cohort_trials['canceled_during_refund_period'])
# #     ]) / all_time_divider
# #
# #
# #     # Prepare data for all comparisons
# #     last_values = [last_total_trials, last_survived_trial, last_survived_refund]
# #     prev_values = [prev_total_trials, prev_survivors_trial, prev_survivors_refund]
# #     six_m_values = [six_m_total_trials, six_m_survivors_trial, six_m_survivors_refund]
# #     all_time_values = [all_time_total_trials, all_time_survivors_trial, all_time_survivors_refund]
# #
# #
# #     # PLOT COMPARISON CHART
# #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
# #     bar_width = 0.35
# #     categories = ['Initial Trials', 'Survived Trial Period', 'Full Members']
# #     x_pos = np.arange(len(categories))
# #
# #     colors1 = ['gray', 'red', 'darkgreen']
# #     colors2 = ['lightgray', 'orange', 'green']
# #
# #
# #     def add_bars_and_labels(ax, values1, values2, label1, label2, color1, color2):
# #         """Helper function to add bars and labels to a subplot"""
# #         bars1 = ax.bar(x_pos - bar_width/2, values1, bar_width, 
# #                       label=label1, color=color1, alpha=0.8)
# #         bars2 = ax.bar(x_pos + bar_width/2, values2, bar_width, 
# #                       label=label2, color=color2, alpha=0.8)
# #
# #         # Add numbers and percentages for first set of bars
# #         for i, (bar, value) in enumerate(zip(bars1, values1)):
# #             height = bar.get_height()
# #             ax.text(bar.get_x() + bar.get_width()/2., height + max(max(values1), max(values2)) * 0.02,
# #                     f'{value:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
# #
# #             # Add percentage relative to initial trials
# #             if values1[0] > 0:
# #                 percentage = (value / values1[0]) * 100
# #                 ax.text(bar.get_x() + bar.get_width()/2., height/2,
# #                         f'{percentage:.1f}%', ha='center', va='center', 
# #                         fontweight='bold', fontsize=9, color='white')
# #
# #         # Add numbers and percentages for second set of bars
# #         for i, (bar, value) in enumerate(zip(bars2, values2)):
# #             height = bar.get_height()
# #             ax.text(bar.get_x() + bar.get_width()/2., height + max(max(values1), max(values2)) * 0.02,
# #                     f'{value:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
# #
# #             # Add percentage relative to initial trials
# #             if values2[0] > 0:
# #                 percentage = (value / values2[0]) * 100
# #                 ax.text(bar.get_x() + bar.get_width()/2., height/2,
# #                         f'{percentage:.1f}%', ha='center', va='center', 
# #                         fontweight='bold', fontsize=9, color='white')
# #
# #         # Configure subplot
# #         ax.set_xticks(x_pos)
# #         ax.set_xticklabels(categories)
# #         ax.set_ylabel('# of users', fontsize=12)
# #         ax.legend(fontsize=10)
# #         ax.grid(axis='y', alpha=0.3)
# #         ax.set_ylim(0, max(max(values1), max(values2)) * 1.15)
# #
# #     # CHART 1: Last Week vs Previous Week
# #     add_bars_and_labels(ax1, last_values, prev_values, 
# #                        f'Last Week ({complete_cohort_week["year_week"]})', 
# #                        f'Previous Week ({prev_cohort_week["year_week"]})',
# #                        colors1, colors2)
# #     ax1.set_title('Last Week vs Previous Week', fontsize=14, fontweight='bold')
# #
# #     # CHART 2: Last Week vs 6 Month Average
# #     add_bars_and_labels(ax2, last_values, six_m_values,
# #                        f'Last Week ({complete_cohort_week["year_week"]})', 
# #                        '6 Month Average',
# #                        colors1, colors2)
# #     ax2.set_title('Last Week vs 6 Month Average', fontsize=14, fontweight='bold')
# #
# #     # CHART 3: Last Week vs All Time Average
# #     add_bars_and_labels(ax3, last_values, all_time_values,
# #                        f'Last Week ({complete_cohort_week["year_week"]})', 
# #                        'All Time Average',
# #                        colors1, colors2)
# #     ax3.set_title('Last Week vs All Time Average', fontsize=14, fontweight='bold')
# #
# #     # Main title
# #     fig.suptitle('CONVERSION FUNNEL COMPARISONS', fontsize=18, fontweight='bold', y=0.98)
# #
# #     # Add overall summary text
# #     last_conversion_rate = (last_survived_refund / last_total_trials * 100) if last_total_trials > 0 else 0
# #     six_m_conversion_rate = (six_m_survivors_refund / six_m_total_trials * 100) if six_m_total_trials > 0 else 0
# #     all_time_conversion_rate = (all_time_survivors_refund / all_time_total_trials * 100) if all_time_total_trials > 0 else 0
# #
# #     summary_text = f"""CONVERSION RATES SUMMARY:
# # Last Week: {last_conversion_rate:.1f}%
# # 6 Month Avg: {six_m_conversion_rate:.1f}%
# # All Time Avg: {all_time_conversion_rate:.1f}%"""
# #
# #     # fig.text(0.02, 0.15, summary_text, fontsize=11, 
# #     #          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
# #
# #     plt.tight_layout()
# #
# #     # Save
# #     filename = f"conversion_funnel_comparison_2by2_{complete_cohort_week['year_week']}_{today_date.strftime('%Y-%m-%d')}.png"
# #     plt.savefig(os.path.join(analysis_dir, filename), dpi=300, bbox_inches='tight')
# #     print(f"Conversion funnel comparison 2x2 saved: {filename}")
# #     plt.show()
# #
# #     return {
# #         'last_cohort': {
# #             'values': last_values,
# #             'conversion_rate': last_conversion_rate,
# #             'week': complete_cohort_week['year_week']
# #         },
# #         'prev_cohort': {
# #             'values': prev_values,
# #             'week': prev_cohort_week['year_week']
# #         },
# #         'six_month_avg': {
# #             'values': six_m_values,
# #             'conversion_rate': six_m_conversion_rate
# #         },
# #         'all_time_avg': {
# #             'values': all_time_values,
# #             'conversion_rate': all_time_conversion_rate
# #         }
# #     }
# #
# # plot_cohort_conversion_funnel_comparison(df, today_date, last_cohort_dict)
# #
# # %%
# ######################################################################################
# todo = "TODO"
#
#
# from reportlab.lib.pagesizes import A3, landscape
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib.units import cm
# from reportlab.lib import colors
# from reportlab.lib.enums import TA_CENTER, TA_LEFT
# import glob
#
#
# def create_analysis_report_pdf(today_date, analysis_dir='analysis'):
#     """
#     Create a comprehensive PDF report page by page
#     """
#
#     # === PDF CONFIGURATION ===
#     week_info = get_specific_past_week(weeks_back=1, reference_date=today_date)
#     pdf_filename = f'ANALYSIS_REPORT_{today_date.strftime("%Y-%m-%d")}.pdf'
#     pdf_path = os.path.join(analysis_dir, pdf_filename)
#
#     # Create PDF document in LANDSCAPE A3 (plus d'espace)
#     doc = SimpleDocTemplate(
#         pdf_path,
#         pagesize=landscape(A3),  # A3 au lieu de A4
#         rightMargin=1*cm,
#         leftMargin=1*cm,
#         topMargin=1*cm,
#         bottomMargin=1*cm
#     )
#
#     # === STYLES ===
#     styles = getSampleStyleSheet()
#
#     # Main title style
#     title_style = ParagraphStyle(
#         'CustomTitle',
#         parent=styles['Heading1'],
#         fontSize=36,
#         spaceAfter=20,
#         alignment=TA_CENTER,
#         textColor=colors.black
#     )
#
#     # Subtitle style
#     subtitle_style = ParagraphStyle(
#         'CustomSubtitle',
#         parent=styles['Heading2'],
#         fontSize=24,  # Plus grand pour A3
#         spaceAfter=20,
#         alignment=TA_CENTER,
#         textColor=colors.black
#     )
#
#     small_subtitle_style = ParagraphStyle(
#         'CustomSubtitle',
#         parent=styles['Normal'],
#         fontSize=18,
#         spaceAfter=20,
#         alignment=TA_CENTER,
#         textColor=colors.black
#     )
#
#     # Section style
#     section_style = ParagraphStyle(
#         'SectionHeading',
#         parent=styles['Heading2'],
#         fontSize=20,
#         spaceAfter=20,
#         textColor=colors.darkblue
#     )
#
#     # Normal text style with spacing
#     normal_style = ParagraphStyle(
#         'CustomNormal',
#         parent=styles['Normal'],
#         fontSize=18,
#         spaceAfter=15,
#         alignment=TA_LEFT
#     ) 
#
#     # Normal text style with spacing
#     big_style = ParagraphStyle(
#         'CustomNormal',
#         parent=styles['Normal'],
#         fontSize=22,
#         spaceAfter=15,
#         alignment=TA_LEFT
#     ) 
#
#     # Style for important metrics
#     metrics_style = ParagraphStyle(
#         'MetricsStyle',
#         parent=styles['Normal'],
#         fontSize=18,
#         spaceAfter=12,
#         textColor=colors.darkgreen,
#         leftIndent=20
#     )
#
#     # === FONCTION HELPER POUR CENTRER LES IMAGES ===
#     def create_centered_image(image_path, width, height, title):
#         """Create a centered image with title"""
#         elements = []
#
#         # Titre de la section
#         elements.append(Paragraph(title, subtitle_style))
#         elements.append(Spacer(1, 0.8*cm))
#
#         # Image centrée dans un tableau
#         centered_image = Image(image_path, width=width, height=height)
#         image_table = Table([[centered_image]], colWidths=[40*cm])
#         image_table.setStyle(TableStyle([
#             ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
#             ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
#         ]))
#
#         elements.append(image_table)
#         return elements
#
#     # === PDF CONTENT ===
#     story = []
#
#     # ============================================================================
#     # === PAGE 1: TITLE PAGE ===
#     # ============================================================================
#     story.append(Spacer(1, 3*cm))
#     story.append(Paragraph("DISHPATCH WEEKLY ANALYSIS REPORT", title_style))
#     story.append(Paragraph(f"{today_date.strftime('%B %d, %Y')}", subtitle_style))
#     story.append(Paragraph(f"(Last week: W{week_info['week_number']} - Monday {week_info['week_start'].strftime('%d-%m')} to Sunday {week_info['week_end'].strftime('%d-%m')})", small_subtitle_style))
#     story.append(Spacer(1, 1*cm))
#
#     # Executive summary on title page
#     story.append(Paragraph("EXECUTIVE SUMMARY", section_style))
#
#     story.append(Paragraph(f"Currently Active full member ever: <b>{dict_full_members['active']}</b>", big_style))
#     story.append(Paragraph(f"<i>Active Full Member 1st year: <b>{renewal_dict['active_1st_year']}</b> - 2nd year: <b>{renewal_dict['active_2nd_year']}</b> - 3rd year: <b>{renewal_dict['active_3rd_year']}</b></i>", normal_style))
#     story.append(Paragraph(f"<i>Most full member ever: {dict_full_members['max_full_members']} on {dict_full_members['max_date'].strftime('%d-%m-%Y')}</i>", normal_style))
#     story.append(Spacer(1, 0.5*cm))
#
#     story.append(Paragraph(f"Renewal Rate: <b>{todo}%</b>", big_style))
#     story.append(Paragraph(f"<i>Renewal rate from 1st year to 2nd year <b>{renewal_dict['y1_to_y2_rate']}%</b> from 2nd year to 3rd year <b>{renewal_dict['y2_to_y3_rate']}%.</b></i>", normal_style))
#     story.append(Spacer(1, 0.5*cm))
#
#     story.append(Paragraph(f"New trial last week: <b>{new_trial_last_week}</b><i> (previous week: {new_trial_prev_week})</i>", big_style))
#     story.append(Paragraph(f"New full member last week: <b>{last_week_full_members}</b><i> (previous week: {prev_week_full_members})</i>", big_style))
#     story.append(Paragraph(f"Churn full member last week: <b>{last_week_churned_members}</b><i> (previous week: {prev_week_churned_members})</i>", big_style))
#     story.append(Spacer(1, 0.5*cm))
#
#     story.append(Paragraph(f"Conversion Rate (from Trial to Full Member): <b>{conversion_rate_dict['conversion_rate']}%</b>", big_style))
#     story.append(Paragraph(f"<i>To be a full member a user must complete their trial, not request a refund, and not be gifted. (refund period {REFUND_PERIOD_DAYS} days)</i>", normal_style))
#     story.append(Spacer(1, 0.5*cm))
#
#     story.append(PageBreak())
#
#     # ============================================================================
#     # === PAGE 2: Trial 8 weeks (Carré) ===
#     # ============================================================================
#     try:
#         trial_files = glob.glob(os.path.join(analysis_dir, "weekly_trials_8_weeks_*.png"))
#         if trial_files:
#             latest_trial = max(trial_files, key=os.path.getctime)
#             # Carré centré, taille maximale pour A3 paysage (plus grand)
#             story.extend(create_centered_image(latest_trial, 20*cm, 20*cm, "TRIAL EACH WEEK"))
#         else:
#             story.append(Paragraph("[8 WEEKS TRIAL CHART NOT AVAILABLE]", normal_style))
#     except Exception as e:
#         story.append(Paragraph(f"[CHART NOT AVAILABLE: {e}]", normal_style))
#
#     story.append(PageBreak())
#
#     # ============================================================================
#     # === PAGE 3: Trial All Time (Rectangle) ===
#     # ============================================================================
#     try:
#         trial_all_files = glob.glob(os.path.join(analysis_dir, "weekly_trials_all_time_*.png"))
#         if trial_all_files:
#             latest_trial_all = max(trial_all_files, key=os.path.getctime)
#             # Rectangle centré, utilise presque toute la largeur A3 (plus grand)
#             story.extend(create_centered_image(latest_trial_all, 38*cm, 18*cm, "TRIAL EACH WEEK"))
#         else:
#             story.append(Paragraph("TRIAL EACH WEEK - ALL TIME", subtitle_style))
#             story.append(Paragraph("[ALL TRIAL CHART NOT AVAILABLE]", normal_style))
#     except Exception as e:
#         story.append(Paragraph("TRIAL EACH WEEK - ALL TIME", subtitle_style))
#         story.append(Paragraph(f"[CHART NOT AVAILABLE: {e}]", normal_style))
#
#     story.append(PageBreak())
#
#     # ============================================================================
#     # === PAGE 4: Full Member Flow 8 weeks (Carré) ===
#     # ============================================================================
#     try:
#         flow_files = glob.glob(os.path.join(analysis_dir, "weekly_flow_8_weeks_*.png"))
#         if flow_files:
#             latest_flow = max(flow_files, key=os.path.getctime)
#             story.extend(create_centered_image(latest_flow, 20*cm, 20*cm, "FULL MEMBER FLOW"))
#         else:
#             story.append(Paragraph("[8 WEEKS FLOW CHART NOT AVAILABLE]", normal_style))
#     except Exception as e:
#         story.append(Paragraph(f"[CHART NOT AVAILABLE: {e}]", normal_style))
#
#     story.append(PageBreak())
#
#     # ============================================================================
#     # === PAGE 5: Full Member Flow All Time (Rectangle) ===
#     # ============================================================================
#     try:
#         flow_all_files = glob.glob(os.path.join(analysis_dir, "weekly_flow_all_time_*.png"))
#         if flow_all_files:
#             latest_flow_all = max(flow_all_files, key=os.path.getctime)
#             story.extend(create_centered_image(latest_flow_all, 38*cm, 18*cm, "FULL MEMBER FLOW"))
#         else:
#             story.append(Paragraph("FULL MEMBER FLOW - ALL TIME", subtitle_style))
#             story.append(Paragraph("[ALL FLOW CHART NOT AVAILABLE]", normal_style))
#     except Exception as e:
#         story.append(Paragraph("FULL MEMBER FLOW - ALL TIME", subtitle_style))
#         story.append(Paragraph(f"[CHART NOT AVAILABLE: {e}]", normal_style))
#
#     story.append(PageBreak())
#
#     # ============================================================================
#     # === PAGE 6: Cohort Conversion (Carré) ===
#     # ============================================================================
#     try:
#         cohort_files = glob.glob(os.path.join(analysis_dir, "Conversion_funnel_*.png"))
#         if cohort_files:
#             latest_cohort = max(cohort_files, key=os.path.getctime)
#             story.extend(create_centered_image(latest_cohort, 20*cm, 20*cm, "COHORT CONVERSION"))
#         else:
#             story.append(Paragraph("[COHORT CONVERSION CHART NOT AVAILABLE]", normal_style))
#     except Exception as e:
#         story.append(Paragraph(f"[CHART NOT AVAILABLE: {e}]", normal_style))
#
#     story.append(PageBreak())
#
#     # ============================================================================
#     # === PAGE 7: Cohort Comparison (Rectangle) ===
#     # ============================================================================
#     try:
#         cohort_comparison_files = glob.glob(os.path.join(analysis_dir, "conversion_funnel_comparison_*.png"))
#         if cohort_comparison_files:
#             latest_cohort_comparison = max(cohort_comparison_files, key=os.path.getctime)
#             story.extend(create_centered_image(latest_cohort_comparison, 38*cm, 18*cm, "COHORT CONVERSION"))
#         else:
#             story.append(Paragraph("[COHORT COMPARISON CHART NOT AVAILABLE]", normal_style))
#     except Exception as e:
#         story.append(Paragraph("COHORT CONVERSION COMPARISON", subtitle_style))
#         story.append(Paragraph(f"[CHART NOT AVAILABLE: {e}]", normal_style))
#
#     # === GENERATE PDF ===
#     try:
#         doc.build(story)
#         print(f"\n✅ PDF Report generated successfully!")
#         print(f"📄 File saved: {pdf_path}")
#         return pdf_path
#     except Exception as e:
#         print(f"❌ Error generating PDF: {e}")
#         return None
#
#
# def generate_complete_report():
#     """
#     Generate the complete manual report
#     """
#     try:
#         pdf_path = create_analysis_report_pdf(today_date, analysis_dir)
#         return pdf_path
#     except Exception as e:
#         print(f"❌ Error in generate_complete_report: {e}")
#         return None
#
#
# # === EXECUTION ===
# print(f"\n📄 Generating PDF report...")
# complete_pdf = generate_complete_report()
#
# if complete_pdf:
#     print(f"\n✅ Report generation complete!")
#     print(f"📁 Check your {analysis_dir} folder for the PDF file.")
# else:
#     print(f"\n⚠️ PDF generation failed.")
#
#
#
# test_df = df[df['trial_start_utc'].isna()].copy()
# print(len(test_df), 'subscriptions without trial')
# test_df['customer_id'].value_counts().head(10)
# test_df['customer_id'].value_counts().plot(kind='bar', figsize=(10, 5))
# plt.show()
# print(test_df['customer_id'].value_counts(), 'customer  counts')
# test_df = test_df[test_df['is_full_member']]
# print(len(test_df), 'full members without trial')
#
# def renewal_rate(df):
#     first_year_sub = df[(df['trial_start_utc'].notna() & (df['is_full_member']))]
#     print(f"first_year_sub: {len(first_year_sub)}")
#
#
# renewal_rate(df)
# #
# #
# # ######################################################################################
# # # ANALYSE DES SUBSCRIPTIONS SANS TRIAL - DÉCOUVERTE DES PATTERNS
# # ######################################################################################
# #
# # def analyze_subscriptions_without_trial(df, today_date):
# #     """
# #     Analyse approfondie des subscriptions sans trial_start_utc
# #     """
# #     print("🔍 ANALYSE DES SUBSCRIPTIONS SANS TRIAL")
# #     print("=" * 60)
# #
# #     # Séparer les deux groupes
# #     with_trial = df[df['trial_start_utc'].notna()].copy()
# #     without_trial = df[df['trial_start_utc'].isna()].copy()
# #
# #     print(f"📊 RÉPARTITION GÉNÉRALE:")
# #     print(f"   Avec trial: {len(with_trial):,} ({len(with_trial)/len(df)*100:.1f}%)")
# #     print(f"   Sans trial: {len(without_trial):,} ({len(without_trial)/len(df)*100:.1f}%)")
# #
# #     # === ANALYSE TEMPORELLE ===
# #     print(f"\n📅 ANALYSE TEMPORELLE DES SUBSCRIPTIONS SANS TRIAL:")
# #
# #     # Distribution par mois de création
# #     without_trial['created_month'] = without_trial['created_utc'].dt.to_period('M')
# #     monthly_counts = without_trial['created_month'].value_counts().sort_index()
# #
# #     print(f"   Distribution par mois:")
# #     for month, count in monthly_counts.head(10).items():
# #         percentage = count / len(without_trial) * 100
# #         print(f"      {month}: {count} subscriptions ({percentage:.1f}%)")
# #
# #     # === COMPARAISON DES CARACTÉRISTIQUES ===
# #     print(f"\n🔍 COMPARAISON DES CARACTÉRISTIQUES:")
# #
# #     # Statut actuel
# #     print(f"   STATUTS ACTUELS:")
# #     for status in ['active', 'canceled', 'trialing', 'past_due']:
# #         with_trial_count = len(with_trial[with_trial['status'] == status])
# #         without_trial_count = len(without_trial[without_trial['status'] == status])
# #
# #         with_trial_pct = with_trial_count / len(with_trial) * 100 if len(with_trial) > 0 else 0
# #         without_trial_pct = without_trial_count / len(without_trial) * 100 if len(without_trial) > 0 else 0
# #
# #         print(f"      {status}:")
# #         print(f"         Avec trial: {with_trial_count} ({with_trial_pct:.1f}%)")
# #         print(f"         Sans trial: {without_trial_count} ({without_trial_pct:.1f}%)")
# #
# #     # Full members
# #     with_trial_full = len(with_trial[with_trial['is_full_member'] == True])
# #     without_trial_full = len(without_trial[without_trial['is_full_member'] == True])
# #
# #     print(f"   FULL MEMBERS:")
# #     print(f"      Avec trial: {with_trial_full} ({with_trial_full/len(with_trial)*100:.1f}%)")
# #     print(f"      Sans trial: {without_trial_full} ({without_trial_full/len(without_trial)*100:.1f}%)")
# #
# #     # === ANALYSE DES DURÉES ===
# #     print(f"\n⏱️ ANALYSE DES DURÉES:")
# #
# #     # Durée moyenne depuis création
# #     avg_duration_with = with_trial['days_since_creation'].mean()
# #     avg_duration_without = without_trial['days_since_creation'].mean()
# #
# #     print(f"   Durée moyenne depuis création:")
# #     print(f"      Avec trial: {avg_duration_with:.0f} jours")
# #     print(f"      Sans trial: {avg_duration_without:.0f} jours")
# #
# #     # Durée de la période actuelle
# #     avg_period_with = with_trial['current_period_duration'].mean()
# #     avg_period_without = without_trial['current_period_duration'].mean()
# #
# #     print(f"   Durée moyenne période actuelle:")
# #     print(f"      Avec trial: {avg_period_with:.0f} jours")
# #     print(f"      Sans trial: {avg_period_without:.0f} jours")
# #
# #     return {
# #         'with_trial_count': len(with_trial),
# #         'without_trial_count': len(without_trial),
# #         'monthly_distribution': monthly_counts,
# #         'avg_duration_with': avg_duration_with,
# #         'avg_duration_without': avg_duration_without
# #     }
# #
# #
# # ######################################################################################
# # # ANALYSE DES PATTERNS TEMPORELS DÉTAILLÉS
# # ######################################################################################
# #
# # def analyze_temporal_patterns(df):
# #     """
# #     Analyse les patterns temporels pour comprendre l'évolution
# #     """
# #     print("📈 ANALYSE DES PATTERNS TEMPORELS")
# #     print("-" * 50)
# #
# #     without_trial = df[df['trial_start_utc'].isna()].copy()
# #
# #     # === ÉVOLUTION MENSUELLE ===
# #     without_trial['created_month'] = without_trial['created_utc'].dt.to_period('M')
# #     total_by_month = df.groupby(df['created_utc'].dt.to_period('M')).size()
# #     without_trial_by_month = without_trial.groupby('created_month').size()
# #
# #     # Pourcentage sans trial par mois
# #     percentage_by_month = (without_trial_by_month / total_by_month * 100).fillna(0)
# #
# #     print(f"   ÉVOLUTION DU % SANS TRIAL PAR MOIS:")
# #     for month in percentage_by_month.index[-12:]:  # 12 derniers mois
# #         if month in total_by_month.index:
# #             total = total_by_month[month]
# #             without = without_trial_by_month.get(month, 0)
# #             pct = percentage_by_month[month]
# #             print(f"      {month}: {without}/{total} ({pct:.1f}%)")
# #
# #     # === ANALYSE PAR SEMAINE ===
# #     print(f"\n📊 ANALYSE PAR SEMAINE (6 dernières semaines):")
# #
# #     without_trial['created_week'] = without_trial['created_utc'].dt.to_period('W')
# #     total_by_week = df.groupby(df['created_utc'].dt.to_period('W')).size()
# #     without_trial_by_week = without_trial.groupby('created_week').size()
# #
# #     percentage_by_week = (without_trial_by_week / total_by_week * 100).fillna(0)
# #
# #     for week in percentage_by_week.index[-6:]:  # 6 dernières semaines
# #         if week in total_by_week.index:
# #             total = total_by_week[week]
# #             without = without_trial_by_week.get(week, 0)
# #             pct = percentage_by_week[week]
# #             print(f"      Semaine {week}: {without}/{total} ({pct:.1f}%)")
# #
# #     # === RECHERCHE DE CHANGEMENTS SIGNIFICATIFS ===
# #     print(f"\n🔍 DÉTECTION DE CHANGEMENTS SIGNIFICATIFS:")
# #
# #     # Moyenne des 3 premiers mois vs 3 derniers mois
# #     first_3_months = percentage_by_month.head(3).mean()
# #     last_3_months = percentage_by_month.tail(3).mean()
# #
# #     print(f"   3 premiers mois (moyenne): {first_3_months:.1f}% sans trial")
# #     print(f"   3 derniers mois (moyenne): {last_3_months:.1f}% sans trial")
# #     print(f"   Évolution: {last_3_months - first_3_months:+.1f} points")
# #
# #     # Recherche de pics/creux
# #     max_month = percentage_by_month.idxmax()
# #     min_month = percentage_by_month.idxmin()
# #
# #     print(f"   Pic maximum: {max_month} ({percentage_by_month[max_month]:.1f}%)")
# #     print(f"   Minimum: {min_month} ({percentage_by_month[min_month]:.1f}%)")
# #
# #     return {
# #         'percentage_by_month': percentage_by_month,
# #         'first_3_months_avg': first_3_months,
# #         'last_3_months_avg': last_3_months,
# #         'max_month': max_month,
# #         'min_month': min_month
# #     }
# #
# #
# # ######################################################################################
# # # HYPOTHÈSES SUR L'ORIGINE DES SUBSCRIPTIONS SANS TRIAL
# # ######################################################################################
# #
# # def hypothesize_no_trial_origins(df):
# #     """
# #     Formule des hypothèses sur l'origine des subscriptions sans trial
# #     """
# #     print("💡 HYPOTHÈSES SUR L'ORIGINE DES SUBSCRIPTIONS SANS TRIAL")
# #     print("-" * 60)
# #
# #     without_trial = df[df['trial_start_utc'].isna()].copy()
# #
# #     # === HYPOTHÈSE 1: RENOUVELLEMENTS AUTOMATIQUES ===
# #     print("🔄 HYPOTHÈSE 1: RENOUVELLEMENTS AUTOMATIQUES STRIPE")
# #
# #     # Clients avec plusieurs subscriptions
# #     customer_counts = without_trial['customer_id'].value_counts()
# #     customers_multiple_subs = customer_counts[customer_counts > 1]
# #
# #     print(f"   Clients sans trial ayant plusieurs subscriptions: {len(customers_multiple_subs)}")
# #
# #     if len(customers_multiple_subs) > 0:
# #         print(f"   Exemple de clients avec multiples subscriptions sans trial:")
# #         for customer_id, count in customers_multiple_subs.head(3).items():
# #             customer_subs = without_trial[without_trial['customer_id'] == customer_id]
# #             dates = customer_subs['created_utc'].dt.strftime('%Y-%m-%d').tolist()
# #             print(f"      Client {customer_id}: {count} subs ({', '.join(dates)})")
# #
# #     # === HYPOTHÈSE 2: CONVERSIONS DIRECTES (SANS TRIAL) ===
# #     print(f"\n💳 HYPOTHÈSE 2: CONVERSIONS DIRECTES")
# #
# #     # Durée de période très courte = peut-être pas de trial
# #     short_periods = without_trial[without_trial['current_period_duration'] < 30]
# #     print(f"   Subscriptions sans trial avec période < 30j: {len(short_periods)}")
# #
# #     # Comparaison des durées de période
# #     with_trial = df[df['trial_start_utc'].notna()]
# #
# #     avg_period_with_trial = with_trial['current_period_duration'].mean()
# #     avg_period_without_trial = without_trial['current_period_duration'].mean()
# #
# #     print(f"   Durée moyenne période - avec trial: {avg_period_with_trial:.0f} jours")
# #     print(f"   Durée moyenne période - sans trial: {avg_period_without_trial:.0f} jours")
# #
# #     # === HYPOTHÈSE 3: MIGRATIONS/UPGRADES ===
# #     print(f"\n⬆️ HYPOTHÈSE 3: MIGRATIONS OU UPGRADES")
# #
# #     # Clients sans trial mais qui sont full members rapidement
# #     quick_full_members = without_trial[
# #         (without_trial['is_full_member'] == True) &
# #         (without_trial['days_since_creation'] < 30)
# #     ]
# #
# #     print(f"   Full members sans trial en < 30 jours: {len(quick_full_members)}")
# #
# #     # === HYPOTHÈSE 4: CHANGEMENTS DE POLITIQUE ===
# #     print(f"\n📋 HYPOTHÈSE 4: CHANGEMENTS DE POLITIQUE")
# #
# #     # Analyse chronologique pour détecter des changements de politique
# #     without_trial['created_month'] = without_trial['created_utc'].dt.to_period('M')
# #     monthly_counts = without_trial['created_month'].value_counts().sort_index()
# #
# #     # Détection de croissance soudaine
# #     for i in range(1, len(monthly_counts)):
# #         current_month = monthly_counts.iloc[i]
# #         previous_month = monthly_counts.iloc[i-1]
# #
# #         if current_month > previous_month * 2:  # Doublement ou plus
# #             month_name = monthly_counts.index[i]
# #             print(f"   Pic détecté en {month_name}: {current_month} (vs {previous_month} le mois précédent)")
# #
# #     return {
# #         'customers_multiple_subs': len(customers_multiple_subs),
# #         'short_periods': len(short_periods),
# #         'quick_full_members': len(quick_full_members),
# #         'avg_period_comparison': {
# #             'with_trial': avg_period_with_trial,
# #             'without_trial': avg_period_without_trial
# #         }
# #     }
# #
# #
# # ######################################################################################
# # # ANALYSE COMPARATIVE DES MÉTADONNÉES
# # ######################################################################################
# #
# # def analyze_metadata_differences(df):
# #     """
# #     Compare les métadonnées entre subscriptions avec et sans trial
# #     """
# #     print("🔍 ANALYSE DES MÉTADONNÉES")
# #     print("-" * 40)
# #
# #     with_trial = df[df['trial_start_utc'].notna()]
# #     without_trial = df[df['trial_start_utc'].isna()]
# #
# #     # === COMPARAISON DES CHAMPS DISPONIBLES ===
# #     print("📊 COMPARAISON DES CHAMPS:")
# #
# #     fields_to_compare = [
# #         'start_utc', 'current_period_start_utc', 'current_period_end_utc',
# #         'trial_end_utc', 'is_gifted_member', 'status'
# #     ]
# #
# #     for field in fields_to_compare:
# #         if field in df.columns:
# #             with_trial_na = with_trial[field].isna().sum()
# #             without_trial_na = without_trial[field].isna().sum()
# #
# #             print(f"   {field}:")
# #             print(f"      Avec trial - valeurs manquantes: {with_trial_na}/{len(with_trial)} ({with_trial_na/len(with_trial)*100:.1f}%)")
# #             print(f"      Sans trial - valeurs manquantes: {without_trial_na}/{len(without_trial)} ({without_trial_na/len(without_trial)*100:.1f}%)")
# #
# #     # === ANALYSE DES ÉCARTS TEMPORELS ===
# #     print(f"\n⏱️ ANALYSE DES ÉCARTS TEMPORELS:")
# #
# #     # Écart entre created_utc et start_utc
# #     if 'start_utc' in df.columns:
# #         with_trial['void_duration'] = (with_trial['start_utc'] - with_trial['created_utc']).dt.total_seconds() / 3600  # en heures
# #         without_trial['void_duration'] = (without_trial['start_utc'] - without_trial['created_utc']).dt.total_seconds() / 3600
# #
# #         avg_void_with = with_trial['void_duration'].mean()
# #         avg_void_without = without_trial['void_duration'].mean()
# #
# #         print(f"   Écart création → début (moyenne):")
# #         print(f"      Avec trial: {avg_void_with:.1f} heures")
# #         print(f"      Sans trial: {avg_void_without:.1f} heures")
# #
# #     # === PATTERNS DANS LES STATUTS ===
# #     print(f"\n📋 PATTERNS DANS LES STATUTS:")
# #
# #     status_comparison = pd.crosstab(
# #         df['trial_start_utc'].notna(), 
# #         df['status'], 
# #         normalize='index'
# #     ) * 100
# #
# #     print("   Distribution des statuts (%):")
# #     print(status_comparison.round(1))
# #
# #
# # ######################################################################################
# # # FONCTION PRINCIPALE D'ANALYSE
# # ######################################################################################
# #
# # def comprehensive_no_trial_analysis(df, today_date):
# #     """
# #     Analyse complète des subscriptions sans trial
# #     """
# #     print("🎯 ANALYSE COMPLÈTE - SUBSCRIPTIONS SANS TRIAL")
# #     print("=" * 70)
# #
# #     # Exécute toutes les analyses
# #     basic_analysis = analyze_subscriptions_without_trial(df, today_date)
# #     temporal_patterns = analyze_temporal_patterns(df)
# #     hypotheses = hypothesize_no_trial_origins(df)
# #     metadata_analysis = analyze_metadata_differences(df)
# #
# #     # === CONCLUSIONS ET RECOMMANDATIONS ===
# #     print(f"\n🎯 CONCLUSIONS ET RECOMMANDATIONS:")
# #     print("-" * 50)
# #
# #     without_trial_pct = basic_analysis['without_trial_count'] / (basic_analysis['with_trial_count'] + basic_analysis['without_trial_count']) * 100
# #
# #     print(f"📊 RÉSUMÉ EXÉCUTIF:")
# #     print(f"   • {basic_analysis['without_trial_count']:,} subscriptions sans trial ({without_trial_pct:.1f}%)")
# #     print(f"   • Évolution récente: {temporal_patterns['first_3_months_avg']:.1f}% → {temporal_patterns['last_3_months_avg']:.1f}%")
# #     print(f"   • Pic maximum: {temporal_patterns['max_month']} ({temporal_patterns['percentage_by_month'][temporal_patterns['max_month']]:.1f}%)")
# #
# #     print(f"\n💡 HYPOTHÈSES PRINCIPALES:")
# #     if hypotheses['customers_multiple_subs'] > 100:
# #         print(f"   ✅ Forte probabilité de renouvellements automatiques")
# #     if abs(hypotheses['avg_period_comparison']['with_trial'] - hypotheses['avg_period_comparison']['without_trial']) > 30:
# #         print(f"   ✅ Types de subscriptions différents (durées différentes)")
# #     if temporal_patterns['last_3_months_avg'] > temporal_patterns['first_3_months_avg'] + 10:
# #         print(f"   ✅ Changement de politique récent possible")
# #
# #     print(f"\n🔍 ACTIONS RECOMMANDÉES:")
# #     print(f"   1. Vérifier la configuration Stripe pour les renouvellements")
# #     print(f"   2. Analyser les webhooks pour comprendre la création sans trial")
# #     print(f"   3. Examiner les logs autour de {temporal_patterns['max_month']}")
# #     print(f"   4. Valider si les subscriptions sans trial sont intentionnelles")
# #
# #     return {
# #         'basic_analysis': basic_analysis,
# #         'temporal_patterns': temporal_patterns,
# #         'hypotheses': hypotheses,
# #         'metadata_analysis': metadata_analysis
# #     }
# #
# #
# # # === EXÉCUTION DE L'ANALYSE COMPLÈTE ===
# # print("🚀 LANCEMENT DE L'ANALYSE DES SUBSCRIPTIONS SANS TRIAL")
# # print("=" * 70)
# #
# # comprehensive_results = comprehensive_no_trial_analysis(df, today_date)  
# #
# #
# #
# # ######################################################################################
# # # VÉRIFICATION DE LA THÉORIE : UN SEUL ABONNEMENT + GIFTED + RENOUVELLEMENTS
# # ######################################################################################
# #
# # def verify_single_subscription_theory(df):
# #     """
# #     Vérifie la théorie :
# #     - Un seul type d'abonnement (1 an)
# #     - Sans trial = Gifted OU Renouvellements automatiques (même sub_id)
# #     """
# #
# #     print("🔍 VÉRIFICATION DE LA THÉORIE")
# #     print("=" * 50)
# #
# #     # === ANALYSE DES SUBSCRIPTIONS SANS TRIAL ===
# #     without_trial = df[df['trial_start_utc'].isna()].copy()
# #
# #     print("📊 ANALYSE DES 2,270 SUBSCRIPTIONS SANS TRIAL:")
# #     print("-" * 50)
# #
# #     # 1. Combien sont gifted ?
# #     gifted_without_trial = without_trial[without_trial['is_gifted_member'] == True]
# #     not_gifted_without_trial = without_trial[without_trial['is_gifted_member'] == False]
# #
# #     print(f"🎁 GIFTED sans trial: {len(gifted_without_trial)} ({len(gifted_without_trial)/len(without_trial)*100:.1f}%)")
# #     print(f"💰 NON-GIFTED sans trial: {len(not_gifted_without_trial)} ({len(not_gifted_without_trial)/len(without_trial)*100:.1f}%)")
# #
# #     # 2. Les non-gifted sont-ils des renouvellements ?
# #     print(f"\n🔄 ANALYSE DES NON-GIFTED SANS TRIAL ({len(not_gifted_without_trial)} subscriptions):")
# #
# #     # Vérification des customer_id multiples
# #     customer_counts = not_gifted_without_trial['customer_id'].value_counts()
# #     customers_single = customer_counts[customer_counts == 1]
# #     customers_multiple = customer_counts[customer_counts > 1]
# #
# #     print(f"   Clients avec 1 seule subscription sans trial: {len(customers_single)}")
# #     print(f"   Clients avec multiples subscriptions sans trial: {len(customers_multiple)}")
# #
# #     if len(customers_multiple) > 0:
# #         print(f"   → Ces multiples pourraient être des renouvellements")
# #
# #         # Examiner quelques exemples
# #         print(f"\n   Exemples de renouvellements potentiels:")
# #         for customer_id, count in customers_multiple.head(5).items():
# #             customer_subs = not_gifted_without_trial[not_gifted_without_trial['customer_id'] == customer_id]
# #             dates = customer_subs['created_utc'].dt.strftime('%Y-%m-%d').tolist()
# #             sub_ids = customer_subs['subscription_id'].tolist()
# #             print(f"      Client {customer_id[:15]}... : {count} subs")
# #             for i, (date, sub_id) in enumerate(zip(dates, sub_ids)):
# #                 print(f"         {i+1}. {date} - {sub_id[:15]}...")
# #
# #     # === VÉRIFICATION DU TYPE D'ABONNEMENT UNIQUE ===
# #     print(f"\n📅 VÉRIFICATION : UN SEUL TYPE D'ABONNEMENT (1 AN)")
# #     print("-" * 50)
# #
# #     # Analyse des durées de période
# #     print(f"📊 DURÉES DES PÉRIODES ACTUELLES:")
# #
# #     # Pour tous les abonnements
# #     duration_counts = df['current_period_duration'].value_counts().head(10)
# #     print(f"   Top 10 des durées de période (tous abonnements):")
# #     for duration, count in duration_counts.items():
# #         print(f"      {duration} jours: {count} subscriptions ({count/len(df)*100:.1f}%)")
# #
# #     # Vérifier si ~365 jours domine
# #     around_1_year = df[(df['current_period_duration'] >= 360) & (df['current_period_duration'] <= 370)]
# #     print(f"\n   Abonnements ~1 an (360-370 jours): {len(around_1_year)} ({len(around_1_year)/len(df)*100:.1f}%)")
# #
# #     # === COMPARAISON AVEC/SANS TRIAL ===
# #     with_trial = df[df['trial_start_utc'].notna()]
# #
# #     print(f"\n📈 COMPARAISON DURÉES AVEC/SANS TRIAL:")
# #     avg_duration_with = with_trial['current_period_duration'].mean()
# #     avg_duration_without = without_trial['current_period_duration'].mean()
# #
# #     print(f"   Durée moyenne AVEC trial: {avg_duration_with:.0f} jours")
# #     print(f"   Durée moyenne SANS trial: {avg_duration_without:.0f} jours")
# #
# #     # Distribution détaillée
# #     print(f"\n   Distribution des durées SANS trial:")
# #     duration_without = without_trial['current_period_duration'].value_counts().head(5)
# #     for duration, count in duration_without.items():
# #         print(f"      {duration} jours: {count} subs ({count/len(without_trial)*100:.1f}%)")
# #
# #     return {
# #         'total_without_trial': len(without_trial),
# #         'gifted_without_trial': len(gifted_without_trial),
# #         'not_gifted_without_trial': len(not_gifted_without_trial),
# #         'customers_single': len(customers_single),
# #         'customers_multiple': len(customers_multiple),
# #         'around_1_year_total': len(around_1_year)
# #     }
# #
# #
# # ######################################################################################
# # # ANALYSE APPROFONDIE DES RENOUVELLEMENTS
# # ######################################################################################
# #
# # def analyze_renewal_patterns_detailed(df):
# #     """
# #     Analyse détaillée des patterns de renouvellement
# #     """
# #
# #     print("🔄 ANALYSE DÉTAILLÉE DES RENOUVELLEMENTS")
# #     print("=" * 50)
# #
# #     without_trial = df[df['trial_start_utc'].isna()]
# #     not_gifted_without_trial = without_trial[without_trial['is_gifted_member'] == False]
# #
# #     # === ANALYSE TEMPORELLE DES RENOUVELLEMENTS ===
# #     print("📅 PATTERNS TEMPORELS DES RENOUVELLEMENTS:")
# #
# #     # Grouper par customer_id et analyser les intervalles
# #     renewal_analysis = []
# #
# #     for customer_id in not_gifted_without_trial['customer_id'].unique():
# #         customer_subs = df[df['customer_id'] == customer_id].sort_values('created_utc')
# #
# #         if len(customer_subs) > 1:
# #             # Calculer les intervalles entre subscriptions
# #             dates = customer_subs['created_utc'].tolist()
# #             trial_flags = customer_subs['trial_start_utc'].notna().tolist()
# #
# #             for i in range(1, len(dates)):
# #                 interval = (dates[i] - dates[i-1]).days
# #
# #                 renewal_analysis.append({
# #                     'customer_id': customer_id,
# #                     'subscription_number': i + 1,
# #                     'interval_days': interval,
# #                     'previous_had_trial': trial_flags[i-1],
# #                     'current_has_trial': trial_flags[i],
# #                     'created_date': dates[i]
# #                 })
# #
# #     if renewal_analysis:
# #         renewal_df = pd.DataFrame(renewal_analysis)
# #
# #         print(f"   Total renouvellements détectés: {len(renewal_df)}")
# #
# #         # Intervalles moyens
# #         avg_interval = renewal_df['interval_days'].mean()
# #         print(f"   Intervalle moyen entre renouvellements: {avg_interval:.0f} jours")
# #
# #         # Distribution des intervalles
# #         print(f"   Distribution des intervalles:")
# #         interval_ranges = [
# #             (350, 380, "~1 an (350-380j)"),
# #             (320, 349, "~11 mois"),
# #             (381, 400, "~13 mois"),
# #             (0, 319, "< 11 mois"),
# #             (401, 999, "> 13 mois")]
# #
# #         for min_int, max_int, label in interval_ranges:
# #             count = len(renewal_df[(renewal_df['interval_days'] >= min_int) & 
# #                                   (renewal_df['interval_days'] <= max_int)])
# #             if count > 0:
# #                 print(f"      {label}: {count} renouvellements ({count/len(renewal_df)*100:.1f}%)")
# #
# #         # Pattern trial → no trial
# #         trial_to_no_trial = renewal_df[
# #             (renewal_df['previous_had_trial'] == True) & 
# #             (renewal_df['current_has_trial'] == False)]
# #
# #         print(f"\n   Renouvellements Trial → Sans Trial: {len(trial_to_no_trial)}")
# #         print(f"   (Confirme que les renouvellements n'ont pas de trial)")
# #
# #         return renewal_df
# #     else:
# #         print("   Aucun pattern de renouvellement détecté")
# #         return pd.DataFrame()
# #
# #
# # ######################################################################################
# # # IMPLICATIONS SUR LES FONCTIONS DE CALCUL
# # ######################################################################################
# #
# # def analyze_function_implications(df):
# #     """
# #     Analyse les implications sur les fonctions de calcul de renouvellement
# #     """
# #
# #     print("⚙️ IMPLICATIONS SUR VOS FONCTIONS DE CALCUL")
# #     print("=" * 50)
# #
# #     without_trial = df[df['trial_start_utc'].isna()]
# #
# #     # === CLASSIFICATION CORRECTE ===
# #     print("📋 CLASSIFICATION CORRECTE DES SUBSCRIPTIONS SANS TRIAL:")
# #
# #     gifted = without_trial[without_trial['is_gifted_member'] == True]
# #     renewals = without_trial[without_trial['is_gifted_member'] == False]
# #
# #     print(f"   🎁 Gifted: {len(gifted)} subscriptions")
# #     print(f"   🔄 Renouvellements: {len(renewals)} subscriptions")
# #     print(f"   📊 Total sans trial: {len(without_trial)} subscriptions")
# #
# #     # === IMPACT SUR LE CALCUL DE CONVERSION ===
# #     print(f"\n📈 IMPACT SUR LE CALCUL DE CONVERSION:")
# #
# #     with_trial = df[df['trial_start_utc'].notna()]
# #     true_new_customers = with_trial  # Seuls les avec trial sont de nouveaux clients
# #
# #     # Ancienne logique (fausse)
# #     old_total = len(df)
# #     old_conversions = len(df[df['is_full_member'] == True])
# #     old_rate = old_conversions / old_total * 100
# #
# #     # Nouvelle logique (correcte)
# #     new_total = len(true_new_customers)
# #     new_conversions = len(true_new_customers[true_new_customers['is_full_member'] == True])
# #     new_rate = new_conversions / new_total * 100
# #
# #     print(f"   ❌ Ancienne logique (incorrecte):")
# #     print(f"      Total: {old_total} | Conversions: {old_conversions} | Taux: {old_rate:.1f}%")
# #     print(f"   ✅ Nouvelle logique (correcte):")
# #     print(f"      Total: {new_total} | Conversions: {new_conversions} | Taux: {new_rate:.1f}%")
# #     print(f"   📊 Différence: {new_rate - old_rate:+.1f} points")
# #
# #     # === IMPACT SUR LE CALCUL DE RENOUVELLEMENT ===
# #     print(f"\n🔄 IMPACT SUR LE CALCUL DE RENOUVELLEMENT:")
# #
# #     print(f"   ❌ Ancienne logique (incorrecte):")
# #     print(f"      Incluait les gifted et les nouveaux sans trial")
# #     print(f"      Mélangeait nouveaux clients et renouvellements")
# #
# #     print(f"   ✅ Nouvelle logique (correcte):")
# #     print(f"      Exclure complètement les gifted")
# #     print(f"      Exclure les renouvellements automatiques")
# #     print(f"      Ne garder que les vrais nouveaux clients (avec trial)")
# #
# #     # === FONCTION CORRIGÉE ===
# #     print(f"\n🛠️ FONCTION DE RENOUVELLEMENT CORRIGÉE:")
# #
# #     # Seuls les nouveaux clients (avec trial) sont éligibles
# #     eligible_correct = true_new_customers[(true_new_customers['days_since_creation'] >= 365) & \
# #         (true_new_customers['is_full_member'] == True)]
# #
# #     renewed_correct = eligible_correct[(eligible_correct['status'] == 'active') | \
# #         ((eligible_correct['canceled_at_utc'].notna()) & ((eligible_correct['canceled_at_utc'] - eligible_correct['created_utc']).dt.days >= 365))]
# #
# #     correct_rate = len(renewed_correct) / len(eligible_correct) * 100 if len(eligible_correct) > 0 else 0
# #
# #     print(f"   Éligibles (nouveaux clients uniquement): {len(eligible_correct)}")
# #     print(f"   Renouvelés: {len(renewed_correct)}")
# #     print(f"   Taux de renouvellement corrigé: {correct_rate:.1f}%")
# #
# #     return {'gifted_count': len(gifted), \
# #         'renewals_count': len(renewals), \
# #         'old_conversion_rate': old_rate, \
# #         'new_conversion_rate': new_rate, \
# #         'correct_renewal_rate': correct_rate}
# #
# #
# # ######################################################################################
# # # FONCTION PRINCIPALE DE VÉRIFICATION
# # ######################################################################################
# #
# # def comprehensive_theory_verification(df):
# #     """
# #     Vérification complète de la théorie
# #     """
# #
# #     print("🎯 VÉRIFICATION COMPLÈTE DE LA THÉORIE")
# #     print("=" * 60)
# #
# #     # Exécuter toutes les analyses
# #     basic_verification = verify_single_subscription_theory(df)
# #     renewal_patterns = analyze_renewal_patterns_detailed(df)
# #     function_implications = analyze_function_implications(df)
# #
# #     # === CONCLUSION ===
# #     print(f"\n🎯 CONCLUSION DE LA VÉRIFICATION")
# #     print("=" * 40)
# #
# #     gifted_pct = basic_verification['gifted_without_trial'] / basic_verification['total_without_trial'] * 100
# #     renewal_pct = basic_verification['not_gifted_without_trial'] / basic_verification['total_without_trial'] * 100
# #
# #     print(f"✅ THÉORIE CONFIRMÉE !")
# #     print(f"   Subscriptions sans trial = {gifted_pct:.1f}% Gifted + {renewal_pct:.1f}% Renouvellements")
# #     print(f"   Un seul type d'abonnement (~1 an)")
# #     print(f"   Renouvellements = même customer, nouvelles sub_id")
# #
# #     print(f"\n📊 IMPACT SUR VOS MÉTRIQUES:")
# #     print(f"   Taux de conversion: {function_implications['old_conversion_rate']:.1f}% → {function_implications['new_conversion_rate']:.1f}%")
# #     print(f"   Taux de renouvellement corrigé: {function_implications['correct_renewal_rate']:.1f}%")
# #
# #     print(f"\n🛠️ ACTIONS REQUISES:")
# #     print(f"   1. Modifier vos fonctions pour exclure gifted et renouvellements")
# #     print(f"   2. Recalculer toutes les métriques historiques")
# #     print(f"   3. Séparer les analyses nouveaux clients vs renouvellements")
# #
# #     return {'theory_confirmed': True, 'basic_verification': basic_verification, 'function_implications': function_implications}
# #
# #
# # # === EXÉCUTION DE LA VÉRIFICATION ===
# # print("🚀 VÉRIFICATION DE LA THÉORIE")
# # print("=" * 50)
# #
# # results = comprehensive_theory_verification(df)
