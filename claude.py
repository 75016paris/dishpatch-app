# %%
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

#df = merge_dataframes(df1, df2)
df = df1 

# %%
######################################################################################
# Removing customers with more than 5 subscriptions (Probably testing accounts)
def remove_high_volume_customers(df, threshold=HIGH_VOLUME_THRESHOLD):
    """Remove customers with more than a specified number of subscriptions"""
    df = df.copy()

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
    df =df.copy()

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
    df = df.copy()

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
    df = df.copy()

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
    df = df.copy()

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

    refund_period_passed = (
            (today_date > df['refund_period_end_utc']) &
            (df['refund_period_end_utc'] < df['current_period_end_utc'])
            )

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
    df = df.copy()

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
# CALCULATING DURATIONS
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
    
    df['days_since_creation'] = (today_date - df['created_utc']).dt.days
   
    return df

df = calculate_duration(df)


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
    
    monday = target_monday.strftime('%d-%m-%y')
    sunday = target_sunday.strftime('%d-%m-%y')

    # Las week info
    week_info = {
        'weeks_ago': weeks_back,
        'week_start': week_start,
        'week_end': week_end, 
        'year': target_monday.year,
        'week_number': target_monday.isocalendar().week,
        'year_week': f"{target_monday.year}-W{target_monday.isocalendar().week:02d}",
        'monday': monday,
        'sunday': sunday,
    }
    
    return week_info

# %%
####################################################################################
# Plot nuber
#
# new_trial_this_week = 110
# last_week_trial = 100
# last_6M_trial = 90
# last_year_trial = 124
# new_renewal_this_week = 45
# churn_this_week = 12
# week_info = get_specific_past_week(weeks_back=1, reference_date=today_date)
#
#
#
# df['week'] = df['created_utc'].dt.to_period('W').apply(lambda r: r.start_time)
# new_trials_per_week = df.groupby('week').size().reset_index(name='new_trials')
#
#
#
#
#
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 11))
# fig.suptitle(f"Weekly Subscription Metrics\n from Monday {week_info['monday']} to Sunday {week_info['sunday']}") 
#
# ax1.set_title('New Trial this week')
# ax1.text(0.5, 0.8, new_trial_this_week, fontsize=16, fontweight='bold', ha='center', va='center', color='grey')
#
# if last_week_trial > new_trial_this_week:
#     ax1.text(0.5, 0.5, f'Last week: {last_week_trial} (-{last_week_trial - new_trial_this_week}) {(((new_trial_this_week - last_week_trial) / last_week_trial) *100):.1f}%', fontsize=12, ha='center', va='center', color='red')
# else:
#     ax1.text(0.5, 0.5, f'Last week: {last_week_trial} ({last_week_trial - new_trial_this_week}) +{(((new_trial_this_week - last_week_trial) / last_week_trial) *100):.1f}%', fontsize=12, ha='center', va='center', color='green')
#
# if last_6M_trial > new_trial_this_week:
#     ax1.text(0.5, 0.3, f'mean Last 6 months: {last_6M_trial} (-{last_6M_trial - new_trial_this_week}) {(((new_trial_this_week - last_6M_trial) / last_6M_trial) *100):.1f}%', fontsize=12, ha='center', va='center', color='red')
# else:
#     ax1.text(0.5, 0.3, f'mean Last 6 months: {last_6M_trial} ({last_6M_trial - new_trial_this_week}) +{(((new_trial_this_week - last_6M_trial) / last_6M_trial) *100):.1f}%', fontsize=12, ha='center', va='center', color='green')
#
# if last_year_trial > new_trial_this_week:
#     ax1.text(0.5, 0.1, f'mean Last year: {last_year_trial} (-{last_year_trial - new_trial_this_week}) {(((new_trial_this_week - last_year_trial) / last_year_trial) *100):.1f}%', fontsize=12, ha='center', va='center', color='red')
# else:
#     ax1.text(0.5, 0.1, f'mean Last year: {last_year_trial} ({last_year_trial - new_trial_this_week}) +{(((new_trial_this_week - last_year_trial) / last_year_trial) *100):.1f}%', fontsize=12, ha='center', va='center', color='green')
#
#
#
# ax1.set_xticklabels([])
# ax1.set_yticklabels([])
# ax1.set_xticks([])
# ax1.set_yticks([])
#
# ax2.set_title('New Renewal this week')
# ax2.text(0.5, 0.5, new_renewal_this_week, fontsize=16, fontweight='bold', ha='center', va='center', color='blue')
# sns.barplot(x='week', y='new_trials', data=new_trials_per_week, color='purple')
# ax2.set_xticklabels([])
# ax2.set_yticklabels([])
# ax2.set_xticks([])
# ax2.set_yticks([])
#
# ax3.set_title('Churn this week')
# ax3.text(0.5, 0.5, churn_this_week, fontsize=16, fontweight='bold', ha='center', va='center', color='red')
# ax3.set_xticklabels([])
# ax3.set_yticklabels([])
# ax3.set_xticks([])
# ax3.set_yticks([])
#
#
# # save figure with week_metric_YYYY-MM-DD.png
# fig.savefig(os.path.join(analysis_dir, f"week_metric_{week_info['year']}-{week_info['week_number']:02d}.png"), bbox_inches='tight', dpi=300)
#
# plt.show()


# %% 
# si les duration < que 40 ce sont des essais 
######################################################################################
def get_full_members_count(df):
    """Count the number of full members"""
    df = df.copy()

    df = df[df['is_full_member'] == True]
    df_active = df[df['status'] == 'active']

    active = len(df_active)
    print(f"Total Active full member: {active}")
    
    
    dict_full_members = {'active': active
                         }

    return dict_full_members


dict_full_member = get_full_members_count(df)


# %%
#######################################################################################
# find the maximun count of full menbers at any time
def get_max_concurrent_full_members(df, dict_full_member):
    """Find maximum concurrent full members using membership periods"""
    # This assumes you have start_date and end_date columns
    events = []
    
    for _, row in df[df['is_full_member']].iterrows():
        events.append((row['refund_period_end_utc'], 1))  # Member starts
        events.append((row['current_period_end_utc'], -1))  # Member ends
    
    events.sort()
    
    current_count = 0
    max_count = 0
    max_date = None
    
    for date, change in events:
        current_count += change
        if current_count > max_count:
            max_count = current_count
            max_date = date

    dict_full_member['max_full_members'] = max_count
    dict_full_member['max_date'] = max_date
    print(f"Maximum concurrent full members: {max_count} on {max_date.strftime('%Y-%m-%d')}")
    
    return dict_full_member


dict_full_members = get_max_concurrent_full_members(df, dict_full_member)


# %%
######################################################################################
# how many trial this week
def get_new_trial_this_week(df, weeks_back=1):
    """Count new trials started this week"""
    week_info = get_specific_past_week(weeks_back=weeks_back, reference_date=today_date)
    
    # Filter for the current week
    df_week = df[(df['trial_start_utc'] >= week_info['week_start']) & 
                 (df['trial_start_utc'] < week_info['week_end'])]
    
    # Count new trials
    new_trials = df_week.shape[0]
    
    print(f"New trials this week ({week_info['year_week']}): {new_trials}")
    
    return new_trials


new_trial_this_week = get_new_trial_this_week(df, weeks_back=1)


# %%
######################################################################################
def get_trial_trends(df, num_weeks=4):
    """Get trial counts for multiple weeks"""
    trends = {}
    for week in range(1, num_weeks + 1):
        trends[f'week_{week}'] = get_new_trial_this_week(df, weeks_back=week)
    return trends

trial_trends = get_trial_trends(df, num_weeks=3)


# %%
######################################################################################
# Count trials that converted to full members        
def get_conversion_rate(df, dict_full_members):
    """Calculate conversion rate from trial to full member"""
    df = df.copy()

    df = df[df['refund_period_end_utc'] < today_date]
    df = df[df['trial_start_utc'].notna()]
    total_trials = len(df)

    df = df[df['is_full_member'] == True]
    total_full_members = len(df)

    if total_trials == 0:
        conversion_rate = 0
    else:
        conversion_rate = (total_full_members / total_trials) * 100


    conversion_rate_dict = {
        'total_trials': total_trials,
        'total_full_members': total_full_members,
        'conversion_rate': round(conversion_rate, 2)
    }

    print(f"Total trials: {total_trials}, Total full members: {total_full_members}, \
          Conversion rate: {conversion_rate}%")
    

    return conversion_rate_dict


conversion_rate_dict = get_conversion_rate(df, dict_full_members)


# %%
######################################################################################
# Count renwals rate
def get_renewal_rate(df):
    """Calculate renewal rate from full members"""
    df = df.copy()

    df = df[df['trial_start_utc'].isna()]
    df = df[df['is_gifted_member'] == False]
    total_no_trial = len(df)

    df = df[df['is_full_member'] == True]
    total_full_members = len(df)
    if total_no_trial == 0:
        renewal_rate = 0
    else:
        renewal_rate = (total_full_members / total_no_trial) * 100

    renewal_rate_dict = {
        'total_no_trial': total_no_trial,
        'total_full_members': total_full_members,
        'renewal_rate': round(renewal_rate, 2)
    }

    print(f"Total no trial &  not gifted: {total_no_trial}, Total full members: {total_full_members}, \
            Renewal rate: {renewal_rate}%")

    return renewal_rate_dict


renewal_rate_dict = get_renewal_rate(df)

# %%
######################################################################################

def identify_second_year_subscriptions(df):
    """Identifier les abonnements probablement en 2ème année"""
    df = df.copy()
    
    # Approximation : si créé il y a plus de 365 jours = potentiellement 2ème année
    df['likely_second_year'] = df['days_since_creation'] > 365
    
    # Mais attention aux pauses/annulations !
    df['active_second_year'] = (
        df['likely_second_year'] & 
        (~df['is_gifted_member'])
    )
    
    return df
df = identify_second_year_subscriptions(df)  
print(df['active_second_year'].sum())



# %%
######################################################################################
# LONG MEMBER :
# How many full menber cancel before one year (rate) (bar plot, nb de mois au moment du cancel)
# How many full menber ask for refund Year 2 ? (rate)
def long_member_analysis(df):
    """Analyze long-term members and their cancellation/refund behavior"""
    df = df.copy()

    # Filter for full members
    full_members = df[df['is_full_member']]

    # Calculate months since creation for each member
    full_members['days_since_creation'] = (today_date - full_members['created_utc']).dt.days

    # Count cancellations before one year
    cancellations_before_one_year = full_members[
        (full_members['canceled_at_utc'].notna()) & 
        (full_members['days_since_creation'] < 366)
    ]

    cancellation_rate = len(cancellations_before_one_year) / len(full_members) * 100 if len(full_members) > 0 else 0

    print(f"Cancellations before one year: {len(cancellations_before_one_year)}")
    print(f"Cancellation rate before one year: {cancellation_rate:.2f}%")

    # Count refund requests in the second year
    second_year_refunds = full_members[
        (full_members['canceled_at_utc'].notna()) & 
        (full_members['days_since_creation'] >= 366) &
        (full_members['days_since_creation'] < 730)
    ]

    refund_rate = len(second_year_refunds) / len(full_members) * 100 if len(full_members) > 0 else 0

    print(f"Refund requests in the second year: {len(second_year_refunds)}")
    print(f"Refund rate in the second year: {refund_rate:.2f}%")

    return {
        'cancellations_before_one_year': len(cancellations_before_one_year),
        'cancellation_rate': cancellation_rate,
        'second_year_refunds': len(second_year_refunds),
        'refund_rate': refund_rate
    }
long_member_stats = long_member_analysis(df)


# %%
######################################################################################
# %%
######################################################################################
# WEEKLY CHART with dual axis barplot north and south (+ cumulative line plot)
# North: stacked conversion + renewal y1 + renewal y2
# South: Churn full member

def create_weekly_metrics_chart(df, today_date, num_weeks=None):
    """
    Create a dual-axis chart with weekly metrics
    Uses existing functions to calculate metrics
    North: Conversions + Renewals (stacked)
    South: Churn full members
    + Cumulative line plot
    """
    
    # Determine analysis period
    if num_weeks is None:
        # Use all data since first date
        first_date = df['created_utc'].min()
        # Calculate number of weeks since first date
        num_weeks = int((today_date - first_date).days / 7) + 1
        print(f"Analysis since first date: {first_date.strftime('%d-%m-%Y')} ({num_weeks} weeks)")
    else:
        print(f"Analysis of last {num_weeks} weeks")
    
    # Create list of weeks using your function
    week_data = []
    week_range = []
    
    for weeks_back in range(num_weeks, 0, -1):  # From oldest to most recent
        week_info = get_specific_past_week(weeks_back=weeks_back, reference_date=today_date)
        week_data.append(week_info)
        week_range.append(week_info['week_start'])
    
    # Initialize lists to store weekly data
    conversions_data = []
    renewals_y1_data = []
    renewals_y2_data = []
    churn_data = []
    new_trials_data = []
    
    # === CALCULATE METRICS FOR EACH WEEK ===
    for i, week_info in enumerate(week_data):
        # === CONVERSIONS (using your existing logic) ===
        # Subscriptions created this week that became full members
        week_conversions = df[
            (df['created_utc'] >= week_info['week_start']) &
            (df['created_utc'] <= week_info['week_end']) &
            (df['trial_start_utc'].notna()) &  # Had a trial
            (df['is_full_member'] == True) &   # Became full member
            (df['refund_period_end_utc'] < today_date)  # Mature period
        ]
        conversions_data.append(len(week_conversions))
        
        # === NEW TRIALS (using your existing function) ===
        week_trials = df[
            (df['trial_start_utc'] >= week_info['week_start']) &
            (df['trial_start_utc'] <= week_info['week_end'])
        ]
        new_trials_data.append(len(week_trials))
        
        # === RENEWALS Y1 (using your existing logic) ===
        # Subscriptions created this week, no trial, no gift, first year
        week_renewals_y1 = df[
            (df['created_utc'] >= week_info['week_start']) &
            (df['created_utc'] <= week_info['week_end']) &
            (df['trial_start_utc'].isna()) &      # No trial = renewal
            (~df['is_gifted_member']) &           # No gift
            (df['is_full_member'] == True) &      # Full member
            (df['days_since_creation'] <= 365)   # First year
        ]
        renewals_y1_data.append(len(week_renewals_y1))
        
        # === RENEWALS Y2+ (using your existing logic) ===
        week_renewals_y2 = df[
            (df['created_utc'] >= week_info['week_start']) &
            (df['created_utc'] <= week_info['week_end']) &
            (df['trial_start_utc'].isna()) &      # No trial = renewal
            (~df['is_gifted_member']) &           # No gift
            (df['is_full_member'] == True) &      # Full member
            (df['days_since_creation'] > 365)    # Second year+
        ]
        renewals_y2_data.append(len(week_renewals_y2))
        
        # === CHURN FULL MEMBERS (using your existing logic) ===
        # Full members who were canceled this week
        week_churn = df[
            (df['canceled_at_utc'] >= week_info['week_start']) &
            (df['canceled_at_utc'] <= week_info['week_end']) &
            (df['is_full_member'] == True)  # Was full member
        ]
        churn_data.append(len(week_churn))
    
    # Convert to pandas series for easier calculations
    conversions_weekly = pd.Series(conversions_data, index=week_range)
    renewals_y1_weekly = pd.Series(renewals_y1_data, index=week_range)
    renewals_y2_weekly = pd.Series(renewals_y2_data, index=week_range)
    churn_weekly = pd.Series(churn_data, index=week_range)
    trials_weekly = pd.Series(new_trials_data, index=week_range)
    
    # Calculate cumulative values
    conversions_cumul = conversions_weekly.cumsum()
    renewals_y1_cumul = renewals_y1_weekly.cumsum()
    renewals_y2_cumul = renewals_y2_weekly.cumsum()
    total_positive_cumul = conversions_cumul + renewals_y1_cumul + renewals_y2_cumul
    churn_cumul = churn_weekly.cumsum()
    
    # === CREATE UNIFIED CHART ===
    fig, ax = plt.subplots(1, 1, figsize=(22, 11))
    
    # Format dates for X axis using your week_info
    weeks_labels = [week_info['monday'] + ' > ' + week_info['sunday'] for week_info in week_data]
    x_pos = range(len(week_range))
    
    # === POSITIVE BARPLOT (NORTH) ===
    # Stacked barplot for gains
    ax.bar(x_pos, conversions_weekly, label='Conversions (Trial→Full)', 
           color='green')
    ax.bar(x_pos, renewals_y1_weekly, bottom=conversions_weekly, 
           label='Renewals Y1', color='lightgreen')
    ax.bar(x_pos, renewals_y2_weekly, 
           bottom=conversions_weekly + renewals_y1_weekly,
           label='Renewals Y2+', color='orange')
    
    # === NEGATIVE BARPLOT (SOUTH) ===
    # Churn barplot (inverted for "south" effect)
    ax.bar(x_pos, -churn_weekly, label='Churn Full Members', 
           color='red')
    
    # === SINGLE CUMULATIVE LINE PLOT ===
    # Calculate net cumulative (positive - negative)
    net_weekly = conversions_weekly + renewals_y1_weekly + renewals_y2_weekly - churn_weekly
    net_cumul = net_weekly.cumsum()
    
    # Secondary axis for cumulative
    ax_twin = ax.twinx()
    ax_twin.plot(x_pos, net_cumul, color='darkblue', 
                 linewidth=1, 
                 label='Net Cumulative (Gains - Losses)')

    
    # === AXIS CONFIGURATION ===
    # Main axis (barplots)
    ax.set_ylabel('Full Members per week\n(Positive: Gains | Negative: Losses)', 
                  fontsize=12, fontweight='bold')
    ax.set_xlabel('Weeks (Monday - Sunday)', fontsize=12, fontweight='bold')

     # Adding total numbers on top of stacked bars (North only)
    for i, (conv, ren1, ren2) in enumerate(zip(conversions_weekly, renewals_y1_weekly, renewals_y2_weekly)):
        total_positive = conv + ren1 + ren2
        if total_positive > 0:  # Only if there's a value
            ax.text(i, total_positive + 1, str(int(total_positive)), 
                   ha='center', va='bottom', fontsize=9, color='green')
    
    # Adding churn numbers below negative bars (South)
    for i, v in enumerate(churn_weekly):
        if v > 0:  # v is positive but bar is negative
            ax.text(i, -v - 1, str(int(v)), 
                   ha='center', va='top', fontsize=9, color='red')    # Secondary axis (line plot)

    ax_twin.set_ylabel('Net Cumulative Total', fontsize=12, fontweight='bold', color='darkblue')
    ax_twin.tick_params(axis='y', labelcolor='darkblue')
    
   
    # === VISUAL CONFIGURATION ===
    # Horizontal line at y=0 to separate positive/negative
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Adjust Y limits for good visual balance
    y_max = max(conversions_weekly + renewals_y1_weekly + renewals_y2_weekly) * 1.2
    y_min = -max(churn_weekly) * 1.2
    ax.set_ylim(y_min, y_max)
    
    # X axis configuration
    ax.set_xticks(x_pos[::max(1, len(x_pos)//15)])  # Adjust number of labels based on length
    ax.set_xticklabels([weeks_labels[i] for i in x_pos[::max(1, len(x_pos)//15)]], 
                       rotation=45, ha='right')


    # === GREY ZONE FOR IMMATURE PERIODS (DATE-BASED) ===
    immature_cutoff = today_date - pd.Timedelta(days=20)

    # Find which weeks fall in the immature period
    immature_indices = []
    for i, week_info in enumerate(week_data):
        if week_info['week_start'] >= immature_cutoff:
            immature_indices.append(i)

    if immature_indices:
        # Create grey zone covering immature weeks
        start_idx = min(immature_indices) - 0.5
        end_idx = max(immature_indices) + 0.5
        
        ax.axvspan(start_idx, end_idx, alpha=0.2, color='grey', 
                   label='Pending conversion period (< 24 days)', zorder=0)
        
        print(f"Grey zone covers {len(immature_indices)} recent weeks")

     # === TITLES AND LEGENDS ===
    if num_weeks == int((today_date - df['created_utc'].min()).days / 7) + 1:
        period_text = f'Since beginning ({week_data[0]["monday"]} - {week_data[-1]["sunday"]})'
    else:
        period_text = f'{num_weeks} weeks ({week_data[0]["monday"]} - {week_data[-1]["sunday"]})'
    
    ax.set_title(f'WEEKLY FULL MEMBERS FLOW\n{period_text}', 
                 fontsize=18, fontweight='bold', pad=30)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # === SUMMARY METRICS (using your existing functions) ===
    print("=== CALCULATING METRICS WITH YOUR FUNCTIONS ===")
    
    # Use your conversion rate function
    conversion_rate_dict = get_conversion_rate(df, {})
    
    # Use your renewal rate function  
    renewal_rate_dict = get_renewal_rate(df)
    
    # Use your long member analysis
    long_member_stats = long_member_analysis(df)
    
    # Use your identify_second_year function
    df_temp = identify_second_year_subscriptions(df)
    active_second_year = df_temp['active_second_year'].sum()
    
    total_conversions = conversions_weekly.sum()
    total_renewals_y1 = renewals_y1_weekly.sum()
    total_renewals_y2 = renewals_y2_weekly.sum()
    total_churn = churn_weekly.sum()
    total_trials = trials_weekly.sum()
    net_growth = total_conversions + total_renewals_y1 + total_renewals_y2 - total_churn
    
    # Summary text with your metrics
    summary_text = f"""SUMMARY {num_weeks} WEEKS:
Total trials: {total_trials:,}
Conversions: {total_conversions:,} ({conversion_rate_dict.get('conversion_rate', 0):.1f}%)
Renewals Year 1: {total_renewals_y1:,}
Renewals Year 2: {total_renewals_y2:,} 
Renewal rate: {renewal_rate_dict.get('renewal_rate', 0):.1f}%
Active 2nd year: {active_second_year:,}
Toral Churn: {total_churn:,}
────────────────
Net growth: {net_growth:,}
Cancellation Y1: {long_member_stats.get('cancellation_rate', 0):.1f}%"""
    
    fig.text(0.4, 0.2, summary_text, fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    if num_weeks is None:
        filename = f"all_time_weekly_full_member_flow_{today_date.strftime('%Y-%m-%d')}.png"
        plt.savefig(os.path.join(analysis_dir, filename), dpi=300, bbox_inches='tight')
    else:
        filename = f"weekly_full_member_flow_{num_weeks}_weeks_{today_date.strftime('%Y-%m-%d')}.png"
        plt.savefig(os.path.join(analysis_dir, filename), dpi=300, bbox_inches='tight')

    print(f"Chart saved: {filename}")
    plt.show()
    
    return {
        'conversions': total_conversions,
        'renewals_y1': total_renewals_y1, 
        'renewals_y2': total_renewals_y2,
        'churn': total_churn,
        'trials': total_trials,
        'net_growth': net_growth,
        'conversion_rate': conversion_rate_dict.get('conversion_rate', 0),
        'renewal_rate': renewal_rate_dict.get('renewal_rate', 0),
        'active_second_year': active_second_year,
        'cancellation_rate_y1': long_member_stats.get('cancellation_rate', 0)
    }

# === USAGE ===
# Option 1: Analyze since first date in data
metrics_summary = create_weekly_metrics_chart(df, today_date)

# Option 2: Analyze only last 12 weeks  
metrics_summary = create_weekly_metrics_chart(df, today_date, num_weeks=8)

print(f"\n Calculated metrics: {metrics_summary}")


# %%
######################################################################################
def plot_weekly_trials(df, today_date, num_weeks=8):
    """
    Plot the number of new trials each week for the last N weeks
    Uses the same style as the full member flow chart
    """
    
    # === CORRECTION: Handle num_weeks=None ===
    if num_weeks is None:
        # Use all data since first date
        first_date = df['created_utc'].min()
        num_weeks = int((today_date - first_date).days / 7) + 1
        print(f"Analysis since first date: {first_date.strftime('%d-%m-%Y')} ({num_weeks} weeks)")
    else:
        print(f"Analysis of last {num_weeks} weeks of trials")
    
    # Create list of weeks using your function
    week_data = []
    week_range = []
    
    for weeks_back in range(num_weeks, 0, -1):  # From oldest to most recent
        week_info = get_specific_past_week(weeks_back=weeks_back, reference_date=today_date)
        week_data.append(week_info)
        week_range.append(week_info['week_start'])
    
    # Initialize list to store weekly trials data
    trials_data = []
    
    # === CALCULATE TRIALS FOR EACH WEEK ===
    for i, week_info in enumerate(week_data):
        # === NEW TRIALS (using your existing logic) ===
        week_trials = df[
            (df['trial_start_utc'] >= week_info['week_start']) &
            (df['trial_start_utc'] <= week_info['week_end'])
        ]
        trials_data.append(len(week_trials))
    
    # Convert to pandas series
    trials_weekly = pd.Series(trials_data, index=week_range)
    
    # === CREATE CHART ===
    fig, ax = plt.subplots(1, 1, figsize=(22, 8))
    
    # Format dates for X axis using your week_info
    weeks_labels = [week_info['monday'] + ' > ' + week_info['sunday'] for week_info in week_data]
    x_pos = range(len(week_range))
    
    # === BARPLOT ===
    bars = ax.bar(x_pos, trials_weekly, label='New Trials', 
                  color='gray', alpha=0.8)
    
    # Adding numbers on top of each bar
    for i, v in enumerate(trials_weekly):
        if v > 0:  # Only if there's a value
            ax.text(i, v + max(trials_weekly) * 0.01, str(int(v)), 
                   ha='center', va='bottom', fontsize=10)
    
    # === AXIS CONFIGURATION ===
    ax.set_ylabel('Number of New Trials per Week', fontsize=12, fontweight='bold')
    ax.set_xlabel('Weeks (Monday - Sunday)', fontsize=12, fontweight='bold')
    
    # === TITLES AND LEGENDS ===
    # === CORRECTION: Handle title based on num_weeks ===
    if num_weeks == int((today_date - df['created_utc'].min()).days / 7) + 1:
        period_text = f'Since beginning ({week_data[0]["monday"]} - {week_data[-1]["sunday"]})'
    else:
        period_text = f'Last {num_weeks} weeks ({week_data[0]["monday"]} - {week_data[-1]["sunday"]})'
    
    ax.set_title(f'WEEKLY NEW TRIALS\n{period_text}', 
                 fontsize=18, fontweight='bold', pad=30)
    
    ax.legend(loc='upper right', fontsize=10)
    
    # === VISUAL CONFIGURATION ===
    # Grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adjust Y limits for good visual balance
    y_max = max(trials_weekly) * 1.1
    ax.set_ylim(0, y_max)
    
    # X axis configuration
    # === CORRECTION: Adjust number of labels for long periods ===
    ax.set_xticks(x_pos[::max(1, len(x_pos)//15)])  # Same logic as your other chart
    ax.set_xticklabels([weeks_labels[i] for i in x_pos[::max(1, len(x_pos)//15)]], 
                       rotation=45, ha='right')
    
    # === SUMMARY METRICS ===
    total_trials = trials_weekly.sum()
    avg_trials = trials_weekly.mean()
    max_trials = trials_weekly.max()
    min_trials = trials_weekly.min()
    
    # === CORRECTION: Avoid error if less than 2 weeks ===
    if len(trials_weekly) >= 2:
        latest_week_text = f"Latest week: {trials_weekly.iloc[-1]:,} trials\nPrevious week: {trials_weekly.iloc[-2]:,} trials"
    else:
        latest_week_text = f"Latest week: {trials_weekly.iloc[-1]:,} trials\nPrevious week: N/A"
    
    # Summary text
    summary_text = f"""SUMMARY {num_weeks} WEEKS:
Total trials: {total_trials:,}
Average per week: {avg_trials:.1f}
Maximum week: {max_trials:,}
Minimum week: {min_trials:,}
────────────────
{latest_week_text}"""
    
    fig.text(0.02, 0.5, summary_text, fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="gray", alpha=0.8))
    
    plt.tight_layout()
    
    # === CORRECTION: Consistent save logic ===
    if num_weeks == int((today_date - df['created_utc'].min()).days / 7) + 1:
        filename = f"all_time_weekly_trials_{today_date.strftime('%Y-%m-%d')}.png"
    else:
        filename = f"weekly_trials_{num_weeks}_weeks_{today_date.strftime('%Y-%m-%d')}.png"
    
    plt.savefig(os.path.join(analysis_dir, filename), dpi=300, bbox_inches='tight')
    
    print(f"Chart saved: {filename}")
    plt.show()
    
    return {
        'total_trials': total_trials,
        'average_per_week': avg_trials,
        'max_week': max_trials,
        'min_week': min_trials,
        'latest_week': trials_weekly.iloc[-1],
        'weekly_data': trials_weekly.tolist()
    }

# === CORRECTED USAGE ===
# Option 1: Analyze last 8 weeks
trials_metrics_8w = plot_weekly_trials(df, today_date, num_weeks=8)

# Option 2: Analyze since beginning  
trials_metrics_all = plot_weekly_trials(df, today_date, num_weeks=None)

print(f"\n📊 8 weeks metrics: {trials_metrics_8w}")
print(f"\n📊 All time metrics: {trials_metrics_all}")


# %%
#######################################################################################
# %%
#######################################################################################
# COHORT CONVERSION FUNNEL ANALYSIS
# Bar 1: Trials
# Bar 2: Survivors after trial cancellation period
# Bar 3: Survivors after refund period

def plot_cohort_conversion_funnel(df, today_date):
    """
    Plot a conversion funnel for different cohorts with 3 bars:
    1. Initial trials
    2. Survivors after trial period (not canceled during trial)
    3. Survivors after refund period (not canceled during refund)
    
    First chart: Last complete cohort (refund period ended)
    Second chart: Comparison of last week vs previous periods
    """
    
    # === CHART 1: LAST COMPLETE COHORT ===
    print("=== Finding last complete cohort (refund period ended) ===")
    
    # Find the most recent week where refund period has ended
    weeks_back = 1
    complete_cohort_found = False
    
    while weeks_back <= 20 and not complete_cohort_found:  # Look back max 20 weeks
        week_info = get_specific_past_week(weeks_back=weeks_back, reference_date=today_date)
        
        # Get trials from this week
        week_trials = df[
            (df['trial_start_utc'] >= week_info['week_start']) &
            (df['trial_start_utc'] <= week_info['week_end'])
        ]
        
        if len(week_trials) > 0:
            # Check if refund period has ended for this cohort
            latest_refund_end = week_trials['refund_period_end_utc'].max()
            if latest_refund_end < today_date:
                complete_cohort_found = True
                complete_cohort_week = week_info
                complete_cohort_trials = week_trials
                print(f"Complete cohort found: Week {week_info['year_week']} ({len(week_trials)} trials)")
            else:
                weeks_back += 1
        else:
            weeks_back += 1
    
    if not complete_cohort_found:
        print("No complete cohort found, using week 4")
        complete_cohort_week = get_specific_past_week(weeks_back=4, reference_date=today_date)
        complete_cohort_trials = df[
            (df['trial_start_utc'] >= complete_cohort_week['week_start']) &
            (df['trial_start_utc'] <= complete_cohort_week['week_end'])
        ]
    
    # Calculate funnel for complete cohort
    total_trials = len(complete_cohort_trials)
    survivors_trial = len(complete_cohort_trials[~complete_cohort_trials['canceled_during_trial']])
    survivors_refund = len(complete_cohort_trials[
        (~complete_cohort_trials['canceled_during_trial']) & 
        (~complete_cohort_trials['canceled_during_refund_period'])
    ])
    
    # === CHART 1: PLOT COMPLETE COHORT ===
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    categories = ['Initial Trials', 'Survived Trial Period', 'Survived Refund Period']
    values = [total_trials, survivors_trial, survivors_refund]
    colors = ['steelblue', 'orange', 'green']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8)
    
    # Add numbers on top of bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                f'{value:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Add percentage of previous stage
        if i > 0:
            prev_value = values[i-1]
            if prev_value > 0:
                percentage = (value / prev_value) * 100
                ax.text(bar.get_x() + bar.get_width()/2., height/2,
                        f'{percentage:.1f}%', ha='center', va='center', 
                        fontweight='bold', fontsize=10, color='white')
    
    ax.set_title(f'CONVERSION FUNNEL - Complete Cohort\nWeek {complete_cohort_week["year_week"]} '
                 f'({complete_cohort_week["monday"]} - {complete_cohort_week["sunday"]})', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Number of Users', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(values) * 1.15)
    
    # Add summary text
    conversion_trial = (survivors_trial / total_trials * 100) if total_trials > 0 else 0
    conversion_refund = (survivors_refund / total_trials * 100) if total_trials > 0 else 0
    
    summary_text = f"""FUNNEL SUMMARY:
Trial Survival Rate: {conversion_trial:.1f}%
Full Conversion Rate: {conversion_refund:.1f}%
Total Drop-off: {100 - conversion_refund:.1f}%"""
    
    fig.text(0.02, 0.7, summary_text, fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    # Save chart 1
    filename1 = f"complete_cohort_funnel_{complete_cohort_week['year_week']}_{today_date.strftime('%Y-%m-%d')}.png"
    plt.savefig(os.path.join(analysis_dir, filename1), dpi=300, bbox_inches='tight')
    print(f"Complete cohort chart saved: {filename1}")
    plt.show()
    
    # === CHART 2: COMPARISON OF MULTIPLE COHORTS ===
    print("\n=== Comparing multiple cohorts ===")
    
    def calculate_cohort_metrics(cohort_df):
        """Calculate funnel metrics for a cohort"""
        if len(cohort_df) == 0:
            return 0, 0, 0
            
        total = len(cohort_df)
        survived_trial = len(cohort_df[~cohort_df['canceled_during_trial']])
        survived_refund = len(cohort_df[
            (~cohort_df['canceled_during_trial']) & 
            (~cohort_df['canceled_during_refund_period'])
        ])
        return total, survived_trial, survived_refund
    
    # Define cohorts to compare
    cohorts_data = []
    
    # 1. Last week cohort
    last_week_info = get_specific_past_week(weeks_back=1, reference_date=today_date)
    last_week_trials = df[
        (df['trial_start_utc'] >= last_week_info['week_start']) &
        (df['trial_start_utc'] <= last_week_info['week_end'])
    ]
    metrics = calculate_cohort_metrics(last_week_trials)
    cohorts_data.append(('Last Week', metrics, 'steelblue'))
    
    # 2. Previous week cohort
    prev_week_info = get_specific_past_week(weeks_back=2, reference_date=today_date)
    prev_week_trials = df[
        (df['trial_start_utc'] >= prev_week_info['week_start']) &
        (df['trial_start_utc'] <= prev_week_info['week_end'])
    ]
    metrics = calculate_cohort_metrics(prev_week_trials)
    cohorts_data.append(('Previous Week', metrics, 'orange'))
    
    # 3. Last 6 months average
    six_months_ago = today_date - pd.DateOffset(months=6)
    six_months_trials = df[df['trial_start_utc'] >= six_months_ago]
    # Group by week and calculate average
    six_months_trials['week'] = six_months_trials['trial_start_utc'].dt.to_period('W')
    weekly_metrics = []
    for week, week_data in six_months_trials.groupby('week'):
        weekly_metrics.append(calculate_cohort_metrics(week_data))
    
    if weekly_metrics:
        avg_metrics = tuple(np.mean([m[i] for m in weekly_metrics]) for i in range(3))
    else:
        avg_metrics = (0, 0, 0)
    cohorts_data.append(('6M Average', avg_metrics, 'green'))
    
    # 4. All time average
    all_time_trials = df[df['trial_start_utc'].notna()]
    all_time_trials['week'] = all_time_trials['trial_start_utc'].dt.to_period('W')
    all_weekly_metrics = []
    for week, week_data in all_time_trials.groupby('week'):
        all_weekly_metrics.append(calculate_cohort_metrics(week_data))
    
    if all_weekly_metrics:
        all_avg_metrics = tuple(np.mean([m[i] for m in all_weekly_metrics]) for i in range(3))
    else:
        all_avg_metrics = (0, 0, 0)
    cohorts_data.append(('All Time Avg', all_avg_metrics, 'purple'))
    
    # === PLOT COMPARISON CHART ===
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    x_labels = [cohort[0] for cohort in cohorts_data]
    bar_width = 0.25
    x_pos = np.arange(len(x_labels))
    
    # Extract data for each stage
    trials_data = [cohort[1][0] for cohort in cohorts_data]
    survived_trial_data = [cohort[1][1] for cohort in cohorts_data]
    survived_refund_data = [cohort[1][2] for cohort in cohorts_data]
    
    # Create bars
    bars1 = ax.bar(x_pos - bar_width, trials_data, bar_width, 
                   label='Initial Trials', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x_pos, survived_trial_data, bar_width, 
                   label='Survived Trial Period', color='orange', alpha=0.8)
    bars3 = ax.bar(x_pos + bar_width, survived_refund_data, bar_width, 
                   label='Survived Refund Period', color='green', alpha=0.8)
    
    # Add numbers on bars
    for i, (trials, surv_trial, surv_refund) in enumerate(zip(trials_data, survived_trial_data, survived_refund_data)):
        # Initial trials
        if trials > 0:
            ax.text(i - bar_width, trials + max(trials_data) * 0.01, f'{trials:.0f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Survived trial
        if surv_trial > 0:
            ax.text(i, surv_trial + max(trials_data) * 0.01, f'{surv_trial:.0f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
            # Add survival rate
            if trials > 0:
                rate = (surv_trial / trials) * 100
                ax.text(i, surv_trial/2, f'{rate:.1f}%',
                        ha='center', va='center', fontweight='bold', fontsize=8, color='white')
        
        # Survived refund
        if surv_refund > 0:
            ax.text(i + bar_width, surv_refund + max(trials_data) * 0.01, f'{surv_refund:.0f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
            # Add conversion rate
            if trials > 0:
                rate = (surv_refund / trials) * 100
                ax.text(i + bar_width, surv_refund/2, f'{rate:.1f}%',
                        ha='center', va='center', fontweight='bold', fontsize=8, color='white')
    
    # Configuration
    ax.set_xlabel('Cohort Period', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Users', fontsize=12, fontweight='bold')
    ax.set_title('CONVERSION FUNNEL COMPARISON\nLast Week vs Historical Averages', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Set y limit
    max_value = max(trials_data) if trials_data else 0
    ax.set_ylim(0, max_value * 1.15)
    
    plt.tight_layout()
    
    # Save chart 2
    filename2 = f"cohort_comparison_funnel_{today_date.strftime('%Y-%m-%d')}.png"
    plt.savefig(os.path.join(analysis_dir, filename2), dpi=300, bbox_inches='tight')
    print(f"Comparison chart saved: {filename2}")
    plt.show()
    
    # Return summary data
    return {
        'complete_cohort': {
            'week': complete_cohort_week['year_week'],
            'total_trials': total_trials,
            'trial_survival_rate': conversion_trial,
            'full_conversion_rate': conversion_refund
        },
        'comparison_data': cohorts_data
    }

# === USAGE ===
cohort_results = plot_cohort_conversion_funnel(df, today_date)
print(f"\n📊 Cohort analysis results: {cohort_results}")
