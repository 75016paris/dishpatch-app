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
        else:
            print(f"Already renamed: {original_name}")
    else:
        print(f"[DEV] Would rename: {original_name} → {new_name}")


df_raw = pd.read_csv(file_path)

# %%

# DATA PREPROCESSING
##################

def preprocess_data(input_df):
    """Clean and preprocess the subscription data"""
    df = input_df.copy()
    print(f"Number of row in df before cleaning {len(df)}")

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
    print(f"📅 Reference date (TODAY) : {reference_date.strftime('%d-%m-%Y')}")

   
    print(f"Number of row in df after cleaning {len(df)}")


    # Set Canceled At (UTC) to Current Period End (UTC) for past_due without cancellation date
    past_due_no_cancel_date = (
        df['canceled_at_utc'].isna() & 
        (df['status'] == 'past_due'))

    # Use current_period_end_utc as a more realistic cancellation proxy
    # If that's also missing, use trial_end_utc, then fall back to created_utc
    df.loc[past_due_no_cancel_date, 'canceled_at_utc'] = df.loc[past_due_no_cancel_date, 
                                                                'current_period_end_utc'].fillna(
        df.loc[past_due_no_cancel_date, 'trial_end_utc']).fillna(df.loc[past_due_no_cancel_date, 
                                                                        'created_utc'])
    # Set Canceled At (UTC) to Current Period End (UTC) for past_due without cancellation date
    past_due_no_cancel_date = (
            df['canceled_at_utc'].isna() & 
            (df['status'] == 'past_due'))


    # Consolidate status
    df.loc[df['status'].isin(['past_due', 'incomplete_expired']), 'status'] = 'canceled'
  
    return df

df = preprocess_data(df_raw)


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

    return df_clean.drop(['duration_days', 'time_diff'], axis=1)

df =  clean_membership_data(df)


# %%


# All Gifted Status
df_all = df.copy()


# Filter out gifted members and analyze the data
def gifted_members(df):
    """Filter gifted members from the DataFrame"""
    return df[df['is_gifted_member'] == True].drop(columns=['is_gifted_member'])

df_gifted = gifted_members(df)



def filter_gifted_members(df):
    """Filter out gifted members from the DataFrame"""
    return df[df['is_gifted_member'] == False].drop(columns=['is_gifted_member'])

df = filter_gifted_members(df)



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

# SUBSCRIPTION end soon
def subscription_end_soon(df):
    """Check if a member's subscription ends soon (within 14 days)"""
    df['subscription_end_soon'] = (
        (df['current_period_end_utc'] > reference_date) &
        (df['current_period_end_utc'] - reference_date < pd.Timedelta(days=14)) &
        (df['status'] == 'active')
        )  
    return df

df_all = subscription_end_soon(df_all)
df = subscription_end_soon(df)

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
        (df['trial_end_utc'] + pd.Timedelta(days=14) > df['canceled_at_utc']) &
        (df['trial_end_utc'].notna())
    )
    return df

df = cancel_during_churn(df)



# %%



# Plotting the cancellation rates with bar of pie chart
def plot_cancellation_rates(df):
    """Plot the cancellation rates with bar of pie chart showing percentages"""
    trial_cancellations = df[df['canceled_during_trial'] == True]
    churn_cancellations = df[df['canceled_during_churn'] == True]
    other_canceled = df[
        (df['status'] == 'canceled') & 
        (df['canceled_during_trial'] == False) & 
        (df['canceled_during_churn'] == False)
    ] 

    # Calculate totals
    total_cancellations = len(trial_cancellations) + len(churn_cancellations) + len(other_canceled)   

    # Active Members
    active_members = df[df['status'] == 'active']
    
    
    # Data for main pie chart
    main_labels = ['Active Members',  'Canceled']
    main_sizes = [len(active_members), total_cancellations]
    main_colors = ['green', 'grey']
    
    # Data for bar chart (cancellation details)
    detail_labels = ['Old Members', 'Trial Cancellations', 'Churn Cancellations']
    detail_sizes = [len(other_canceled), len(trial_cancellations), len(churn_cancellations)]
    detail_colors = ['grey', 'orange', 'red']
    
    
    
    # Subplot 1: Main pie chart
    ax1 = plt.subplot(1, 2, 1)
    wedges, texts, autotexts = ax1.pie(main_sizes, labels=main_labels, colors=main_colors,
                                       autopct='%1.1f%%', startangle=90, explode=(0.1, 0.1), 
                                       textprops={'color': 'white', 'fontsize': 12,
                                                  'fontweight': 'bold'})
    
    # Improve pie chart text appearance
    
    ax1.set_title('Overall Member Distribution', color='white', fontweight='bold', pad=20)
    
    # Subplot 2: Bar chart for cancellation details
    ax2 = plt.subplot(1, 2, 2)
    
    # Calculate percentages relative to total
    total = len(df)
    detail_percentages = [size/total*100 for size in detail_sizes]
    
    bars = ax2.bar(detail_labels, detail_sizes, color=detail_colors, alpha=0.8)
    
    # Add values and percentages on bars
    for bar, value, pct in zip(bars, detail_sizes, detail_percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(detail_sizes)*0.02, f'{value:,}\n({pct:.1f}%)', ha='center', va='bottom', color='white', fontweight='bold', fontsize=11)
    
    ax2.set_title('Cancellation Breakdown', color='white', fontweight='bold', pad=20)
    ax2.set_ylabel('Number of Cancellations', color='white', fontweight='bold')
    ax2.set_facecolor('#282828')
    
    # Improve bar chart appearance
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('white')
    
    # Rotate x labels for better readability
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(wspace=0.6)
    plt.show()
    
    # Display detailed statistics with percentages
    print(f"\n=== MEMBER STATUS BREAKDOWN ===")
    print(f"Total subscription : {total:,}")
    print(f"\n--- Main Distribution ---")
    print(f"├─ Active members: {len(active_members):,} ({len(active_members)/total*100:.1f}%)")
    print(f"└─ Canceled : {total_cancellations:,} ({total_cancellations/total*100:.1f}%)")
    print(f"\n--- Cancellation Details ---")

    print(f"├─ Old members: {len(other_canceled):,} ({len(other_canceled)/total*100:.1f}%)")
    print(f"├─ Trial Cancellations: {len(trial_cancellations):,} ({len(trial_cancellations)/total*100:.1f}%)")
    print(f"└─ Churn Cancellations: {len(churn_cancellations):,} ({len(churn_cancellations)/total*100:.1f}%)")
    print("\n")

plot_cancellation_rates(df)



# %%



 # Plotting evolution of active members over time
 # Plotting evolution of subscription over time 

def plot_active_members_over_time(df):
    """Plot the evolution of active members over time"""
    df.loc[:, 'week'] = df['created_utc'].dt.tz_convert('UTC').dt.to_period('W')

    active_counts = df[df['status'] == 'active'].groupby('week').size()

    subscription_counts = df.groupby('week').size()

    active_counts.plot(color='green', alpha=0.7, label='Active full Members')
    subscription_counts.plot(color='grey', alpha=0.5, label='Total Subscriptions')
    
    plt.title('New Active Members Over Time')
    plt.ylabel('Number of Members')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_active_members_over_time(df)



# %%



# Plotting evolution of active members over time (cumulative)

def plot_cumulative_active_members_over_time(df):
    """Plot the evolution of active members over time"""
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    df.loc[:, 'week'] = df['created_utc'].dt.tz_convert('UTC').dt.to_period('W')

    active_counts = df[df['status'] == 'active'].groupby('week').size().cumsum()

    subscription_counts = df.groupby('week').size().cumsum()



    subscription_counts.plot(color='grey', alpha=0.5, label='Subscriptions')
    active_counts.plot(color='green', alpha=0.7, label='Active Full Members')

    plt.title('Total Active Members Over Time')
    plt.ylabel('Number of Active Members')
    # add number of active members at the end of the plot
    plt.annotate(f'Active Members: {active_counts.iloc[-1]}', xy=(0.95, 0.47), xycoords='axes fraction', ha='right', va='top', fontsize=12, color='white',  bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='green'))
    plt.annotate(f'Total subscription : {subscription_counts.iloc[-1]}', xy=(0.90, 0.90), xycoords='axes fraction', ha='right', va='top', fontsize=12, color='white', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='grey'))
 
    plt.tight_layout()
    plt.show()

plot_cumulative_active_members_over_time(df)


# Creating dataframe for past 5 weeks analysis
def create_past_weeks_dataframe(df, weeks=5):
    """Create a DataFrame for the last 'weeks' weeks"""
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    df.loc[:, 'week'] = df['created_utc'].dt.tz_convert('UTC').dt.to_period('W')
    recent_weeks = df['week'].unique()[-weeks:]
    df_recent = df[df['week'].isin(recent_weeks)]
    return df_recent

df_recent = create_past_weeks_dataframe(df, weeks=5)



# %%



# Bar plot new active members last 5 weeks
def plot_new_active_members_last_weeks(df_recent, weeks=5):
    """Plot the number of new active members in the last 'weeks' weeks"""
    # Use df_recent instead of df
    #df_recent['created_utc'] = pd.to_datetime(df_recent['created_utc'])
    df_recent['week'] = df_recent['created_utc'].dt.to_period('W')
    recent_weeks = df_recent['week'].unique()[-weeks:]
    
    # Use df_recent instead of df
    new_active_counts = df_recent[df_recent['status'].isin(['active', 'canceled'])].groupby('week').size().reindex(recent_weeks, fill_value=0)
    
    new_trialing_counts = df_recent[df_recent['status'] == 'trialing'].groupby('week').size().reindex(recent_weeks, fill_value=0)
    
    
    
    # Create side-by-side bars
    x = np.arange(len(recent_weeks))
    width = 0.35
    
    plt.bar(x - width/2, new_active_counts.values, width, label='New Active Members', color='green', alpha=0.7)
    plt.bar(x + width/2, new_trialing_counts.values, width, label='New Trialing Members', color='grey', alpha=0.7)
    
    plt.title(f'New Active Members in the Last {weeks} Weeks')
    plt.ylabel('Number of New Members')
    plt.xlabel('Week')
    
    # Display weeks on X axis
    plt.xticks(x, [str(week) for week in recent_weeks], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


# %%

print('\n')
print(len(df_gifted[df_gifted['status']== 'active']), 'Active Gifted Members')
print(len(df[df['status'] == 'active']), 'Active non-Gifted Members')
print(len(df[df['status'] == 'trialing']), 'Currently Trialing Members')

# %%


# LAST WEEK ANALYSIS

def last_week_analysis(df):
    """Analyze the last week of active members"""
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    last_week = df['created_utc'].max() - pd.Timedelta(days=7)
    
    # Filter for the last week
    df_last_week = df[df['created_utc'] >= last_week]
    
    # Count new active members
    new_active_members = df_last_week[df_last_week['status'] == 'active'].shape[0]
    
    # Count new trialing members
    new_trialing_members = df_last_week[df_last_week['status'] == 'trialing'].shape[0]
    
    print("\n")
    print(f"📅 Last Week Analysis (from {last_week.strftime('%d-%m-%Y')}):")
    print(f"New Active Members in the Last Week: {new_active_members}")
    print(f"New Trialing Members in the Last Week: {new_trialing_members}") 
    print(f"Still in churn period in the Last Week: {df_last_week[df_last_week['in_churn_period'] == True].shape[0]}")
    print(f"Cancel during Trial in the Last Week: {df_last_week[df_last_week['canceled_during_trial'] == True].shape[0]}")
    print(f"Cancel during Churn in the Last Week: {df_last_week[df_last_week['canceled_during_churn'] == True].shape[0]}")

last_week_analysis(df) 

# %%

# PREVIOUS WEEK ANALYSIS 
def previous_week_analysis(df):
    """Analyze the previous week of active members"""
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    last_week = df['created_utc'].max() - pd.Timedelta(days=7)
    previous_week = last_week - pd.Timedelta(days=7)
    
    # Filter for the previous week
    df_previous_week = df[(df['created_utc'] >= previous_week) & (df['created_utc'] < last_week)]
    
    # Count new active members
    new_active_members = df_previous_week[df_previous_week['status'] == 'active'].shape[0]
    
    # Count new trialing members
    new_trialing_members = df_previous_week[df_previous_week['status'] == 'trialing'].shape[0]
    
    print("\n")
    print(f"📅 Previous Week Analysis (from {previous_week.strftime('%d-%m-%Y')} to {last_week.strftime('%d-%m-%Y')}):")
    print(f"New Active Members in the Previous Week: {new_active_members}")
    print(f"New Trialing Members in the Previous Week: {new_trialing_members}") 
    print(f"Still in churn period in the Previous Week: {df_previous_week[df_previous_week['in_churn_period'] == True].shape[0]}")
    print(f"Cancel during Trial in the Previous Week: {df_previous_week[df_previous_week['canceled_during_trial'] == True].shape[0]}")
    print(f"Cancel during Churn in the Previous Week: {df_previous_week[df_previous_week['canceled_during_churn'] == True].shape[0]}")

previous_week_analysis(df)


# %%



# LAST 6 MONTHS ANALYSIS (mean by weeks)
def last_6_months_analysis(df):
    """Analyze the last 6 months of active members"""
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    six_months_ago = df['created_utc'].max() - pd.Timedelta(days=180)
    
    # Filter for the last 6 months
    df_last_6_months = df[df['created_utc'] >= six_months_ago]
    
    # Group by week and calculate mean
    df_last_6_months.loc[:, 'week'] = df_last_6_months['created_utc'].dt.tz_convert('UTC').dt.to_period('W')
    weekly_counts = df_last_6_months.groupby('week').size().reset_index(name='count')
    
    # Calculate mean per week
    mean_per_week = weekly_counts['count'].mean()
    
    print("\n")
    print(f"📅 Last 6 Months Analysis (from {six_months_ago.strftime('%d-%m-%Y')}):")
    print(f"Mean New Active Members per Week: {mean_per_week:.2f}")
    print(f"Mean New Trialing Members per Week: {df_last_6_months[df_last_6_months['status'] == 'trialing'].groupby('week').size().mean():.2f}") 
    print(f"Mean Still in churn period per Week: {df_last_6_months[df_last_6_months['in_churn_period'] == True].groupby('week').size().mean():.2f}")
    print(f"Mean Cancel during Trial per Week: {df_last_6_months[df_last_6_months['canceled_during_trial'] == True].groupby('week').size().mean():.2f}")
    print(f"Mean Cancel during Churn per Week: {df_last_6_months[df_last_6_months['canceled_during_churn'] == True].groupby('week').size().mean():.2f}")

last_6_months_analysis(df)


# %%


# CRITICAL TIMING
# print all gifted members who's subscription is ending soon
def critical_timing_gifted(df):
    """Print all gifted members whose subscription is ending soon"""
    df['current_period_end_utc'] = pd.to_datetime(df['current_period_end_utc'])
    df_gifted_ending_soon = df[(df['is_gifted_member'] == True) & (df['subscription_end_soon'] == True)]
    
    if not df_gifted_ending_soon.empty:
        print(f"\n{len(df_gifted_ending_soon)} Gifted Members with Subscription Ending Soon:")
        for index, row in df_gifted_ending_soon.iterrows():
            print(f"Name: {row['customer_name']}, Subscription End: {row['current_period_end_utc'].strftime('%d-%m-%Y')}")
    else:
        print("\nNo Gifted Members with Subscription Ending Soon.")

# print all non-gifted members who's subscription is ending soon
def critical_timing_non_gifted(df):
    """Print all non-gifted members whose subscription is ending soon"""
    df['current_period_end_utc'] = pd.to_datetime(df['current_period_end_utc'])
    df_non_gifted_ending_soon = df[(df['is_gifted_member'] == False) & (df['subscription_end_soon'] == True)]
    
    if not df_non_gifted_ending_soon.empty:
        print(f"\n{len(df_non_gifted_ending_soon)} Non-Gifted Members with Subscription Ending Soon:")
        for index, row in df_non_gifted_ending_soon.iterrows():
            print(f"Name: {row['customer_name']}, Subscription End: {row['current_period_end_utc'].strftime('%d-%m-%Y')}")
    else:
        print("\nNo Non-Gifted Members with Subscription Ending Soon.")



critical_timing_gifted(df_all)
critical_timing_non_gifted(df_all)

# %%


print("""
Subscription without trial
--------------------------
1486 subscription without trial


## GIFTED
---------
866 subscription without trial
58,7% of (all untrailed subscription) are gifted


## UN-GIFTED
------------
SWEET SPOT - 13.5% 
200 subscription without trial
16 sept 2024 > 97.7% Conversion rate -  43 subscriptions
23 sept 2024 > 100% Conversion rate - 54 subscriptions

MASS MARKET - 20.3%
300 subscription without trial
11 sept 2024 > Conversion rate 41% - 85 subscriptions

UNKNOWN REASON - 7,5% - 100 subscriptions
""")

# %%
def plot_signup_conversion_funnel_layered(df_recent, weeks=5):
    """Plot conversion funnel: wide signup background + foreground cancellations & conversions"""
    df_recent['week'] = df_recent['created_utc'].dt.to_period('W')
    recent_weeks = df_recent['week'].unique()[-weeks:]
    
    # New signups total
    new_signups = df_recent.groupby('week').size().reindex(recent_weeks, fill_value=0)
    
    # Canceled during trial
    cancel_trial = df_recent[df_recent['canceled_during_trial'] == True].groupby('week').size().reindex(recent_weeks, fill_value=0)
    
    # Canceled during churn period  
    cancel_churn = df_recent[df_recent['canceled_during_churn'] == True].groupby('week').size().reindex(recent_weeks, fill_value=0)
    
    # Full members (active + not in trial + not in churn period)
    full_members = df_recent[
        (df_recent['status'] == 'active') & 
        (df_recent['in_churn_period'] == False)
    ].groupby('week').size().reindex(recent_weeks, fill_value=0)
    
    x = np.arange(len(recent_weeks))
    wide_width = 0.6  # Large width for background
    narrow_width = 0.3  # Narrow width for foreground bars
    
    plt.figure(figsize=(14, 8))
    
    # Background: Wide signup bars (behind everything)
    plt.bar(x, new_signups.values, wide_width, label='Total Signups', 
             color='gray', alpha=0.8, zorder=1)
    
    # Foreground: Stacked cancellations (trial + churn)
    plt.bar(x - narrow_width/2, cancel_trial.values, narrow_width, 
             label='Canceled During Trial', color='orange', zorder=3)
    plt.bar(x - narrow_width/2, cancel_churn.values, narrow_width, 
             bottom=cancel_trial.values, label='Canceled During Churn', 
             color='red', zorder=3)
    
    # Foreground: Full members converted
    plt.bar(x + narrow_width/2, full_members.values, narrow_width, 
             label='Full Members (Converted)', color='green', alpha=0.9, zorder=3)
    
    plt.title(f'Signup → Cancellation → Conversion Funnel - Last {weeks} Weeks')
    plt.ylabel('Number of Members')
    plt.xlabel('Signup Week')
    plt.xticks(x, [str(week) for week in recent_weeks], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.show()
    
    # Print conversion rates
    total_signups = new_signups.sum()
    total_full_members = full_members.sum()
    total_trial_cancel = cancel_trial.sum()
    total_churn_cancel = cancel_churn.sum()
    
    print(f"\n=== CONVERSION SUMMARY - Last {weeks} weeks ===")
    print(f"📊 Total Signups: {total_signups}")
    print(f"❌ Trial Cancellations: {total_trial_cancel} ({total_trial_cancel/total_signups*100:.1f}%)")
    print(f"⚠️  Churn Cancellations: {total_churn_cancel} ({total_churn_cancel/total_signups*100:.1f}%)")
    print(f"✅ Full Member Conversions: {total_full_members} ({total_full_members/total_signups*100:.1f}%)")
    print(f"🔄 Still Processing: {total_signups - total_trial_cancel - total_churn_cancel - total_full_members}")


plot_signup_conversion_funnel_layered(df_recent, weeks=5)



def add_cohort_analysis(df):
    """Add cohort analysis columns to track conversion by signup period"""
    
    # Create cohort periods
    df['signup_month'] = df['created_utc'].dt.to_period('M')
    df['signup_week'] = df['created_utc'].dt.to_period('W')
    
    # Define conversion states
    df['converted_to_paid'] = (
        (df['trial_end_utc'].notna()) & 
        (df['status'] == 'active') & 
        (df['in_churn_period'] == False)
    )
    
    # Time-based metrics
    df['days_since_signup'] = (reference_date - df['created_utc']).dt.days
    df['trial_length_days'] = (df['trial_end_utc'] - df['trial_start_utc']).dt.days
    
    return df

def plot_cohort_conversion_rates(df, period='month'):
    """Plot conversion rates by cohort"""
    period_col = f'signup_{period}'
    
    # Group by cohort and calculate metrics
    cohort_stats = df.groupby(period_col).agg({
        'customer_name': 'count',  # Total signups
        'converted_to_paid': 'sum',  # Successful conversions
        'canceled_during_trial': 'sum',  # Trial cancellations
        'canceled_during_churn': 'sum'   # Churn cancellations
    }).rename(columns={'customer_name': 'total_signups'})
    
    # Calculate rates
    cohort_stats['conversion_rate'] = (cohort_stats['converted_to_paid'] / cohort_stats['total_signups'] * 100)
    cohort_stats['trial_cancel_rate'] = (cohort_stats['canceled_during_trial'] / cohort_stats['total_signups'] * 100)
    cohort_stats['churn_cancel_rate'] = (cohort_stats['canceled_during_churn'] / cohort_stats['total_signups'] * 100)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top plot: Absolute numbers
    x = range(len(cohort_stats))
    width = 0.25
    
    ax1.bar([i - width for i in x], cohort_stats['total_signups'], width, 
            label='Total Signups', color='blue', alpha=0.7)
    ax1.bar(x, cohort_stats['converted_to_paid'], width, 
            label='Converted to Paid', color='green', alpha=0.7)
    ax1.bar([i + width for i in x], cohort_stats['canceled_during_trial'] + cohort_stats['canceled_during_churn'], width, 
            label='Total Cancellations', color='red', alpha=0.7)
    
    ax1.set_title(f'Cohort Performance by {period.title()}')
    ax1.set_ylabel('Number of Members')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(period) for period in cohort_stats.index], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Conversion rates
    ax2.plot(x, cohort_stats['conversion_rate'], marker='o', linewidth=2, 
             label='Conversion Rate', color='green')
    ax2.plot(x, cohort_stats['trial_cancel_rate'], marker='s', linewidth=2, 
             label='Trial Cancel Rate', color='orange')
    ax2.plot(x, cohort_stats['churn_cancel_rate'], marker='^', linewidth=2, 
             label='Churn Cancel Rate', color='red')
    
    ax2.set_title(f'Conversion Rates by {period.title()}')
    ax2.set_ylabel('Rate (%)')
    ax2.set_xlabel(f'Signup {period.title()}')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(period) for period in cohort_stats.index], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n=== COHORT ANALYSIS SUMMARY ===")
    print(f"Overall Conversion Rate: {cohort_stats['conversion_rate'].mean():.1f}%")
    print(f"Best Performing Cohort: {cohort_stats['conversion_rate'].idxmax()} ({cohort_stats['conversion_rate'].max():.1f}%)")
    print(f"Worst Performing Cohort: {cohort_stats['conversion_rate'].idxmin()} ({cohort_stats['conversion_rate'].min():.1f}%)")
    
    return cohort_stats

# Usage:
df = add_cohort_analysis(df)
cohort_monthly = plot_cohort_conversion_rates(df, 'month')
cohort_weekly = plot_cohort_conversion_rates(df, 'week')



def clean_membership_data_fixed(df):
    """Fixed duplicate detection - only remove TRUE duplicates"""
    
    # Remove very short subscriptions (likely test accounts)
    df['duration_days'] = (pd.to_datetime(df['ended_at_utc']) - pd.to_datetime(df['created_utc'])).dt.days
    df_clean = df[~((df['duration_days'] < 1) & ~(df['status'].isin(['active', 'trialing'])))]

    # ONLY remove TRUE duplicates (same day, likely data errors)
    df_clean = df_clean.sort_values(['customer_name', 'created_utc'], ascending=[True, True])  # ← Fixed sort order
    df_clean['time_diff'] = df_clean.groupby('customer_name')['created_utc'].diff()
    
    # Much more restrictive duplicate detection
    true_duplicates = (
        (df_clean['time_diff'] < pd.Timedelta(hours=2)) &  # Only within 2 hours
        (df_clean['time_diff'].notna())
    )
    
    duplicates_to_remove = df_clean[true_duplicates]
    if len(duplicates_to_remove) > 0:
        print(f"Removing {len(duplicates_to_remove)} TRUE duplicates (< 2 hours apart)")
    
    df_clean = df_clean[~true_duplicates]
    return df_clean.drop(['duration_days', 'time_diff'], axis=1)


print(len(df_raw))
df_test = preprocess_data(df_raw) 
df = clean_membership_data_improved(df_test)
print(f"Number of rows after improved cleaning: {len(df)}")
