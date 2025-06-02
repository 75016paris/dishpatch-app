# %%
##################
# IMPORT LIBRARIES
##################

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# %%
##################
# VISUAL SETTINGS
##################

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 12
sns.set_palette("viridis")

# %%
##################
# LOADING CSV
##################

RENAME_FILES = False
data_dir = 'data'

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
##################
# DATA PREPROCESSING
##################

df = df_raw.copy()

# Date conversion
date_cols = [col for col in df.columns if '(UTC)' in col]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

df = df.sort_values(by='Created (UTC)')

# Column selection and renaming
df = df[['id', 'Customer ID', 'Customer Name', 'Status', 'Cancellation Reason',
         'Created (UTC)', 'Start (UTC)', 'Start Date (UTC)', 'Current Period Start (UTC)',
         'Current Period End (UTC)', 'Trial Start (UTC)', 'Trial End (UTC)',
         'Canceled At (UTC)', 'Ended At (UTC)', 'senderShopifyCustomerId (metadata)']]

df.rename(columns={
    'Customer ID': 'customer_id',
    'Customer Name': 'customer_name',
    'Status': 'status',
    'Created (UTC)': 'created_utc',
    'Start (UTC)': 'start_utc',
    'Start Date (UTC)': 'start_date_utc',
    'Current Period Start (UTC)': 'current_period_start_utc',
    'Current Period End (UTC)': 'current_period_end_date_utc',
    'Trial Start (UTC)': 'trial_start_utc',
    'Trial End (UTC)': 'trial_end_utc',
    'Canceled At (UTC)': 'canceled_at_utc',
    'Ended At (UTC)': 'ended_at_utc',
    'senderShopifyCustomerId (metadata)': 'is_gifted_member'
}, inplace=True)

df['is_gifted_member'] = df['is_gifted_member'].notna()
reference_date = datetime.now()

# Consolidate status
df.loc[df['status'].isin(['past_due', 'incomplete_expired']), 'status'] = 'canceled'

# %%
##################
# HELPER FUNCTIONS
##################

def clean_membership_data(df):
    """Clean and prepare membership data for analysis"""
    # Remove very short subscriptions
    df['duration_days'] = (pd.to_datetime(df['ended_at_utc']) - pd.to_datetime(df['created_utc'])).dt.days
    df_clean = df[~((df['duration_days'] < 1) & (df['duration_days'].notna()))]
    
    # Remove duplicate signups (within 6 hours)
    df_clean = df_clean.sort_values(['customer_name', 'created_utc'])
    df_clean['time_diff'] = df_clean.groupby('customer_name')['created_utc'].diff()
    df_clean = df_clean[~((df_clean['time_diff'] < pd.Timedelta(hours=6)) & (df_clean['time_diff'].notna()))]
    
    return df_clean.drop(['duration_days', 'time_diff'], axis=1)

def calculate_subscription_duration(row):
    """Calculate subscription duration accurately"""
    start_date = row['trial_end_utc'] if pd.notna(row['trial_end_utc']) else row['created_utc']
    
    if row['status'] == 'active':
        end_date = reference_date
    elif pd.notna(row['canceled_at_utc']):
        end_date = row['canceled_at_utc']
    elif pd.notna(row['ended_at_utc']):
        end_date = row['ended_at_utc']
    else:
        end_date = reference_date
    
    return (end_date - start_date).days

def validate_conversion_funnel(customer_df):
    """Validate funnel consistency"""
    total_signups = len(customer_df)
    initial_conversions = customer_df['paid_after_trial'].sum()
    eligible_renewals = customer_df['eligible_for_1st_renewal'].sum()
    actual_renewals = customer_df['actually_renewed_1st'].sum()
    
    print("=== VALIDATION DU FUNNEL ===")
    print(f"Signups totaux: {total_signups:,}")
    print(f"Conversions initiales: {initial_conversions:,}")
    print(f"Éligibles renouvellement: {eligible_renewals:,}")
    print(f"Renouvellements réels: {actual_renewals:,}")
    
    assert actual_renewals <= eligible_renewals, "Plus de renouvellements que d'éligibles!"
    assert eligible_renewals <= initial_conversions, "Plus d'éligibles que de conversions!"
    assert initial_conversions <= total_signups, "Plus de conversions que de signups!"
    
    print("✓ Cohérence du funnel validée")
    return {
        'signups': total_signups,
        'conversions': initial_conversions,
        'eligible_renewals': eligible_renewals,
        'actual_renewals': actual_renewals
    }

# %%
##################
# DATA PROCESSING
##################

df = clean_membership_data(df)
print(f"📅 Reference date for analysis: {reference_date.strftime('%Y-%m-%d')}")

# Filter to regular signups only
analysis_df = df[~df['is_gifted_member']].copy()

# Define conversion logic
analysis_df['paid_after_trial'] = (
    ((analysis_df['status'] == 'active') & analysis_df['canceled_at_utc'].isna()) |
    ((analysis_df['status'] == 'canceled') & (analysis_df['canceled_at_utc'].notna()) & 
     (analysis_df['trial_end_utc'].notna()) & 
     (analysis_df['canceled_at_utc'] > analysis_df['trial_end_utc'] + pd.Timedelta(days=14))) |
    ((analysis_df['status'] == 'canceled') & (analysis_df['canceled_at_utc'].notna()) & 
     (analysis_df['trial_end_utc'].isna()) & (analysis_df['current_period_start_utc'].notna()) & 
     (analysis_df['current_period_end_date_utc'].notna()) &
     ((analysis_df['current_period_end_date_utc'] - analysis_df['current_period_start_utc']).dt.days > 50))
)

analysis_df['is_trial_cancellation'] = (
    ((analysis_df['status'] == 'canceled') & (analysis_df['trial_end_utc'].notna()) & 
     (analysis_df['canceled_at_utc'].notna()) & 
     (analysis_df['canceled_at_utc'] <= analysis_df['trial_end_utc'])) |
    ((analysis_df['status'] == 'canceled') & (analysis_df['trial_end_utc'].isna()) & 
     (analysis_df['canceled_at_utc'].notna()))
)

analysis_df['is_refund'] = (
    (analysis_df['status'] == 'canceled') & (analysis_df['canceled_at_utc'].notna()) &
    (analysis_df['trial_end_utc'].notna()) & 
    (analysis_df['canceled_at_utc'] > analysis_df['trial_end_utc']) &
    ((analysis_df['canceled_at_utc'] - analysis_df['trial_end_utc']).dt.days <= 14)
)

analysis_df['is_currently_trialing'] = analysis_df['status'] == 'trialing'

# %%
##################
# CUSTOMER AGGREGATION
##################

customer_df = analysis_df.groupby('customer_name').agg({
    'customer_id': 'first',
    'created_utc': 'first',
    'status': 'last',
    'current_period_start_utc': 'last',
    'current_period_end_date_utc': 'last',
    'trial_start_utc': 'first',
    'trial_end_utc': 'first',
    'canceled_at_utc': 'last',
    'ended_at_utc': 'last',
    'is_gifted_member': 'any',
    'paid_after_trial': 'any',
    'is_trial_cancellation': 'any',
    'is_refund': 'any',
    'is_currently_trialing': 'any',
    'id': 'count'
}).rename(columns={'id': 'subscription_count'}).reset_index()

# Calculate subscription duration for each customer
customer_df['subscription_duration_days'] = customer_df.apply(calculate_subscription_duration, axis=1)

# Define customer-level metrics
customer_df['subscription_start_date'] = np.where(
    customer_df['trial_end_utc'].notna(),
    customer_df['trial_end_utc'],
    customer_df['created_utc']
)

customer_df['eligible_for_1st_renewal'] = (
    customer_df['paid_after_trial'] & 
    (customer_df['subscription_duration_days'] >= 300)
)

customer_df['actually_renewed_1st'] = (
    customer_df['paid_after_trial'] & 
    (customer_df['subscription_duration_days'] >= 400)
)

customer_df['actually_renewed_2nd'] = (
    customer_df['paid_after_trial'] & 
    (customer_df['subscription_duration_days'] >= 730)
)

# %%
##################
# STATUS DETERMINATION
##################

def determine_customer_status(row):
    if row['is_trial_cancellation']:
        return 'Trial Canceled'
    elif row['is_currently_trialing']:
        return 'Trialing'
    elif not row['paid_after_trial']:
        return 'Never Converted'
    elif row['actually_renewed_1st'] and row['status'] in ['active', 'past_due']:
        return 'Active - Renewed'
    elif row['paid_after_trial'] and row['status'] in ['active', 'past_due']:
        return 'Active - First Year'
    else:
        return 'Churned'

customer_df['current_status_agg'] = customer_df.apply(determine_customer_status, axis=1)

# %%
##################
# KPI CALCULATIONS
##################

total_unique_signups = len(customer_df)
total_initial_conversions = customer_df['paid_after_trial'].sum()
total_trial_cancellations = customer_df['is_trial_cancellation'].sum()
total_refunded = customer_df['is_refund'].sum()
total_eligible_for_renewal = customer_df['eligible_for_1st_renewal'].sum()
total_actually_renewed = customer_df['actually_renewed_1st'].sum()
current_active_members = customer_df[customer_df['status'] == 'active']['customer_name'].nunique()

# Validate funnel
funnel_metrics = validate_conversion_funnel(customer_df)

# Calculate rates
kpi_conversion_rate = (total_initial_conversions / total_unique_signups * 100) if total_unique_signups > 0 else 0
kpi_trial_cancel_rate = (total_trial_cancellations / total_unique_signups * 100) if total_unique_signups > 0 else 0
kpi_refund_rate = (total_refunded / total_initial_conversions * 100) if total_initial_conversions > 0 else 0
kpi_renewal_rate = (total_actually_renewed / total_eligible_for_renewal * 100) if total_eligible_for_renewal > 0 else 0

# %%
##################
# RESULTS SUMMARY
##################

print(f"Total unique non-gift signups: {total_unique_signups:,}")
print(f"Total unique initial conversions: {total_initial_conversions:,}")
print(f"Currently active paying members: {current_active_members:,}")
print(f"Initial conversion rate: {kpi_conversion_rate:.1f}%")
print(f"Trial cancellation rate: {kpi_trial_cancel_rate:.1f}%")
print(f"Refund rate (on initial conversions): {kpi_refund_rate:.1f}%")
print(f"Customers eligible for 1st renewal: {total_eligible_for_renewal:,}")
print(f"Customers who actually renewed 1st time: {total_actually_renewed:,}")
print(f"Renewal rate (on eligible): {kpi_renewal_rate:.1f}%")

# %%
##################
# COHORT ANALYSIS
##################

customer_df['signup_week'] = customer_df['created_utc'].dt.to_period('W-SUN')

cohort_analysis_df = customer_df.groupby('signup_week').agg(
    total_signups_cohort=('customer_name', 'nunique'),
    initial_conversions_cohort=('paid_after_trial', 'sum'),
    trial_cancellations_cohort=('is_trial_cancellation', 'sum'),
    refunds_cohort=('is_refund', 'sum'),
    eligible_for_renewal_cohort=('eligible_for_1st_renewal', 'sum'),
    actually_renewed_cohort=('actually_renewed_1st', 'sum')
).reset_index()

# Calculate cohort rates
cohort_analysis_df['conversion_rate_cohort'] = np.where(
    cohort_analysis_df['total_signups_cohort'] > 0,
    (cohort_analysis_df['initial_conversions_cohort'] / cohort_analysis_df['total_signups_cohort'] * 100),
    0
).round(1)

cohort_analysis_df['trial_cancel_rate_cohort'] = np.where(
    cohort_analysis_df['total_signups_cohort'] > 0,
    (cohort_analysis_df['trial_cancellations_cohort'] / cohort_analysis_df['total_signups_cohort'] * 100),
    0
).round(1)

cohort_analysis_df['refund_rate_cohort'] = np.where(
    cohort_analysis_df['initial_conversions_cohort'] > 0,
    (cohort_analysis_df['refunds_cohort'] / cohort_analysis_df['initial_conversions_cohort'] * 100),
    0
).round(1)

cohort_analysis_df['renewal_rate_cohort'] = np.where(
    cohort_analysis_df['eligible_for_renewal_cohort'] > 0,
    (cohort_analysis_df['actually_renewed_cohort'] / cohort_analysis_df['eligible_for_renewal_cohort'] * 100),
    0
).round(1)

# %%
##################
# VISUALIZATIONS
##################

def create_improved_visualizations():
    """Creates visualizations with corrected logic"""
    
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    fig1.suptitle('Dishpatch Supper Club - Overall Business Metrics', 
                  fontsize=20, fontweight='bold')

    # Conversion Funnel
    funnel_stages = ['Unique\nSignups', 'Initial\nNet Conversions', 'Currently\nActive', 'Renewed\n(1st Time)']
    funnel_counts = [total_unique_signups, total_initial_conversions, current_active_members, total_actually_renewed]
    funnel_colors = ['#4a7bab', '#69a760', '#f08c4c', '#c45d5d']

    bars = ax1.bar(funnel_stages, funnel_counts, color=funnel_colors)
    ax1.set_title('Conversion Funnel (Unique Customers)', fontweight='bold')
    ax1.set_ylabel('Number of Customers')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage = (height / funnel_counts[0] * 100) if funnel_counts[0] > 0 else 0
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.01 * max(funnel_counts)), 
                 f'{height:,}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax1.set_ylim(0, max(funnel_counts) * 1.15)

    # Status Distribution
    status_counts_agg = customer_df['current_status_agg'].value_counts()
    colors = plt.cm.viridis(np.linspace(0, 1, len(status_counts_agg)))
    
    wedges, texts, autotexts = ax2.pie(status_counts_agg.values, labels=status_counts_agg.index, 
                                       autopct='%1.1f%%', colors=colors, startangle=90, pctdistance=0.85)
    ax2.set_title('Current Status Distribution (Unique Customers)', fontweight='bold')
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    ax2.add_artist(centre_circle)

    # Key Business KPIs
    kpi_names = ['Initial\nConversion Rate', 'Trial\nCancellation Rate', 'Refund Rate\n(on Conversions)', 'Renewal Rate\n(1st Time)']
    kpi_values = [kpi_conversion_rate, kpi_trial_cancel_rate, kpi_refund_rate, kpi_renewal_rate]
    kpi_colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    
    kpi_bars = ax3.bar(kpi_names, kpi_values, color=kpi_colors)
    ax3.set_title('Key Performance Indicators (KPIs)', fontweight='bold')
    ax3.set_ylabel('Rate (%)')
    
    for bar in kpi_bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', 
                 ha='center', va='bottom', fontweight='bold')
    ax3.set_ylim(0, max(kpi_values) * 1.2 if kpi_values else 10)

    # Simple member count over time
    ax4.text(0.5, 0.5, f"Current Active Members:\n{current_active_members:,}", 
             ha='center', va='center', transform=ax4.transAxes, fontsize=16, fontweight='bold')
    ax4.set_title('Active Members Summary', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Cohort Analysis Visualization
def create_cohort_analysis():
    recent_cohorts = cohort_analysis_df.tail(26).copy()
    recent_cohorts['cohort_week_str'] = recent_cohorts['signup_week'].astype(str)

    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    fig2.suptitle('Weekly Cohort Analysis (Last 26 Weeks)', fontsize=20, fontweight='bold')
    x_pos_cohort = np.arange(len(recent_cohorts))

    # Signups vs Conversions
    ax1.bar(x_pos_cohort - 0.2, recent_cohorts['total_signups_cohort'], width=0.4, label='Signups', color='#3498db', alpha=0.8)
    ax1.bar(x_pos_cohort + 0.2, recent_cohorts['initial_conversions_cohort'], width=0.4, label='Initial Conversions', color='#2ecc71', alpha=0.8)
    ax1.set_title('Signups vs Initial Conversions by Cohort', fontweight='bold')
    ax1.set_ylabel('Number of Customers')
    ax1.set_xticks(x_pos_cohort)
    ax1.set_xticklabels(recent_cohorts['cohort_week_str'], rotation=45, ha="right", fontsize=8)
    ax1.legend()

    # Rates
    ax2.plot(x_pos_cohort, recent_cohorts['conversion_rate_cohort'], linewidth=2, color='#2ecc71', marker='o', label='Initial Conversion Rate')
    ax2.plot(x_pos_cohort, recent_cohorts['trial_cancel_rate_cohort'], linewidth=2, color='#e74c3c', marker='s', label="Trial Cancellation Rate")
    ax2.set_title('Conversion and Trial Cancellation Rates by Cohort', fontweight='bold')
    ax2.set_ylabel('Rate (%)')
    ax2.set_xticks(x_pos_cohort)
    ax2.set_xticklabels(recent_cohorts['cohort_week_str'], rotation=45, ha="right", fontsize=8)
    ax2.legend()

    # Refund Rate
    ax3.bar(x_pos_cohort, recent_cohorts['refund_rate_cohort'], color='#f39c12', alpha=0.8)
    ax3.set_title('Refund Rate by Cohort', fontweight='bold')
    ax3.set_ylabel('Refund Rate (%)')
    ax3.set_xticks(x_pos_cohort)
    ax3.set_xticklabels(recent_cohorts['cohort_week_str'], rotation=45, ha="right", fontsize=8)

    # Renewal Rate
    mature_cohorts = recent_cohorts[recent_cohorts['eligible_for_renewal_cohort'] > 0]
    if not mature_cohorts.empty:
        x_pos_renewal = np.arange(len(mature_cohorts))
        ax4.bar(x_pos_renewal, mature_cohorts['renewal_rate_cohort'], color='#9b59b6', alpha=0.8)
        ax4.set_xticks(x_pos_renewal)
        ax4.set_xticklabels(mature_cohorts['cohort_week_str'], rotation=45, ha="right", fontsize=8)
    else:
        ax4.text(0.5, 0.5, "No mature cohorts\nfor renewal analysis", 
                 ha='center', va='center', transform=ax4.transAxes, fontsize=12)

    ax4.set_title('Renewal Rate by Mature Cohort', fontweight='bold')
    ax4.set_ylabel('Renewal Rate (%)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# %%
# Run visualizations
create_improved_visualizations()
create_cohort_analysis()