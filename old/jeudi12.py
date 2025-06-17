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
plt.rcParams['xtick.color'] = 'black'
#plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'black'
#plt.rcParams['ytick.color'] = 'white'
plt.rcParams['figure.figsize'] = (22, 11)

# Grid with opacity and in background
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = 'lightgray'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['axes.axisbelow'] = True

plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlecolor'] = 'black'
#plt.rcParams['axes.titlecolor'] = 'white'
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
start_date = pd.Timestamp('2023-09-25 09:04:00+0000', tz='UTC')

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
        'id': 'subscription_id',
        'Customer ID': 'customer_id',
        'Customer Name': 'customer_name',
        'Status': 'status',
        'Cancellation Reason': 'cancellation_reason',
        'Created (UTC)': 'created',
        'Start (UTC)': 'start_utc',
        'Current Period Start (UTC)': 'current_period_start_utc',
        'Current Period End (UTC)': 'current_period_end_utc',
        'Trial Start (UTC)': 'trial_start',
        'Trial End (UTC)': 'trial_end',
        'Canceled At (UTC)': 'canceled_at',
        'Ended At (UTC)': 'ended_at',
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
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fast_weekly_analysis_from_csv(df, today_date, export_csv=True):
    """
    OPTIMIZED: Fast weekly analysis without expensive timeline reconstruction
    Uses vectorized operations instead of loops
    """
    print("🚀 FAST WEEKLY SUBSCRIPTION ANALYSIS")
    print("=" * 50)
    
    # Prepare data
    df = df.copy()
    date_cols = ['created', 'trial_start', 'trial_end', 'canceled_at', 'ended_at', 
                 'current_period_start', 'current_period_end']
    
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    
    # Generate weekly periods
    start_date = df['created'].min().normalize()
    end_date = today_date.normalize()
    
    # Create weekly periods (much faster than individual processing)
    week_starts = pd.date_range(start_date, end_date, freq='W-MON')
    
    print(f"📅 Processing {len(week_starts)} weeks from {start_date.date()} to {end_date.date()}")
    
    # Pre-calculate subscription phases for vectorized operations
    df = add_subscription_phases(df, today_date)
    
    weekly_results = []
    
    # Process weeks in batches for speed
    batch_size = 10
    for i in range(0, len(week_starts), batch_size):
        batch_weeks = week_starts[i:i+batch_size]
        print(f"📊 Processing batch {i//batch_size + 1}/{(len(week_starts)-1)//batch_size + 1}")
        
        for week_start in batch_weeks:
            week_end = week_start + pd.Timedelta(days=6, hours=23, minutes=59, seconds=59)
            
            # Fast vectorized calculations
            week_metrics = calculate_week_metrics_vectorized(df, week_start, week_end, today_date)
            weekly_results.append(week_metrics)
    
    # Convert to DataFrame
    weekly_df = pd.DataFrame(weekly_results)
    
    # Add derived metrics
    weekly_df['net_growth'] = weekly_df['new_subscriptions'] - weekly_df['cancellations']
    weekly_df['trial_conversion_rate'] = np.where(
        weekly_df['trial_signups'] > 0,
        weekly_df['trial_conversions'] / weekly_df['trial_signups'] * 100,
        0
    )
    
    print(f"✅ Analysis complete! Processed {len(weekly_df)} weeks")
    
    # Export to CSV
    if export_csv:
        filename = f"weekly_subscription_analysis_{today_date.strftime('%Y%m%d')}.csv"
        weekly_df.to_csv(filename, index=False)
        print(f"💾 Results exported to: {filename}")
        
        # Also export summary
        summary_filename = f"subscription_summary_{today_date.strftime('%Y%m%d')}.csv"
        create_summary_export(df, weekly_df, summary_filename)
        print(f"📋 Summary exported to: {summary_filename}")
    
    return weekly_df

def add_subscription_phases(df, today_date):
    """
    Pre-calculate subscription lifecycle phases for fast filtering
    """
    print("⚡ Pre-calculating subscription phases...")
    
    # Handle gift column - check what exists in your DataFrame
    if 'is_gifted_member' in df.columns:
        df['is_gift'] = df['is_gifted_member'].fillna(False).astype(bool)
    elif 'is_gift' not in df.columns:
        df['is_gift'] = False
    else:
        df['is_gift'] = df['is_gift'].fillna(False).astype(bool)
    
    # Calculate key lifecycle dates
    df['paying_start_date'] = np.where(
        df['trial_end'].notna(),
        df['trial_end'],  # Post-trial: paying starts after trial
        df['created']     # No trial: paying starts immediately
    )
    
    # Subscription status at different points
    df['is_trial_subscription'] = df['trial_start'].notna()
    df['trial_duration_days'] = (df['trial_end'] - df['trial_start']).dt.days.fillna(0)
    
    # Determine final outcomes
    df['trial_converted'] = (
        df['is_trial_subscription'] &
        (df['canceled_at'].isna() | (df['canceled_at'] > df['trial_end']))
    )
    
    df['trial_canceled'] = (
        df['is_trial_subscription'] &
        df['canceled_at'].notna() &
        (df['canceled_at'] <= df['trial_end'])
    )
    
    # Calculate subscription lifetime
    df['subscription_end_date'] = df['ended_at'].fillna(today_date)
    df['subscription_lifetime_days'] = (df['subscription_end_date'] - df['created']).dt.days
    
    # Active periods
    df['was_active'] = df['status'].isin(['active', 'trialing', 'canceled'])
    
    return df

def calculate_week_metrics_vectorized(df, week_start, week_end, today_date):
    """
    Calculate weekly metrics using vectorized operations (MUCH faster)
    """
    
    # 1. NEW SUBSCRIPTIONS THIS WEEK
    new_subs_mask = (df['created'] >= week_start) & (df['created'] <= week_end)
    new_subscriptions = df[new_subs_mask]
    
    # 2. TRIAL SIGNUPS THIS WEEK
    trial_signups_mask = (
        (df['trial_start'] >= week_start) & 
        (df['trial_start'] <= week_end)
    )
    trial_signups = df[trial_signups_mask]
    
    # 3. TRIAL CONVERSIONS THIS WEEK (trial ended and converted)
    trial_conversion_mask = (
        (df['trial_end'] >= week_start) & 
        (df['trial_end'] <= week_end) &
        (df['trial_converted'] == True)
    )
    trial_conversions = df[trial_conversion_mask]
    
    # 4. CANCELLATIONS THIS WEEK
    cancellation_mask = (
        (df['canceled_at'] >= week_start) & 
        (df['canceled_at'] <= week_end)
    )
    cancellations = df[cancellation_mask]
    
    # 5. ACTIVE SUBSCRIPTIONS AT END OF WEEK
    # Subscription was created before week end AND (not canceled OR canceled after week end)
    active_at_week_end_mask = (
        (df['created'] <= week_end) &
        (
            df['canceled_at'].isna() |
            (df['canceled_at'] > week_end)
        ) &
        (
            df['ended_at'].isna() |
            (df['ended_at'] > week_end)
        )
    )
    active_at_week_end = df[active_at_week_end_mask]
    
    # 6. FULL MEMBERS AT END OF WEEK (active + past 14-day refund period)
    refund_cutoff = week_end - pd.Timedelta(days=14)
    full_members_mask = (
        active_at_week_end_mask &
        (df['paying_start_date'] <= refund_cutoff) &
        (~df['is_gift'])
    )
    full_members = df[full_members_mask]
    
    return {
        'year_week': f"{week_start.year}-W{week_start.isocalendar().week:02d}",
        'week_start': week_start,
        'week_end': week_end,
        
        # Weekly events
        'new_subscriptions': len(new_subscriptions),
        'trial_signups': len(trial_signups),
        'trial_conversions': len(trial_conversions),
        'cancellations': len(cancellations),
        'trial_cancellations': len(cancellations[cancellations['trial_canceled']]),
        
        # Member counts at end of week
        'active_subscriptions_end': len(active_at_week_end),
        'full_members_end': len(full_members),
        'gift_subscriptions_end': len(active_at_week_end[active_at_week_end['is_gift']]),
        
        # Member types in new subscriptions
        'new_trial_subscriptions': len(new_subscriptions[new_subscriptions['is_trial_subscription']]),
        'new_immediate_paid': len(new_subscriptions[~new_subscriptions['is_trial_subscription']]),
        'new_gift_subscriptions': len(new_subscriptions[new_subscriptions['is_gift']]),
    }

def create_customer_journey_analysis_fast(df):
    """
    Fast customer journey analysis using groupby operations
    """
    print("\n🛤️ FAST CUSTOMER JOURNEY ANALYSIS")
    print("=" * 40)
    
    # Check what columns actually exist in your DataFrame
    print(f"🔍 Available columns: {list(df.columns)}")
    
    # Check what columns exist and create aggregation dict accordingly
    agg_dict = {
        'created': ['min', 'max', 'count'],
        'status': lambda x: list(x),
    }
    
    # Add optional columns if they exist (using your actual column names)
    if 'trial_start' in df.columns:
        agg_dict['trial_start'] = 'count'
    if 'canceled_at' in df.columns:
        agg_dict['canceled_at'] = 'count'
    
    # Check for gift column - could be is_gift, is_gifted_member, etc.
    gift_column = None
    for col in ['is_gift', 'is_gifted_member', 'gift']:
        if col in df.columns:
            gift_column = col
            agg_dict[col] = 'any'
            break
    
    # Group by customer for vectorized analysis
    customer_stats = df.groupby('customer_id').agg(agg_dict).reset_index()
    
    # Flatten column names based on what we actually have
    base_columns = ['customer_id', 'first_subscription', 'last_subscription', 'subscription_count', 'status_history']
    
    new_columns = base_columns.copy()
    
    if 'trial_start' in df.columns:
        new_columns.append('trial_count')
    if 'canceled_at' in df.columns:
        new_columns.append('cancellation_count')
    if gift_column:
        new_columns.append('ever_had_gift')
    
    customer_stats.columns = new_columns
    
    # Add missing columns with defaults if they don't exist
    if 'trial_count' not in customer_stats.columns:
        customer_stats['trial_count'] = 0
    if 'cancellation_count' not in customer_stats.columns:
        customer_stats['cancellation_count'] = 0
    if 'ever_had_gift' not in customer_stats.columns:
        customer_stats['ever_had_gift'] = False
    
    # Calculate total lifetime days manually (much simpler than trying to use non-existent column)
    customer_stats['total_lifetime_days'] = (
        customer_stats['last_subscription'] - customer_stats['first_subscription']
    ).dt.days
    
    # Classify journey types
    customer_stats['journey_type'] = np.select([
        customer_stats['subscription_count'] == 1,
        (customer_stats['subscription_count'] == 2) & customer_stats['ever_had_gift'],
        customer_stats['subscription_count'] >= 2
    ], [
        'Continuous',
        'Gift→Upgrade', 
        'Churn→Return'
    ], default='Complex')
    
    # Calculate customer lifetime (same as total_lifetime_days, keeping for compatibility)
    customer_stats['customer_lifetime_days'] = customer_stats['total_lifetime_days']
    
    print(f"📊 Customer journey distribution:")
    print(customer_stats['journey_type'].value_counts())
    
    return customer_stats

def create_cohort_analysis_fast(df):
    """
    Fast cohort analysis using pandas operations
    """
    print("\n👥 FAST COHORT ANALYSIS")
    print("=" * 30)
    
    # Create cohort based on first subscription week
    df['cohort_week'] = df['created'].dt.to_period('W-MON')
    
    # Check what columns actually exist and build aggregation dynamically
    agg_dict = {
        'created': 'count',
        'canceled_at': lambda x: x.notna().sum(),
        'status': lambda x: (x == 'active').sum(),
    }
    
    # Add trial columns if they exist
    if 'trial_start' in df.columns:
        agg_dict['trial_start'] = 'count'
    
    # Add derived columns if they exist (these are created in add_subscription_phases)
    if 'trial_converted' in df.columns:
        agg_dict['trial_converted'] = 'sum'
    if 'trial_canceled' in df.columns:
        agg_dict['trial_canceled'] = 'sum'
    
    # Add gift column (check multiple possible names)
    gift_column = None
    for col in ['is_gift', 'is_gifted_member']:
        if col in df.columns:
            gift_column = col
            agg_dict[col] = 'sum'
            break
    
    # Aggregate by cohort
    cohort_metrics = df.groupby('cohort_week').agg(agg_dict).reset_index()
    
    # Build column names based on what we actually aggregated
    new_columns = ['cohort_week', 'cohort_size', 'total_cancellations', 'currently_active']
    
    if 'trial_start' in df.columns:
        new_columns.insert(2, 'trial_signups')
    if 'trial_converted' in df.columns:
        new_columns.append('trial_conversions')
    if 'trial_canceled' in df.columns:
        new_columns.append('trial_cancellations')
    if gift_column:
        new_columns.append('gift_subscriptions')
    
    # Handle column assignment safely
    expected_cols = len(new_columns)
    actual_cols = len(cohort_metrics.columns)
    
    if expected_cols == actual_cols:
        cohort_metrics.columns = new_columns
    else:
        # Fallback: keep original column names and add prefixes
        print(f"⚠️ Column mismatch: expected {expected_cols}, got {actual_cols}")
        print(f"Keeping original columns: {list(cohort_metrics.columns)}")
    
    # Add missing columns with defaults
    required_columns = ['trial_signups', 'trial_conversions', 'trial_cancellations', 'gift_subscriptions']
    for col in required_columns:
        if col not in cohort_metrics.columns:
            cohort_metrics[col] = 0
    
    # Calculate rates (with safe division)
    cohort_metrics['trial_conversion_rate'] = np.where(
        cohort_metrics['trial_signups'] > 0,
        cohort_metrics['trial_conversions'] / cohort_metrics['trial_signups'] * 100,
        0
    )
    
    cohort_metrics['retention_rate'] = np.where(
        cohort_metrics['cohort_size'] > 0,
        cohort_metrics['currently_active'] / cohort_metrics['cohort_size'] * 100,
        0
    )
    
    cohort_metrics['cohort_week_str'] = cohort_metrics['cohort_week'].astype(str)
    
    print(f"📈 Analyzed {len(cohort_metrics)} cohorts")
    if len(cohort_metrics) > 0:
        print(f"📊 Average cohort size: {cohort_metrics['cohort_size'].mean():.1f}")
    
    return cohort_metrics

def create_summary_export(df, weekly_df, filename):
    """
    Create a summary CSV with key insights
    """
    
    # Overall metrics
    total_customers = df['customer_id'].nunique()
    total_subscriptions = len(df)
    active_subscriptions = len(df[df['status'] == 'active'])
    trial_conversion_rate = df[df['is_trial_subscription']]['trial_converted'].mean() * 100
    
    # Recent performance (last 4 weeks)
    recent_weeks = weekly_df.tail(4)
    avg_weekly_signups = recent_weeks['new_subscriptions'].mean()
    avg_weekly_conversions = recent_weeks['trial_conversions'].mean()
    
    # Growth metrics
    if len(weekly_df) >= 2:
        growth_rate = ((weekly_df.iloc[-1]['active_subscriptions_end'] / 
                       weekly_df.iloc[0]['active_subscriptions_end']) - 1) * 100
    else:
        growth_rate = 0
    
    summary_data = {
        'Metric': [
            'Total Customers',
            'Total Subscriptions', 
            'Active Subscriptions',
            'Trial Conversion Rate (%)',
            'Avg Weekly Signups (Last 4 weeks)',
            'Avg Weekly Conversions (Last 4 weeks)',
            'Overall Growth Rate (%)',
            'Analysis Date',
            'Weeks Analyzed'
        ],
        'Value': [
            total_customers,
            total_subscriptions,
            active_subscriptions,
            f"{trial_conversion_rate:.1f}%",
            f"{avg_weekly_signups:.1f}",
            f"{avg_weekly_conversions:.1f}",
            f"{growth_rate:.1f}%",
            pd.Timestamp.now().strftime('%Y-%m-%d'),
            len(weekly_df)
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(filename, index=False)

def run_optimized_analysis(df, today_date, export_csv=True):
    """
    Run the complete optimized analysis pipeline
    """
    print("🚀 OPTIMIZED SUBSCRIPTION ANALYSIS PIPELINE")
    print("=" * 60)
    
    start_time = pd.Timestamp.now()
    
    # 1. Fast weekly analysis (this also creates derived columns)
    weekly_df = fast_weekly_analysis_from_csv(df, today_date, export_csv)
    
    # 2. Fast customer journey analysis (uses original df)
    customer_journeys = create_customer_journey_analysis_fast(df)
    
    # 3. Fast cohort analysis (also uses original df - we'll calculate what we can)
    cohort_analysis = create_cohort_analysis_fast(df)
    
    # Export additional analyses if requested
    if export_csv:
        customer_journeys.to_csv(f"customer_journeys_{today_date.strftime('%Y%m%d')}.csv", index=False)
        cohort_analysis.to_csv(f"cohort_analysis_{today_date.strftime('%Y%m%d')}.csv", index=False)
        print(f"💾 Customer journeys exported to: customer_journeys_{today_date.strftime('%Y%m%d')}.csv")
        print(f"💾 Cohort analysis exported to: cohort_analysis_{today_date.strftime('%Y%m%d')}.csv")
    
    end_time = pd.Timestamp.now()
    processing_time = (end_time - start_time).total_seconds()
    
    print(f"\n⚡ PERFORMANCE:")
    print(f"   • Processing time: {processing_time:.1f} seconds")
    print(f"   • Weeks processed: {len(weekly_df)}")
    print(f"   • Speed: {len(weekly_df)/processing_time:.1f} weeks/second")
    
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"   • Weekly metrics: {len(weekly_df)} weeks")
    print(f"   • Customer journeys: {len(customer_journeys)} customers")
    print(f"   • Cohorts analyzed: {len(cohort_analysis)} cohorts")
    
    # Show some key insights
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"   • Total subscriptions: {len(df)}")
    print(f"   • Unique customers: {df['customer_id'].nunique()}")
    print(f"   • Current active: {len(df[df['status'] == 'active'])}")
    print(f"   • Customer types: {customer_journeys['journey_type'].value_counts().to_dict()}")
    
    return {
        'weekly_analysis': weekly_df,
        'customer_journeys': customer_journeys,
        'cohort_analysis': cohort_analysis,
        'processing_time_seconds': processing_time
    }

# Usage:
results = run_optimized_analysis(df, today_date, export_csv=True)

def create_corrected_dashboard(df, today_date, export_path="corrected_subscription_dashboard.png"):
    """
    Create a corrected comprehensive dashboard with fixed business logic
    """
    
    print("🔧 CREATING CORRECTED DASHBOARD WITH FIXED LOGIC")
    print("=" * 60)
    
    # Debug current data first
    debug_data_issues(df, today_date)
    
    # Prepare data with corrected business logic
    df_processed = prepare_corrected_data(df, today_date)
    
    # Generate corrected weekly analysis
    weekly_metrics = generate_corrected_weekly_metrics(df_processed, today_date)
    
    # Create the dashboard with better spacing
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3, 
                         height_ratios=[1, 1, 1, 0.6], width_ratios=[1, 1, 1])
    
    # Convert week_start to datetime for plotting
    weekly_metrics['week_date'] = pd.to_datetime(weekly_metrics['week_start'])
    
    # Color scheme
    colors = {
        'primary': '#2E86AB',
        'secondary': '#A23B72', 
        'success': '#27AE60',
        'warning': '#F39C12',
        'danger': '#E74C3C',
        'info': '#8E44AD'
    }
    
    # 1. FULL MEMBERS GROWTH (Top Left) - CORRECTED
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(weekly_metrics['week_date'], weekly_metrics['full_members_at_start'], 
             linewidth=3, color=colors['primary'], marker='o', markersize=2)
    ax1.set_title('Full Members Growth\n(Corrected)', fontsize=12, fontweight='bold', pad=15)
    ax1.set_ylabel('Full Members Count', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Add trend line
    if len(weekly_metrics) > 1:
        z = np.polyfit(range(len(weekly_metrics)), weekly_metrics['full_members_at_start'], 1)
        p = np.poly1d(z)
        ax1.plot(weekly_metrics['week_date'], p(range(len(weekly_metrics))), 
                 "--", alpha=0.7, color=colors['secondary'], linewidth=2)
    
    # 2. WEEKLY NEW VS CHURNED MEMBERS (Top Center) - CORRECTED
    ax2 = fig.add_subplot(gs[0, 1])
    width = pd.Timedelta(days=2)
    ax2.bar(weekly_metrics['week_date'] - width/2, weekly_metrics['new_full_members'], 
            width=width, label='New Full Members', color=colors['success'], alpha=0.8)
    ax2.bar(weekly_metrics['week_date'] + width/2, weekly_metrics['churned_full_members'], 
            width=width, label='Churned Members', color=colors['danger'], alpha=0.8)
    ax2.set_title('Weekly New vs Churned\nFull Members (Corrected)', fontsize=12, fontweight='bold', pad=15)
    ax2.set_ylabel('Members Count', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    
    # 3. TRIAL SIGNUPS & CONVERSIONS (Top Right) - CORRECTED
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(weekly_metrics['week_date'], weekly_metrics['trial_signups'], 
            width=pd.Timedelta(days=4), label='Trial Signups', color=colors['info'], alpha=0.7)
    ax3.bar(weekly_metrics['week_date'], weekly_metrics['trial_conversions'], 
            width=pd.Timedelta(days=4), label='Trial Conversions', color=colors['success'], alpha=0.9)
    ax3.set_title('Weekly Trial Signups\nvs Conversions (Corrected)', fontsize=12, fontweight='bold', pad=15)
    ax3.set_ylabel('Count', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45, labelsize=8)
    
    # 4. CONVERSION RATE TRENDS (Second Row Left) - CORRECTED
    ax4 = fig.add_subplot(gs[1, 0])
    # Apply smoothing to reduce noise
    conversion_rate_smooth = weekly_metrics['trial_conversion_rate'].rolling(window=4, min_periods=1).mean()
    ax4.plot(weekly_metrics['week_date'], conversion_rate_smooth, 
             linewidth=3, color=colors['warning'], marker='o', markersize=2)
    ax4.set_title('Trial → Full Member\nConversion Rate (4-week MA)', fontsize=12, fontweight='bold', pad=15)
    ax4.set_ylabel('Conversion Rate (%)', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45, labelsize=8)
    ax4.set_ylim(0, 100)  # Cap at 100%
    
    # Add average line
    avg_conversion = weekly_metrics['trial_conversion_rate'].mean()
    ax4.axhline(y=avg_conversion, color=colors['danger'], linestyle='--', alpha=0.7, 
                label=f'Average: {avg_conversion:.1f}%')
    ax4.legend(fontsize=9)
    
    # 5. TRIAL CANCELLATIONS & REFUNDS (Second Row Center) - CORRECTED
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(weekly_metrics['week_date'] - width/2, weekly_metrics['trial_cancellations'], 
            width=width, label='Trial Cancellations', color=colors['danger'], alpha=0.7)
    ax5.bar(weekly_metrics['week_date'] + width/2, weekly_metrics['refund_requests'], 
            width=width, label='Refund Requests', color=colors['warning'], alpha=0.7)
    ax5.set_title('Weekly Trial Cancellations\n& Refund Requests', fontsize=12, fontweight='bold', pad=15)
    ax5.set_ylabel('Count', fontsize=10)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(axis='x', rotation=45, labelsize=8)
    
    # 6. NET GROWTH (Second Row Right) - CORRECTED
    ax6 = fig.add_subplot(gs[1, 2])
    net_growth = weekly_metrics['new_full_members'] - weekly_metrics['churned_full_members']
    colors_net = [colors['success'] if x >= 0 else colors['danger'] for x in net_growth]
    ax6.bar(weekly_metrics['week_date'], net_growth, 
            width=pd.Timedelta(days=4), color=colors_net, alpha=0.7)
    ax6.set_title('Weekly Net Growth\n(New - Churned)', fontsize=12, fontweight='bold', pad=15)
    ax6.set_ylabel('Net Members', fontsize=10)
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(axis='x', rotation=45, labelsize=8)
    
    # 7. YEARLY RENEWAL ANALYSIS (Third Row Left) - CORRECTED
    ax7 = fig.add_subplot(gs[2, 0])
    if weekly_metrics['year1_completions'].sum() > 0:
        ax7.bar(weekly_metrics['week_date'] - width/2, weekly_metrics['year1_completions'], 
                width=width, label='Year 1 Completions', color=colors['success'], alpha=0.8)
        ax7.bar(weekly_metrics['week_date'] + width/2, weekly_metrics['year1_cancellations'], 
                width=width, label='Year 1 Cancellations', color=colors['danger'], alpha=0.8)
        ax7.set_title('Year 1 → Year 2\nRenewal Analysis', fontsize=12, fontweight='bold', pad=15)
        ax7.set_ylabel('Count', fontsize=10)
        ax7.legend(fontsize=9)
    else:
        # Show what data we do have
        total_year1_eligible = len(df_processed[df_processed['subscription_age_days'] >= 365])
        ax7.text(0.5, 0.5, f'Year 1 → Year 2 Renewal Analysis\n\n{total_year1_eligible} customers have\nreached 1+ year\n\nNeed more time-series data\nfor weekly renewal tracking', 
                ha='center', va='center', transform=ax7.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        ax7.set_title('Year 1 → Year 2\nRenewal Analysis', fontsize=12, fontweight='bold', pad=15)
    ax7.grid(True, alpha=0.3)
    ax7.tick_params(axis='x', rotation=45, labelsize=8)
    
    # 8. CUMULATIVE METRICS (Third Row Center) - CORRECTED
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(weekly_metrics['week_date'], weekly_metrics['trial_signups'].cumsum(), 
             linewidth=3, color=colors['info'], label='Cumulative Trials', marker='o', markersize=2)
    ax8.plot(weekly_metrics['week_date'], weekly_metrics['trial_conversions'].cumsum(), 
             linewidth=3, color=colors['success'], label='Cumulative Conversions', marker='s', markersize=2)
    ax8.set_title('Cumulative Trial Signups\n& Conversions', fontsize=12, fontweight='bold', pad=15)
    ax8.set_ylabel('Cumulative Count', fontsize=10)
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)
    ax8.tick_params(axis='x', rotation=45, labelsize=8)
    
    # 9. REFUND RATE TRENDS (Third Row Right) - CORRECTED WITH CAP
    ax9 = fig.add_subplot(gs[2, 2])
    # Fixed refund rate calculation with reasonable cap
    refund_rate = np.where(
        weekly_metrics['new_full_members'] > 0,
        np.minimum(weekly_metrics['refund_requests'] / weekly_metrics['new_full_members'] * 100, 100),
        0
    )
    ax9.plot(weekly_metrics['week_date'], refund_rate, 
             linewidth=3, color=colors['warning'], marker='o', markersize=2)
    ax9.set_title('Weekly Refund Rate\n(Capped at 100%)', fontsize=12, fontweight='bold', pad=15)
    ax9.set_ylabel('Refund Rate (%)', fontsize=10)
    ax9.set_ylim(0, 100)  # Cap at reasonable level
    ax9.grid(True, alpha=0.3)
    ax9.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Add average line
    avg_refund_rate = refund_rate[refund_rate > 0].mean()
    if not np.isnan(avg_refund_rate):
        ax9.axhline(y=avg_refund_rate, color=colors['danger'], linestyle='--', alpha=0.7, 
                    label=f'Average: {avg_refund_rate:.1f}%')
        ax9.legend(fontsize=9)
    
    # 10. CORRECTED KEY METRICS SUMMARY (Bottom Row)
    ax10 = fig.add_subplot(gs[3, :])
    ax10.axis('off')
    
    # Calculate corrected summary metrics
    total_customers = df_processed['customer_id'].nunique()
    total_subscriptions = len(df_processed)
    current_full_members = weekly_metrics['full_members_at_start'].iloc[-1] if len(weekly_metrics) > 0 else 0
    total_trials = weekly_metrics['trial_signups'].sum()
    total_conversions = weekly_metrics['trial_conversions'].sum()
    total_churned = weekly_metrics['churned_full_members'].sum()
    total_refunds = weekly_metrics['refund_requests'].sum()
    
    overall_conversion_rate = (total_conversions / total_trials * 100) if total_trials > 0 else 0
    avg_conversion_rate = weekly_metrics['trial_conversion_rate'].mean()
    
    # Growth calculation
    if len(weekly_metrics) >= 2:
        first_week_members = weekly_metrics['full_members_at_start'].iloc[0]
        growth_rate = ((current_full_members - first_week_members) / first_week_members * 100) if first_week_members > 0 else 0
    else:
        growth_rate = 0
    
    # Churn rate
    total_ever_full_members = len(df_processed[df_processed['ever_was_full_member'] == True])
    churn_rate = (total_churned / total_ever_full_members * 100) if total_ever_full_members > 0 else 0
    
    # Create corrected summary text with better formatting
    summary_text = f"""CORRECTED KEY METRICS SUMMARY (Sept 2023 - May 2025)

CUSTOMER BASE:  {total_customers:,} unique customers  |  {total_subscriptions:,} total subscriptions  |  {current_full_members:,} current full members

TRIAL PERFORMANCE:  {total_trials:,} trial signups  |  {total_conversions:,} conversions  |  {overall_conversion_rate:.1f}% overall conversion rate

GROWTH & RETENTION:  {growth_rate:+.1f}% full member growth since Sept 2023  |  {avg_conversion_rate:.1f}% average weekly conversion

CHURN ANALYSIS:  {total_churned:,} total churned members  |  {churn_rate:.1f}% historical churn rate  |  {total_refunds:,} total refund requests"""
    
    ax10.text(0.5, 0.5, summary_text, ha='center', va='center', 
              transform=ax10.transAxes, fontsize=12, 
              bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.1))
    
    # Add title and subtitle with better spacing
    fig.suptitle('CORRECTED SUBSCRIPTION ANALYTICS DASHBOARD', 
                fontsize=20, fontweight='bold', y=0.97)
    fig.text(0.5, 0.94, f'Analysis Period: September 2023 - May 2025  |  Generated: {datetime.now().strftime("%B %d, %Y")} | FIXED LOGIC',
             ha='center', fontsize=12, style='italic', color='green')
    
    # Save the dashboard
    plt.savefig(export_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"💾 Corrected dashboard exported to: {export_path}")
    
    # Show the plot
    plt.show()
    
    return fig, weekly_metrics

def debug_data_issues(df, today_date):
    """
    Debug the data issues found in the original dashboard
    """
    print("🔍 DEBUGGING DATA ISSUES")
    print("=" * 40)
    
    # Check subscription statuses
    print("📊 Subscription status distribution:")
    print(df['status'].value_counts())
    
    # Check cancellation data
    canceled_count = len(df[df['canceled_at'].notna()])
    ended_count = len(df[df['ended_at'].notna()])
    print(f"\n❌ Cancellation data:")
    print(f"   • Subscriptions with canceled_at: {canceled_count:,}")
    print(f"   • Subscriptions with ended_at: {ended_count:,}")
    
    # Check trial data
    trial_count = len(df[df['trial_start'].notna()])
    print(f"\n🧪 Trial data:")
    print(f"   • Subscriptions with trials: {trial_count:,}")
    
    # Check gift data
    gift_count = len(df[df['is_gifted_member'] == True])
    print(f"\n🎁 Gift data:")
    print(f"   • Gift subscriptions: {gift_count:,}")
    
    # Check date ranges
    print(f"\n📅 Date ranges:")
    print(f"   • Created: {df['created'].min()} to {df['created'].max()}")
    if 'canceled_at' in df.columns:
        valid_canceled = df['canceled_at'].dropna()
        if len(valid_canceled) > 0:
            print(f"   • Canceled: {valid_canceled.min()} to {valid_canceled.max()}")

def prepare_corrected_data(df, today_date):
    """
    Prepare data with corrected business logic
    """
    print("\n⚡ APPLYING CORRECTED BUSINESS LOGIC")
    print("=" * 45)
    
    df = df.copy()
    
    # Ensure date columns are datetime
    date_cols = ['created', 'trial_start', 'trial_end', 'canceled_at', 'ended_at']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    
    # Handle gift members properly
    df['is_gift'] = df['is_gifted_member'].fillna(False).astype(bool)
    
    # Calculate when paying membership starts (key correction)
    df['paying_start_date'] = np.where(
        df['trial_end'].notna(),
        df['trial_end'],  # After trial completion
        df['created']     # Immediate for non-trial
    )
    
    # Trial analysis (corrected)
    df['had_trial'] = df['trial_start'].notna()
    
    # CORRECTED: Trial conversion logic
    df['trial_converted'] = (
        df['had_trial'] &
        (
            # Either never canceled, OR canceled after trial ended
            (df['canceled_at'].isna()) |
            (df['canceled_at'] > df['trial_end'])
        ) &
        # And trial actually ended (not ongoing)
        (df['trial_end'] <= today_date)
    )
    
    # CORRECTED: Trial cancellation logic
    df['trial_canceled'] = (
        df['had_trial'] &
        df['canceled_at'].notna() &
        (df['canceled_at'] <= df['trial_end'])
    )
    
    # CORRECTED: Refund period calculation (14 days after becoming paying customer)
    df['refund_period_end'] = df['paying_start_date'] + pd.Timedelta(days=14)
    
    # CORRECTED: Refund request logic
    df['requested_refund'] = (
        df['canceled_at'].notna() &
        (df['canceled_at'] <= df['refund_period_end']) &
        (df['canceled_at'] > df['paying_start_date'])  # Must be after becoming paying customer
    )
    
    # CORRECTED: Full member definition
    df['is_full_member'] = (
        (df['status'] == 'active') &
        (~df['is_gift']) &
        (df['paying_start_date'] + pd.Timedelta(days=14) <= today_date)  # Past refund period
    )
    
    # Track who was ever a full member (for churn calculation)
    df['ever_was_full_member'] = (
        (~df['is_gift']) &
        (
            (df['status'] == 'active') |  # Currently active
            (df['canceled_at'] > df['paying_start_date'] + pd.Timedelta(days=14))  # Was active past refund period
        )
    )
    
    # CORRECTED: Churn definition - was full member and then canceled/ended
    df['churned_full_member'] = (
        df['ever_was_full_member'] &
        (
            (df['status'] == 'canceled') |
            (df['ended_at'].notna())
        )
    )
    
    # Year analysis
    df['subscription_age_days'] = (today_date - df['paying_start_date']).dt.days
    df['completed_year_1'] = df['subscription_age_days'] >= 365
    
    # Year 2 analysis
    year_2_start = df['paying_start_date'] + pd.Timedelta(days=365)
    year_2_refund_end = year_2_start + pd.Timedelta(days=14)
    
    df['year_2_refund'] = (
        df['completed_year_1'] &
        df['canceled_at'].notna() &
        (df['canceled_at'] >= year_2_start) &
        (df['canceled_at'] <= year_2_refund_end)
    )
    
    df['canceled_during_year_1'] = (
        df['canceled_at'].notna() &
        (df['canceled_at'] < year_2_start) &
        (df['canceled_at'] > df['paying_start_date'])
    )
    
    print(f"✅ Data prepared:")
    print(f"   • Total subscriptions: {len(df):,}")
    print(f"   • Current full members: {df['is_full_member'].sum():,}")
    print(f"   • Ever full members: {df['ever_was_full_member'].sum():,}")
    print(f"   • Churned full members: {df['churned_full_member'].sum():,}")
    print(f"   • Trial conversions: {df['trial_converted'].sum():,}")
    
    return df

def generate_corrected_weekly_metrics(df, today_date):
    """
    Generate corrected weekly metrics with proper business logic
    """
    print("\n📊 GENERATING CORRECTED WEEKLY METRICS")
    print("=" * 45)
    
    # Generate weekly periods
    start_date = df['created'].min().normalize()
    end_date = today_date.normalize()
    week_starts = pd.date_range(start_date, end_date, freq='W-MON')
    
    weekly_results = []
    
    for i, week_start in enumerate(week_starts):
        week_end = week_start + pd.Timedelta(days=6, hours=23, minutes=59, seconds=59)
        
        if i % 20 == 0:  # Progress indicator
            print(f"   Processing week {i+1}/{len(week_starts)}: {week_start.strftime('%Y-%m-%d')}")
        
        # 1. CORRECTED: Full members at start of week
        full_members_at_start = len(df[
            (df['paying_start_date'] + pd.Timedelta(days=14) < week_start) &  # Became full member before week
            (df['is_full_member'] | df['ever_was_full_member']) &  # Is or was full member
            (
                (df['canceled_at'].isna()) |  # Never canceled OR
                (df['canceled_at'] >= week_start)  # Canceled after/during this week
            )
        ])
        
        # 2. CORRECTED: New full members during week (completed refund period)
        new_full_members = len(df[
            (df['paying_start_date'] + pd.Timedelta(days=14) >= week_start) &
            (df['paying_start_date'] + pd.Timedelta(days=14) <= week_end) &
            (df['ever_was_full_member'] == True)
        ])
        
        # 3. CORRECTED: Churned full members during week
        churned_full_members = len(df[
            (
                (df['canceled_at'] >= week_start) & (df['canceled_at'] <= week_end)
            ) |
            (
                (df['ended_at'] >= week_start) & (df['ended_at'] <= week_end)
            ) &
            (df['ever_was_full_member'] == True)
        ])
        
        # 4. Trial signups during week
        trial_signups = len(df[
            (df['trial_start'] >= week_start) &
            (df['trial_start'] <= week_end)
        ])
        
        # 5. Trial conversions during week (trial ended and converted)
        trial_conversions = len(df[
            (df['trial_end'] >= week_start) &
            (df['trial_end'] <= week_end) &
            (df['trial_converted'] == True)
        ])
        
        # 6. Trial cancellations during week
        trial_cancellations = len(df[
            (df['canceled_at'] >= week_start) &
            (df['canceled_at'] <= week_end) &
            (df['trial_canceled'] == True)
        ])
        
        # 7. Refund requests during week
        refund_requests = len(df[
            (df['canceled_at'] >= week_start) &
            (df['canceled_at'] <= week_end) &
            (df['requested_refund'] == True)
        ])
        
        # 8. Year 1 completions
        year_1_completions = len(df[
            (df['paying_start_date'] + pd.Timedelta(days=365) >= week_start) &
            (df['paying_start_date'] + pd.Timedelta(days=365) <= week_end) &
            (df['completed_year_1'] == True) &
            (df['status'] == 'active')  # Still active after 1 year
        ])
        
        # 9. Year 1 cancellations
        year_1_cancellations = len(df[
            (df['canceled_at'] >= week_start) &
            (df['canceled_at'] <= week_end) &
            (df['canceled_during_year_1'] == True)
        ])
        
        # 10. Year 2 refunds
        year_2_refunds = len(df[
            (df['canceled_at'] >= week_start) &
            (df['canceled_at'] <= week_end) &
            (df['year_2_refund'] == True)
        ])
        
        weekly_results.append({
            'year_week': f"{week_start.year}-W{week_start.isocalendar().week:02d}",
            'week_start': week_start,
            'week_end': week_end,
            'full_members_at_start': full_members_at_start,
            'new_full_members': new_full_members,
            'churned_full_members': churned_full_members,
            'trial_signups': trial_signups,
            'trial_conversions': trial_conversions,
            'trial_cancellations': trial_cancellations,
            'refund_requests': refund_requests,
            'year1_completions': year_1_completions,
            'year1_cancellations': year_1_cancellations,
            'year2_refunds': year_2_refunds,
            'trial_conversion_rate': (trial_conversions / trial_signups * 100) if trial_signups > 0 else 0
        })
    
    weekly_df = pd.DataFrame(weekly_results)
    
    print(f"✅ Generated corrected weekly metrics:")
    print(f"   • Total weeks: {len(weekly_df)}")
    print(f"   • Total new full members: {weekly_df['new_full_members'].sum():,}")
    print(f"   • Total churned members: {weekly_df['churned_full_members'].sum():,}")
    print(f"   • Total trial signups: {weekly_df['trial_signups'].sum():,}")
    print(f"   • Total trial conversions: {weekly_df['trial_conversions'].sum():,}")
    
    return weekly_df

def run_corrected_dashboard(df, today_date):
    """
    Main function to run the corrected dashboard
    """
    print("🚀 RUNNING CORRECTED SUBSCRIPTION DASHBOARD")
    print("=" * 60)
    
    # Create and export the corrected dashboard
    fig, weekly_metrics = create_corrected_dashboard(df, today_date)
    
    # Export the corrected weekly metrics
    weekly_metrics.to_csv("corrected_weekly_metrics.csv", index=False)
    print("💾 Corrected weekly metrics exported to: corrected_weekly_metrics.csv")
    
    return fig, weekly_metrics

# Run the corrected dashboard
fig, metrics = run_corrected_dashboard(df, today_date)
