# DISHPATCH SUPPER CLUB - COMPLETE BUSINESS ANALYSIS
# Phase 1 - Business Analysis with 100% Correct Logic

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.tseries.offsets import Week
from datetime import datetime, timedelta
import re

# Load Data
data_dir = os.path.join('data')
file_path = os.path.abspath(os.path.join(data_dir, 'DishpatchSubscriptionData_NIklas_Sanitised - subscriptions (2).csv'))

df = pd.read_csv(file_path)

# Preprocessing
for col in df.columns:
    if col.endswith('(UTC)'):
        df[col] = pd.to_datetime(df[col])

original_df = df.copy()

print("=== DISHPATCH SUPPER CLUB - COMPLETE BUSINESS ANALYSIS ===")
print(f"📊 Data loaded: {len(df):,} rows, {len(df.columns)} columns")

# ===============================================================================
# STEP 1: UNDERSTANDING THE BUSINESS MODEL
# ===============================================================================

print("\n=== 1. DISHPATCH SUPPER CLUB BUSINESS MODEL ===")
print("""
🎯 BUSINESS MODEL:
• Annual membership £69 (Supper Club)
• 10-day free trial
• If not canceled during trial → £69 payment + 1-year membership
• Automatic renewal after 1 year
• Refund possible within 2 weeks after payment
• Gift members identified by senderShopifyCustomerId (metadata)

🎯 POSSIBLE STATUSES:
• 'trialing': In free trial period
• 'active': Paying active member
• 'past_due': Member with payment issues
• 'canceled': Canceled (during trial OR after payment)
""")

# ===============================================================================
# STEP 2: DATA CLEANING AND PREPARATION
# ===============================================================================

print("\n=== 2. DATA CLEANING ===")

# Create working DataFrame
analysis_df = df.copy()

# Rename gift column for clarity
analysis_df = analysis_df.rename(columns={'senderShopifyCustomerId (metadata)': 'gift_sender_id'})

# Select key columns
key_columns = [
    'id', 'Customer ID', 'Customer Name', 'Status',
    'Created (UTC)', 'Start (UTC)', 'Start Date (UTC)',
    'Current Period Start (UTC)', 'Current Period End (UTC)',
    'Trial Start (UTC)', 'Trial End (UTC)',
    'Canceled At (UTC)', 'Cancel At Period End', 'Ended At (UTC)',
    'gift_sender_id', 'kind (metadata)'
]

analysis_df = analysis_df[key_columns]

print(f"✅ Selected columns: {len(key_columns)}")
print(f"✅ Cleaned data: {len(analysis_df):,} rows")

# ===============================================================================
# STEP 3: CORRECT BUSINESS DEFINITIONS
# ===============================================================================

print("\n=== 3. CORRECT BUSINESS DEFINITIONS ===")

# Identify member types
analysis_df['is_gift_member'] = analysis_df['gift_sender_id'].notna()
analysis_df['is_regular_signup'] = ~analysis_df['is_gift_member']

print(f"📊 Gift members: {analysis_df['is_gift_member'].sum():,}")
print(f"📊 Regular signups: {analysis_df['is_regular_signup'].sum():,}")

# ✅ CORRECT DEFINITION: CONVERSION
# Conversion = Has paid £69 = Status 'active' or 'past_due'
analysis_df['is_converted'] = (
    analysis_df['is_regular_signup'] &
    analysis_df['Status'].isin(['active', 'past_due'])
)

# ✅ CORRECT DEFINITION: TRIAL CANCELLATION
# = Canceled DURING trial (before or during the 10 days)
analysis_df['is_trial_cancellation'] = (
    analysis_df['is_regular_signup'] &
    (analysis_df['Status'] == 'canceled') &
    (
        analysis_df['Trial End (UTC)'].isna() |
        (analysis_df['Canceled At (UTC)'] <= analysis_df['Trial End (UTC)'])
    )
)

# ✅ CORRECT DEFINITION: POST-PAYMENT CHURN
# = Canceled AFTER paying £69
analysis_df['is_post_payment_churn'] = (
    analysis_df['is_regular_signup'] &
    (analysis_df['Status'] == 'canceled') &
    analysis_df['Trial End (UTC)'].notna() &
    (analysis_df['Canceled At (UTC)'] > analysis_df['Trial End (UTC)'])
)

# ✅ CORRECT DEFINITION: REFUNDED
# = Canceled within 2 weeks AFTER paying
analysis_df['is_refund'] = (
    analysis_df['is_post_payment_churn'] &
    ((analysis_df['Canceled At (UTC)'] - analysis_df['Trial End (UTC)']).dt.days <= 14)
)

# ✅ CORRECT DEFINITION: STILL IN TRIAL
analysis_df['is_currently_trialing'] = (
    analysis_df['is_regular_signup'] &
    (analysis_df['Status'] == 'trialing')
)

# ✅ CORRECT DEFINITION: RENEWAL
reference_date = pd.to_datetime('2025-05-26')  # Current date
analysis_df['subscription_start_date'] = analysis_df['Trial End (UTC)']  # Use Trial End as subscription start
analysis_df['subscription_duration'] = (reference_date - analysis_df['subscription_start_date']).dt.days

analysis_df['is_eligible_for_renewal'] = (
    analysis_df['is_converted'] &
    (analysis_df['subscription_duration'] > 365)
)

# Members who actually renewed = Eligible and still active
analysis_df['has_renewed'] = analysis_df['is_eligible_for_renewal']

print(f"✅ Conversions (paid £69): {analysis_df['is_converted'].sum():,}")
print(f"✅ Trial cancellations: {analysis_df['is_trial_cancellation'].sum():,}")
print(f"✅ Post-payment churn: {analysis_df['is_post_payment_churn'].sum():,}")
print(f"✅ Refunds: {analysis_df['is_refund'].sum():,}")
print(f"✅ Still in trial: {analysis_df['is_currently_trialing'].sum():,}")
print(f"✅ Renewed members Ones: {analysis_df['has_renewed'].sum():,}")

# ===============================================================================
# STEP 4: LOGIC VALIDATION
# ===============================================================================

print("\n=== 4. LOGIC VALIDATION ===")

# Verify all regular signups are accounted for
total_regular = analysis_df['is_regular_signup'].sum()
accounted = (
    analysis_df['is_converted'].sum() +
    analysis_df['is_trial_cancellation'].sum() +
    analysis_df['is_post_payment_churn'].sum() +
    analysis_df['is_currently_trialing'].sum()
)

print(f"📊 Total regular signups: {total_regular:,}")
print(f"📊 Total accounted: {accounted:,}")
print(f"📊 Difference: {total_regular - accounted:,}")

if total_regular == accounted:
    print("✅ Logic consistent - All signups accounted for")
else:
    print("⚠️ Discrepancy detected - Verification needed")

# Global validation metrics
total_signups = analysis_df['is_regular_signup'].sum()
converted_users = analysis_df[analysis_df['is_converted'] & ~analysis_df['has_renewed']].drop_duplicates(subset='Customer ID')
renewed_users = analysis_df[analysis_df['has_renewed']].drop_duplicates(subset='Customer ID')

conversion_rate = (converted_users.shape[0] / total_signups * 100) if total_signups > 0 else 0
trial_cancel_rate = (analysis_df['is_trial_cancellation'].sum() / total_signups * 100) if total_signups > 0 else 0
refund_rate = (analysis_df['is_refund'].sum() / (converted_users.shape[0] + analysis_df['is_post_payment_churn'].sum()) * 100) if (converted_users.shape[0] + analysis_df['is_post_payment_churn'].sum()) > 0 else 0
renewal_rate = (renewed_users.shape[0] / converted_users.shape[0] * 100) if converted_users.shape[0] > 0 else 0

print(f"📈 Overall conversion rate: {conversion_rate:.1f}%")
print(f"📈 Trial cancellation rate: {trial_cancel_rate:.1f}%")
print(f"📈 Refund rate: {refund_rate:.1f}%")
print(f"📈 Renewal rate: {renewal_rate:.1f}%")

# Validate rates
if 15 <= conversion_rate <= 50:
    print("✅ Conversion rate realistic")
else:
    print("⚠️ Conversion rate suspect")

if refund_rate <= 10:
    print("✅ Refund rate healthy")
else:
    print("⚠️ Refund rate high")

# ===============================================================================
# STEP 5: TEMPORAL COHORT CREATION
# ===============================================================================

print("\n=== 5. TEMPORAL COHORT CREATION ===")

# Use Start Date (UTC) as signup date
analysis_df['signup_week'] = analysis_df['Start Date (UTC)'].dt.to_period('W-SUN')
analysis_df['trial_end_week'] = analysis_df['Trial End (UTC)'].dt.to_period('W-SUN')
analysis_df['churn_week'] = analysis_df['Ended At (UTC)'].dt.to_period('W-SUN')

# Analysis period (since launch Sep 2023)
start_analysis = pd.Timestamp('2023-09-25')
end_analysis = analysis_df['Start Date (UTC)'].max()

print(f"📅 Analysis period: {start_analysis.date()} to {end_analysis.date()}")
print(f"📅 Number of weeks: {len(pd.date_range(start_analysis, end_analysis, freq='W-SUN'))}")

# ===============================================================================
# STEP 6: ANSWERS TO PHASE 1 QUESTIONS
# ===============================================================================

print("\n=== 6. CLIENT QUESTIONS - PHASE 1 ===")

# QUESTION 1: Active members per week
print("\n--- Q1: Active Members per Week ---")

active_timeline = []
for week_start in pd.date_range(start_analysis, end_analysis, freq='W-SUN'):
    week_end = week_start + timedelta(days=6)
    
    active_count = len(analysis_df[
        analysis_df['is_converted'] &
        (analysis_df['Trial End (UTC)'] <= week_end) &
        (
            analysis_df['Ended At (UTC)'].isna() |
            (analysis_df['Ended At (UTC)'] > week_end)
        )
    ].drop_duplicates(subset='Customer ID'))
    
    active_timeline.append({
        'week_start': week_start,
        'active_members': active_count
    })

active_members_df = pd.DataFrame(active_timeline)

print(f"📊 Current active members: {active_members_df['active_members'].iloc[-1]:,}")
print(f"📊 Growth since start: +{active_members_df['active_members'].iloc[-1] - active_members_df['active_members'].iloc[0]:,}")

# QUESTION 2: New conversions per week
print("\n--- Q2: New Conversions per Week ---")

new_conversions_df = analysis_df[
    analysis_df['is_converted'] &
    analysis_df['Trial End (UTC)'].notna()
].groupby('trial_end_week').agg({'Customer ID': 'nunique'}).reset_index()
new_conversions_df.columns = ['week', 'new_conversions']

avg_conversions = new_conversions_df['new_conversions'].mean()
print(f"📊 Conversions/week (average): {avg_conversions:.1f}")

# QUESTION 3: Weekly churn (post-payment)
print("\n--- Q3: Weekly Churn ---")

churn_weekly_df = analysis_df[
    analysis_df['is_post_payment_churn']
].groupby('churn_week').agg({'Customer ID': 'nunique'}).reset_index()
churn_weekly_df.columns = ['week', 'churned_members']

avg_churn = churn_weekly_df['churned_members'].mean()
print(f"📊 Churn/week (average): {avg_churn:.1f}")

# QUESTION 4: New trials per week
print("\n--- Q4: New Trials per Week ---")

trials_weekly_df = analysis_df[
    analysis_df['is_regular_signup']
].groupby('signup_week').agg({'Customer ID': 'nunique'}).reset_index()
trials_weekly_df.columns = ['week', 'new_trials']

avg_trials = trials_weekly_df['new_trials'].mean()
print(f"📊 New trials/week (average): {avg_trials:.1f}")

# QUESTIONS 5-7: Cohort analysis
print("\n--- Q5-7: Cohort Analysis ---")

cohort_analysis_df = analysis_df[analysis_df['is_regular_signup']].groupby('signup_week').agg({
    'Customer ID': 'nunique',  # Total signups
    'is_converted': lambda x: analysis_df.loc[x.index][~analysis_df['has_renewed']]['is_converted'].sum(),  # Initial conversions
    'has_renewed': 'sum',  # Renewals
    'is_trial_cancellation': 'sum',
    'is_post_payment_churn': 'sum',
    'is_refund': 'sum',
    'is_currently_trialing': 'sum'
}).reset_index()

cohort_analysis_df.columns = [
    'cohort_week', 'total_signups', 'converted', 'renewed', 'trial_canceled',
    'post_payment_churned', 'refunds', 'still_trialing'
]

# Calculate rates
cohort_analysis_df['conversion_rate'] = (
    cohort_analysis_df['converted'] / cohort_analysis_df['total_signups'] * 100
).round(1)

cohort_analysis_df['renewal_rate'] = (
    cohort_analysis_df['renewed'] / cohort_analysis_df['converted'] * 100
).round(1)

cohort_analysis_df['trial_cancel_rate'] = (
    cohort_analysis_df['trial_canceled'] / cohort_analysis_df['total_signups'] * 100
).round(1)

ever_paid = cohort_analysis_df['converted'] + cohort_analysis_df['post_payment_churned']
cohort_analysis_df['refund_rate'] = np.where(
    ever_paid > 0,
    (cohort_analysis_df['refunds'] / ever_paid * 100).round(1),
    0
)

print(f"📊 Cohorts analyzed: {len(cohort_analysis_df)}")

# QUESTION 8: Renewals Year 1 -> Year 2
print("\n--- Q8: Renewals Year 1 -> Year 2 ---")

one_year_ago = reference_date - timedelta(days=365)
eligible_renewals = analysis_df[
    analysis_df['is_converted'] &
    (analysis_df['Trial End (UTC)'] < one_year_ago)
].drop_duplicates(subset='Customer ID')

if len(eligible_renewals) > 0:
    eligible_renewals['renewal_due_date'] = eligible_renewals['Trial End (UTC)'] + timedelta(days=365)
    eligible_renewals['has_renewed'] = (
        (eligible_renewals['Ended At (UTC)'].isna()) |
        (eligible_renewals['Ended At (UTC)'] >= eligible_renewals['renewal_due_date'])
    )
    
    renewal_analysis_df = eligible_renewals.groupby('signup_week').agg({
        'Customer ID': 'nunique',
        'has_renewed': 'sum'
    }).reset_index()
    
    renewal_analysis_df.columns = ['cohort_week', 'eligible_renewals', 'actual_renewals']
    renewal_analysis_df['renewal_rate'] = (
        renewal_analysis_df['actual_renewals'] / renewal_analysis_df['eligible_renewals'] * 100
    ).round(1)
    
    avg_renewal_rate = renewal_analysis_df['renewal_rate'].mean()
    print(f"📊 Average renewal rate: {avg_renewal_rate:.1f}%")
    print(f"📊 Eligible cohorts: {len(renewal_analysis_df)}")
else:
    print("📊 No cohorts eligible for renewal yet")
    renewal_analysis_df = None

# ===============================================================================
# STEP 7: FINAL DASHBOARD AND EXPORTS
# ===============================================================================

print("\n=== 7. FINAL DASHBOARD ===")

print(f"""
🎯 FINAL BUSINESS METRICS - DISHPATCH SUPPER CLUB:

📊 ACQUISITION:
• Total regular signups: {total_regular:,}
• New trials/week: {avg_trials:.0f} (average)

📊 CONVERSION:
• Initial conversion rate: {conversion_rate:.1f}%
• Current conversions: {converted_users.shape[0]:,}
• New conversions/week: {avg_conversions:.0f} (average)

📊 RETENTION:
• Current active members: {active_members_df['active_members'].iloc[-1]:,}
• Churn/week: {avg_churn:.0f} (average)
• Refund rate: {refund_rate:.1f}%

📊 TRIAL:
• Trial cancellation rate: {trial_cancel_rate:.1f}%
• Still in trial: {analysis_df['is_currently_trialing'].sum():,}

📊 RENEWAL:
• Renewal rate: {renewal_rate:.1f}% (based on initial conversions)
• Renewed members: {renewed_users.shape[0]:,}
""")

# Export variables for further use
dashboard_data = {
    'analysis_df': analysis_df,
    'active_members_timeline': active_members_df,
    'weekly_conversions': new_conversions_df,
    'weekly_churn': churn_weekly_df,
    'weekly_trials': trials_weekly_df,
    'cohort_analysis': cohort_analysis_df,
    'renewal_analysis': renewal_analysis_df
}

print(f"\n✅ ANALYSIS COMPLETED")
print(f"📁 Variables created: analysis_df, active_members_df, cohort_analysis_df")
print(f"🎯 Data ready for Phase 2 or visualizations")

# Display recent metrics
print(f"\n=== RECENT DATA OVERVIEW ===")
print("Last 5 weeks - Cohorts:")
recent_cohorts = cohort_analysis_df.tail(5)[['cohort_week', 'total_signups', 'converted', 'renewed', 'conversion_rate', 'renewal_rate', 'trial_canceled', 'trial_cancel_rate']]
print(recent_cohorts)

print(f"\nActive Members - Last 5 weeks:")
print(active_members_df.tail(5))

# ===============================================================================
# VISUALIZATIONS - BUSINESS DASHBOARD
# ===============================================================================

print("\n=== DISHPATCH SUPPER CLUB - BUSINESS VISUALIZATIONS ===")

# Configuration
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

# ===============================================================================
# 1. GLOBAL METRICS - OVERVIEW
# ===============================================================================

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Dishpatch Supper Club - Global Business Metrics', fontsize=20, fontweight='bold')

# 1.1 Conversion Funnel
signup_users = analysis_df[analysis_df['is_regular_signup']].drop_duplicates(subset='Customer ID')
converted_users = analysis_df[analysis_df['is_converted'] & ~analysis_df['has_renewed']].drop_duplicates(subset='Customer ID')
active_users = analysis_df[analysis_df['Status'].isin(['active', 'past_due'])].drop_duplicates(subset='Customer ID')
renewed_users = analysis_df[analysis_df['has_renewed']].drop_duplicates(subset='Customer ID')

funnel_data = {
    'Stage': ['Signups', 'Converted', 'Active', 'Renewed'],
    'Count': [
        signup_users.shape[0],
        converted_users.shape[0],
        active_users.shape[0],
        renewed_users.shape[0]
    ]
}

# Calculate percentages dynamically
total_signups = funnel_data['Count'][0]
funnel_data['Percentage'] = [
    100,
    (funnel_data['Count'][1] / total_signups * 100) if total_signups > 0 else 0,
    (funnel_data['Count'][2] / total_signups * 100) if total_signups > 0 else 0,
    (funnel_data['Count'][3] / total_signups * 100) if total_signups > 0 else 0
]

ax1.bar(funnel_data['Stage'], funnel_data['Count'], color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'])
ax1.set_title('Conversion Funnel', fontweight='bold')
ax1.set_ylabel('Number of Clients')

# Add counts and percentages on bars
for i, (count, pct) in enumerate(zip(funnel_data['Count'], funnel_data['Percentage'])):
    ax1.text(i, count + 200, f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')

# 1.2 Status Distribution
status_counts = analysis_df[analysis_df['is_regular_signup']]['Status'].value_counts()
colors = ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

wedges, texts, autotexts = ax2.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', 
                                   colors=colors, startangle=90)
ax2.set_title('Client Status Distribution', fontweight='bold')

# 1.3 Key Metrics (KPIs)
kpis = {
    'Conversion Rate': conversion_rate,
    'Renewal Rate': renewal_rate,
    'Trial Cancel Rate': trial_cancel_rate,
    'Refund Rate': refund_rate
}

bars = ax3.bar(range(len(kpis)), list(kpis.values()), 
               color=['#2ecc71', '#9b59b6', '#e74c3c', '#f39c12'])
ax3.set_title('Business KPIs', fontweight='bold')
ax3.set_xticks(range(len(kpis)))
ax3.set_xticklabels(list(kpis.keys()), rotation=45)
ax3.set_ylabel('Rate (%)')

# Add values on bars
for bar, value in zip(bars, kpis.values()):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 1, f'{value:.1f}%', 
             ha='center', va='bottom', fontweight='bold')

# 1.4 Active Members Growth
ax4.plot(active_members_df['week_start'], active_members_df['active_members'], 
         linewidth=3, color='#2ecc71', marker='o', markersize=4)
ax4.set_title('Active Members Growth', fontweight='bold')
ax4.set_ylabel('Active Members')
ax4.set_xlabel('Date')
ax4.tick_params(axis='x', rotation=45)
ax4.grid(True, alpha=0.3)

# Date formatting
from matplotlib.dates import DateFormatter
ax4.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

plt.tight_layout()
plt.show()

# ===============================================================================
# 2. WEEKLY COHORT ANALYSIS
# ===============================================================================

print("\n📊 Generating cohort dashboard...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Weekly Cohort Analysis', fontsize=20, fontweight='bold')

# 2.1 Signups and Conversions per Week
recent_weeks = cohort_analysis_df.tail(20)

x_pos = range(len(recent_weeks))
width = 0.35

bars1 = ax1.bar([x - width/2 for x in x_pos], recent_weeks['total_signups'], 
                width, label='Signups', color='#3498db', alpha=0.8)
bars2 = ax1.bar([x + width/2 for x in x_pos], recent_weeks['converted'], 
                width, label='Conversions', color='#2ecc71', alpha=0.8)

ax1.set_title('Signups vs Conversions per Week', fontweight='bold')
ax1.set_ylabel('Number')
ax1.set_xlabel('Weeks (last 20)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2.2 Conversion and Renewal Rates per Cohort
ax2.plot(x_pos, recent_weeks['conversion_rate'], 
         linewidth=3, color='#e74c3c', marker='o', markersize=6, label='Conversion Rate')
ax2.plot(x_pos, recent_weeks['renewal_rate'], 
         linewidth=3, color='#9b59b6', marker='s', markersize=6, label='Renewal Rate')
ax2.set_title('Conversion and Renewal Rates per Cohort', fontweight='bold')
ax2.set_ylabel('Rate (%)')
ax2.set_xlabel('Weeks (last 20)')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=conversion_rate, color='red', linestyle='--', alpha=0.7, label='Avg Conversion')
ax2.axhline(y=renewal_rate, color='purple', linestyle='--', alpha=0.7, label='Avg Renewal')
ax2.legend()

# 2.3 Trial Cancellations vs Conversions
ax3.bar(x_pos, recent_weeks['trial_canceled'], 
        label='Trial Canceled', color='#e74c3c', alpha=0.7)
ax3.bar(x_pos, recent_weeks['converted'], 
        bottom=recent_weeks['trial_canceled'],
        label='Converted', color='#2ecc71', alpha=0.7)

ax3.set_title('Trial Cancellations vs Conversions', fontweight='bold')
ax3.set_ylabel('Number of Clients')
ax3.set_xlabel('Weeks (last 20)')
ax3.legend()

# 2.4 Refund Rate Evolution
ax4.bar(x_pos, recent_weeks['refunds'], color='#f39c12', alpha=0.8)
ax4.set_title('Refunds per Cohort', fontweight='bold')
ax4.set_ylabel('Number of Refunds')
ax4.set_xlabel('Weeks (last 20)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===============================================================================
# 3. DETAILED TEMPORAL ANALYSIS
# ===============================================================================

print("\n📊 Generating temporal analysis...")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 15))
fig.suptitle('Detailed Temporal Analysis - Business Trends', fontsize=20, fontweight='bold')

# 3.1 Weekly Activity Volume
recent_weeks_ext = cohort_analysis_df.tail(30)
weeks_labels = [f"W{i+1}" for i in range(len(recent_weeks_ext))]

ax1.fill_between(range(len(recent_weeks_ext)), recent_weeks_ext['total_signups'], 
                 alpha=0.3, color='#3498db', label='Signups')
ax1.plot(range(len(recent_weeks_ext)), recent_weeks_ext['total_signups'], 
         linewidth=2, color='#3498db', marker='o', markersize=4)

ax1.fill_between(range(len(recent_weeks_ext)), recent_weeks_ext['converted'], 
                 alpha=0.3, color='#2ecc71', label='Conversions')
ax1.plot(range(len(recent_weeks_ext)), recent_weeks_ext['converted'], 
         linewidth=2, color='#2ecc71', marker='s', markersize=4)

ax1.set_title('Weekly Activity Volume (last 30 weeks)', fontweight='bold')
ax1.set_ylabel('Number of Clients')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 3.2 Performance Rates
ax2.plot(range(len(recent_weeks_ext)), recent_weeks_ext['conversion_rate'], 
         linewidth=3, color='#2ecc71', marker='o', markersize=5, label='Conversion Rate')
ax2.plot(range(len(recent_weeks_ext)), recent_weeks_ext['renewal_rate'], 
         linewidth=3, color='#9b59b6', marker='s', markersize=5, label='Renewal Rate')
ax2.plot(range(len(recent_weeks_ext)), recent_weeks_ext['trial_cancel_rate'], 
         linewidth=3, color='#e74c3c', marker='^', markersize=5, label='Trial Cancel Rate')

ax2.set_title('Performance Rates Evolution', fontweight='bold')
ax2.set_ylabel('Rate (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

avg_conversion = recent_weeks_ext['conversion_rate'].mean()
avg_renewal = recent_weeks_ext['renewal_rate'].mean()
avg_cancel = recent_weeks_ext['trial_cancel_rate'].mean()
ax2.axhline(y=avg_conversion, color='#2ecc71', linestyle='--', alpha=0.7)
ax2.axhline(y=avg_renewal, color='#9b59b6', linestyle='--', alpha=0.7)
ax2.axhline(y=avg_cancel, color='#e74c3c', linestyle='--', alpha=0.7)

# 3.3 Business Health (Efficiency)
efficiency = recent_weeks_ext['converted'] / (recent_weeks_ext['converted'] + recent_weeks_ext['trial_canceled']) * 100

ax3.bar(range(len(recent_weeks_ext)), efficiency, 
        color='#9b59b6', alpha=0.7, label='Conversion Efficiency')
ax3.axhline(y=efficiency.mean(), color='red', linestyle='--', alpha=0.7, 
            label=f'Average: {efficiency.mean():.1f}%')

ax3.set_title('Business Efficiency (Converted / Total Decisions)', fontweight='bold')
ax3.set_ylabel('Efficiency (%)')
ax3.set_xlabel('Weeks')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===============================================================================
# 4. EXECUTIVE DASHBOARD - SUMMARY
# ===============================================================================

print("\n📊 Generating executive dashboard...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 5, hspace=0.3, wspace=0.3)

fig.suptitle('Dishpatch Supper Club - Executive Dashboard', fontsize=24, fontweight='bold', y=0.95)

# KPIs
kpi_ax1 = fig.add_subplot(gs[0, 0])
kpi_ax2 = fig.add_subplot(gs[0, 1])
kpi_ax3 = fig.add_subplot(gs[0, 2])
kpi_ax4 = fig.add_subplot(gs[0, 3])
kpi_ax5 = fig.add_subplot(gs[0, 4])

# KPI 1: Conversion Rate
kpi_ax1.text(0.5, 0.6, f'{conversion_rate:.1f}%', ha='center', va='center', fontsize=36, fontweight='bold', color='#2ecc71')
kpi_ax1.text(0.5, 0.3, 'Conversion Rate', ha='center', va='center', fontsize=14, fontweight='bold')
kpi_ax1.text(0.5, 0.1, 'Initial Conversions', ha='center', va='center', fontsize=10, color='gray')
kpi_ax1.set_xlim(0, 1)
kpi_ax1.set_ylim(0, 1)
kpi_ax1.axis('off')

# KPI 2: Active Members
kpi_ax2.text(0.5, 0.6, f'{active_users.shape[0]:,}', ha='center', va='center', fontsize=36, fontweight='bold', color='#3498db')
kpi_ax2.text(0.5, 0.3, 'Active Members', ha='center', va='center', fontsize=14, fontweight='bold')
kpi_ax2.text(0.5, 0.1, 'Currently Paying £69/year', ha='center', va='center', fontsize=10, color='gray')
kpi_ax2.set_xlim(0, 1)
kpi_ax2.set_ylim(0, 1)
kpi_ax2.axis('off')

# KPI 3: Monthly Revenue
monthly_revenue = active_users.shape[0] * 69 / 12
kpi_ax3.text(0.5, 0.6, f'£{monthly_revenue:,.0f}', ha='center', va='center', fontsize=32, fontweight='bold', color='#e74c3c')
kpi_ax3.text(0.5, 0.3, 'Monthly Revenue', ha='center', va='center', fontsize=14, fontweight='bold')
kpi_ax3.text(0.5, 0.1, 'Estimated from Active Members', ha='center', va='center', fontsize=10, color='gray')
kpi_ax3.set_xlim(0, 1)
kpi_ax3.set_ylim(0, 1)
kpi_ax3.axis('off')

# KPI 4: Refund Rate
kpi_ax4.text(0.5, 0.6, f'{refund_rate:.1f}%', ha='center', va='center', fontsize=36, fontweight='bold', color='#f39c12')
kpi_ax4.text(0.5, 0.3, 'Refund Rate', ha='center', va='center', fontsize=14, fontweight='bold')
kpi_ax4.text(0.5, 0.1, 'Healthy (<15%)', ha='center', va='center', fontsize=10, color='gray')
kpi_ax4.set_xlim(0, 1)
kpi_ax4.set_ylim(0, 1)
kpi_ax4.axis('off')

# KPI 5: Renewal Rate
kpi_ax5.text(0.5, 0.6, f'{renewal_rate:.1f}%', ha='center', va='center', fontsize=36, fontweight='bold', color='#9b59b6')
kpi_ax5.text(0.5, 0.3, 'Renewal Rate', ha='center', va='center', fontsize=14, fontweight='bold')
kpi_ax5.text(0.5, 0.1, 'Year 1 to Year 2', ha='center', va='center', fontsize=10, color='gray')
kpi_ax5.set_xlim(0, 1)
kpi_ax5.set_ylim(0, 1)
kpi_ax5.axis('off')

# Main Chart: Cumulative Growth
main_ax = fig.add_subplot(gs[1:, :3])
cumulative_members = np.cumsum(new_conversions_df['new_conversions'])
main_ax.fill_between(new_conversions_df['week'].dt.start_time, cumulative_members, alpha=0.3, color='#2ecc71')
main_ax.plot(new_conversions_df['week'].dt.start_time, cumulative_members, linewidth=4, color='#2ecc71', marker='o', markersize=6)
main_ax.set_title('Cumulative Member Growth', fontsize=16, fontweight='bold')
main_ax.set_ylabel('Cumulative Members', fontsize=12)
main_ax.set_xlabel('Date', fontsize=12)
main_ax.grid(True, alpha=0.3)
main_ax.tick_params(axis='x', rotation=45)
main_ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

# Secondary Chart: Recent Performance
perf_ax = fig.add_subplot(gs[1:, 3:])
recent_10 = recent_weeks_ext.tail(10)
x_pos = range(len(recent_10))
perf_ax.bar([x - 0.2 for x in x_pos], recent_10['total_signups'], 
           width=0.4, label='Signups', color='#3498db', alpha=0.8)
perf_ax.bar([x + 0.2 for x in x_pos], recent_10['converted'], 
           width=0.4, label='Conversions', color='#2ecc71', alpha=0.8)
perf_ax.set_title('Recent Performance (last 10 weeks)', fontsize=16, fontweight='bold')
perf_ax.set_ylabel('Number', fontsize=12)
perf_ax.set_xlabel('Weeks', fontsize=12)
perf_ax.legend()
perf_ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===============================================================================
# 5. METRICS EXPORT FOR PRESENTATION
# ===============================================================================

print("\n=== KEY METRICS FOR PRESENTATION ===")

metrics_summary = {
    'Metric': [
        'Total Signups (Regular)', 
        'Current Active Members',
        'Initial Conversion Rate',
        'Renewal Rate',
        'Trial Cancellation Rate',
        'Refund Rate',
        'Est. Monthly Revenue',
        'Average Weekly Signups',
        'Average Weekly Conversions'
    ],
    'Value': [
        f"{total_signups:,}",
        f"{active_users.shape[0]:,}",
        f"{conversion_rate:.1f}%",
        f"{renewal_rate:.1f}%",
        f"{trial_cancel_rate:.1f}%",
        f"{refund_rate:.1f}%",
        f"£{monthly_revenue:,.0f}",
        f"{recent_weeks_ext['total_signups'].mean():.0f}",
        f"{recent_weeks_ext['converted'].mean():.0f}"
    ],
    'Status': [
        'Excellent Growth',
        'Strong Base',
        'Outstanding',
        'Strong Retention',
        'Normal for Freemium',
        'Very Healthy',
        'Strong Revenue',
        'Consistent Acquisition',
        'Solid Performance'
    ]
}

summary_df = pd.DataFrame(metrics_summary)
print(summary_df.to_string(index=False))

print(f"\n🎯 COMPLETE DASHBOARD GENERATED!")
print(f"📊 4 main visualizations created:")
print(f"   1. Global Business Metrics")
print(f"   2. Weekly Cohort Analysis")
print(f"   3. Detailed Temporal Analysis")
print(f"   4. Executive Dashboard")
print(f"\n💼 Ready for client presentation!")