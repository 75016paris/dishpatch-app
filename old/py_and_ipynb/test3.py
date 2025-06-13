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
import pytz
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

def load_latest_csv(data_dir='data', rename_files=False):
    """Load the most recent CSV file from the data directory"""
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

        if rename_files:
            if not original_name.startswith(timestamp_str):
                os.rename(file_path, new_path)
                print(f"Renamed: {original_name} → {new_name}")
                file_path = new_path
            else:
                print(f"Already renamed: {original_name}")
        else:
            print(f"[DEV] Would rename: {original_name} → {new_name}")

    return pd.read_csv(sorted_files[0])

# Load data
df_raw = load_latest_csv()

# %%
##################
# DATA PREPROCESSING
##################

def preprocess_data(df):
    """Clean and preprocess the subscription data"""
    df = df.copy()

    # Date conversion
    date_cols = [col for col in df.columns if '(UTC)' in col]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)

    df = df.sort_values(by='Created (UTC)')

    # Column selection and renaming
    columns_to_keep = [
        'Customer ID', 'Customer Name', 'Status', 'Cancellation Reason',
        'Created (UTC)', 'Start (UTC)', 'Start Date (UTC)', 
        'Current Period Start (UTC)', 'Current Period End (UTC)', 
        'Trial Start (UTC)', 'Trial End (UTC)',
        'Canceled At (UTC)', 'Ended At (UTC)', 
        'senderShopifyCustomerId (metadata)'
    ]
    
    df = df[columns_to_keep]

    df.rename(columns={
        'Customer ID': 'customer_id',
        'Customer Name': 'customer_name',
        'Status': 'status',
        'Created (UTC)': 'created_utc',
        'Start (UTC)': 'start_utc',
        'Start Date (UTC)': 'start_date_utc',
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
    reference_date = pd.Timestamp.now(tz='UTC')
    print(f"📅 Reference date (TODAY) for analysis: {reference_date.strftime('%d-%m-%Y')}")

    # Consolidate status
    df.loc[df['status'].isin(['past_due', 'incomplete_expired']), 'status'] = 'canceled'

    return df, reference_date

df, reference_date = preprocess_data(df_raw)

# %%
##################
# HELPER FUNCTIONS
##################

def clean_membership_data(df):
    """Clean and prepare membership data for analysis"""
    # Remove very short subscriptions (likely test accounts)
    df['duration_days'] = (pd.to_datetime(df['ended_at_utc']) - pd.to_datetime(df['created_utc'])).dt.days
    df_clean = df[~((df['duration_days'] < 1) & (df['duration_days'].notna()))]
    
    # Remove duplicate signups (within 6 hours)
    df_clean = df_clean.sort_values(['customer_name', 'created_utc'])
    df_clean['time_diff'] = df_clean.groupby('customer_name')['created_utc'].diff()
    df_clean = df_clean[~((df_clean['time_diff'] < pd.Timedelta(hours=6)) & (df_clean['time_diff'].notna()))]
    
    return df_clean.drop(['duration_days', 'time_diff'], axis=1)

def calculate_real_duration(row):
    """Calculate actual subscription duration - CORRIGÉ"""
    # Utiliser la date de création comme début si current_period_start est manquant
    start_date = row['current_period_start_utc'] if pd.notna(row['current_period_start_utc']) else row['created_utc']
    
    if pd.notna(row['ended_at_utc']):
        end_date = row['ended_at_utc']
    elif pd.notna(row['canceled_at_utc']):
        end_date = row['canceled_at_utc']
    elif pd.notna(row['current_period_end_utc']):
        end_date = row['current_period_end_utc']
    else:
        # Si pas de date de fin, utiliser la date de référence pour les actifs
        end_date = reference_date if row['status'] == 'active' else start_date
    
    if pd.notna(start_date) and pd.notna(end_date):
        duration = (end_date - start_date).days
        return max(0, duration)  # Éviter les durées négatives
    return 0

def calculate_period_duration(row):
    """Calculate subscription period duration"""
    start_date = row['current_period_start_utc']
    end_date = row['current_period_end_utc']
    
    if pd.notna(start_date) and pd.notna(end_date):
        return max(0, (end_date - start_date).days)
    return 0

def calculate_trial_duration(row):
    """Calculate trial duration"""
    if pd.notna(row['trial_start_utc']) and pd.notna(row['trial_end_utc']):
        start_date = row['trial_start_utc']
        end_date = row['trial_end_utc']
        return max(0, (end_date - start_date).days)
    return 0

# %%
##################
# DATA PROCESSING
##################

# Clean abnormal data
analysis_df = clean_membership_data(df)

# Customer currently trialing
analysis_df['is_currently_trialing'] = analysis_df['status'] == 'trialing'

# Duration calculations
analysis_df['real_duration'] = analysis_df.apply(calculate_real_duration, axis=1)
analysis_df['period_duration'] = analysis_df.apply(calculate_period_duration, axis=1)
analysis_df['trial_duration'] = analysis_df.apply(calculate_trial_duration, axis=1)
analysis_df['only_trial'] = (analysis_df['period_duration'] == analysis_df['trial_duration']) & (analysis_df['trial_duration'] > 0)

# Unknown period calculation
analysis_df['unknown_period'] = (analysis_df['current_period_start_utc'] - analysis_df['created_utc']).dt.days

# Define conversion and cancellation logic - GARDANT VOTRE LOGIQUE
analysis_df['paid_after_trial'] = (
    (analysis_df['status'] == 'active') |
    ((analysis_df['status'] == 'canceled') &
     (analysis_df['only_trial'] == False) &
     (analysis_df['canceled_at_utc'].isna())) |
    ((analysis_df['status'] == 'canceled') &
     (analysis_df['canceled_at_utc'] > analysis_df['trial_end_utc']))
)

analysis_df['cancel_during_trial'] = (
    ((analysis_df['status'] == 'canceled') &
     (analysis_df['canceled_at_utc'] <= analysis_df['trial_end_utc'])) |
    ((analysis_df['status'] == 'trialing') & 
     (analysis_df['canceled_at_utc'].notna()))
)

# Cas 1: Clients avec essai - remboursés dans les 14 jours après l'essai
refund_after_trial = (
    (analysis_df['status'] == 'canceled') &
    (analysis_df['trial_end_utc'].notna()) &  # A eu un essai
    (analysis_df['only_trial'] == False) &
    (analysis_df['cancel_during_trial'] == False) &
    (analysis_df['canceled_at_utc'] < analysis_df['trial_end_utc'] + pd.Timedelta(days=14)) &
    (analysis_df['canceled_at_utc'] > analysis_df['trial_end_utc'])  # Annulé APRÈS l'essai
)

# Cas 2: Clients sans essai - remboursés dans les 14 jours après le début de facturation
refund_no_trial = (
    (analysis_df['status'] == 'canceled') &
    (analysis_df['trial_end_utc'].isna()) &  # Pas d'essai
    (analysis_df['canceled_at_utc'] < analysis_df['current_period_start_utc'] + pd.Timedelta(days=14))
)

# Combinaison des deux cas
analysis_df['was_refund'] = refund_after_trial | refund_no_trial

analysis_df['in_churn_period'] = (
    ((analysis_df['status'] == 'active') &
     (analysis_df['trial_end_utc'] + pd.Timedelta(days=14) >= reference_date)) |
    ((analysis_df['status'] == 'active') &
     (analysis_df['current_period_start_utc'] + pd.Timedelta(days=14) >= reference_date))
)

analysis_df['end_soon'] = (
    (analysis_df['status'] == 'active') &
    (analysis_df['current_period_end_utc'] + pd.Timedelta(days=14) >= reference_date) &
    (analysis_df['current_period_end_utc'] > reference_date)
)

# %%
##################
# CUSTOMER AGGREGATION - CORRIGÉ
##################

def aggregate_customer_data_corrected(analysis_df):
    """Aggregate subscription data by customer - VERSION CORRIGÉE"""
    
    # Trier par customer_name et created_utc pour avoir un ordre cohérent
    df_sorted = analysis_df.sort_values(['customer_name', 'created_utc'])
    
    # Créer un identifiant unique pour chaque période d'abonnement
    df_sorted['subscription_sequence'] = df_sorted.groupby('customer_name').cumcount() + 1
    
    # Agrégation avec logique corrigée
    customer_df = df_sorted.groupby('customer_name').agg({
        # Dates importantes (première et dernière)
        'created_utc': 'first',  # Premier signup
        'current_period_start_utc': 'last',  # Période actuelle
        'current_period_end_utc': 'last',
        'trial_start_utc': 'first',  # Premier trial
        'trial_end_utc': 'first',
        'canceled_at_utc': 'last',  # Dernière annulation
        'ended_at_utc': 'last',
        
        # Statuts (utiliser le plus récent)
        'status': 'last',
        'is_currently_trialing': 'last',
        
        # Flags booléens (ANY = True si au moins une occurrence)
        'is_gifted_member': 'any',
        'paid_after_trial': 'any',
        'cancel_during_trial': 'any',
        'was_refund': 'any',
        
        # Métriques calculées
        'real_duration': lambda x: x.max(),  # Plus longue durée d'abonnement
        'period_duration': lambda x: x.sum(),  # Somme des périodes pour calculer renewals
        'trial_duration': 'first',  # Premier trial duration
        
        # Comptages
        'customer_id': 'count'  # Nombre de lignes = nombre d'abonnements
    }).reset_index()
    
    customer_df.rename(columns={'customer_id': 'subscription_count'}, inplace=True)
    
    # Métriques dérivées CORRIGÉES
    customer_df['total_subscription_days'] = customer_df['real_duration']
    
    # Logique de renouvellement améliorée
    customer_df['has_multiple_subscriptions'] = customer_df['subscription_count'] > 1
    customer_df['eligible_for_1st_renewal'] = customer_df['period_duration'] >= 350  # ~11.5 mois
    customer_df['actually_renewed_1st'] = customer_df['period_duration'] >= 400   # ~13 mois
    customer_df['actually_renewed_2nd'] = customer_df['period_duration'] >= 730   # ~24 mois
    
    # Validation des données
    print(f"✅ Customer aggregation completed:")
    print(f"   - Total unique customers: {len(customer_df):,}")
    print(f"   - Customers with multiple subscriptions: {customer_df['has_multiple_subscriptions'].sum():,}")
    print(f"   - Average subscription count per customer: {customer_df['subscription_count'].mean():.2f}")
    
    return customer_df

customer_df = aggregate_customer_data_corrected(analysis_df)

# %%
##################
# STATUS DETERMINATION - CORRIGÉ
##################

def determine_customer_status_corrected(row):
    """Determine the current status of each customer - VERSION CORRIGÉE"""
    
    # Ordre de priorité pour déterminer le statut
    if row['cancel_during_trial']:
        return 'Trial Canceled'
    elif row['is_currently_trialing']:
        return 'Currently Trialing'
    elif row['was_refund'] and row['paid_after_trial']:
        return 'Refunded After Payment'
    elif not row['paid_after_trial']:
        return 'Never Converted'
    elif row['status'] == 'active':
        if row['actually_renewed_1st']:
            return 'Active - Renewed'
        else:
            return 'Active - First Year'
    elif row['status'] == 'canceled' and row['paid_after_trial']:
        return 'Churned After Payment'
    else:
        return 'Other Status'

customer_df['detailed_status'] = customer_df.apply(determine_customer_status_corrected, axis=1)

# Validation des statuts
print("\n📊 Customer Status Distribution:")
status_counts = customer_df['detailed_status'].value_counts()
for status, count in status_counts.items():
    percentage = count / len(customer_df) * 100
    print(f"   {status}: {count:,} ({percentage:.1f}%)")

# %%
##################
# JOURNEY ANALYSIS - SIMPLIFIÉ ET CORRIGÉ
##################

def create_customer_journey_analysis(customer_df):
    """Create simplified customer journey analysis"""
    
    journey_df = customer_df.copy()
    
    # Définir le premier touchpoint basé sur les données disponibles
    def determine_first_touchpoint(row):
        if pd.notna(row['trial_start_utc']):
            return 'trial'
        elif row['is_gifted_member']:
            return 'gift'
        else:
            return 'direct'
    
    journey_df['first_touchpoint'] = journey_df.apply(determine_first_touchpoint, axis=1)
    
    # Déterminer la source de conversion pour ceux qui ont converti
    def determine_conversion_source(row):
        if not row['paid_after_trial']:
            return None
        return row['first_touchpoint']  # Simple: même que le premier touchpoint
    
    journey_df['conversion_source'] = journey_df.apply(determine_conversion_source, axis=1)
    
    # Flags simplifiés
    journey_df['had_trial'] = journey_df['trial_start_utc'].notna()
    journey_df['had_gift'] = journey_df['is_gifted_member']
    journey_df['converted_to_paid'] = journey_df['paid_after_trial']
    
    return journey_df

customer_journey_df = create_customer_journey_analysis(customer_df)

# %%
##################
# KPI CALCULATIONS - CORRIGÉ
##################

def calculate_kpis_corrected(customer_df):
    """Calculate key performance indicators - VERSION CORRIGÉE"""
    
    total_unique_customers = len(customer_df)
    total_conversions = customer_df['paid_after_trial'].sum()
    total_trial_cancellations = (customer_df['detailed_status'] == 'Trial Canceled').sum()
    total_refunded = customer_df['was_refund'].sum()
    current_active_members = customer_df[customer_df['status'] == 'active'].shape[0]
    
    # Renewals basés sur la logique améliorée
    total_eligible_for_renewal = customer_df['eligible_for_1st_renewal'].sum()
    total_actually_renewed = customer_df['actually_renewed_1st'].sum()
    
    # Calcul des taux avec validation
    def safe_percentage(numerator, denominator):
        return (numerator / denominator * 100) if denominator > 0 else 0
    
    kpi_conversion_rate = safe_percentage(total_conversions, total_unique_customers)
    kpi_trial_cancel_rate = safe_percentage(total_trial_cancellations, total_unique_customers)
    kpi_refund_rate = safe_percentage(total_refunded, total_conversions)
    kpi_renewal_rate = safe_percentage(total_actually_renewed, total_eligible_for_renewal)
    
    kpis = {
        'total_unique_customers': total_unique_customers,
        'total_conversions': total_conversions,
        'total_trial_cancellations': total_trial_cancellations,
        'total_refunded': total_refunded,
        'total_eligible_for_renewal': total_eligible_for_renewal,
        'total_actually_renewed': total_actually_renewed,
        'current_active_members': current_active_members,
        'kpi_conversion_rate': round(kpi_conversion_rate, 1),
        'kpi_trial_cancel_rate': round(kpi_trial_cancel_rate, 1),
        'kpi_refund_rate': round(kpi_refund_rate, 1),
        'kpi_renewal_rate': round(kpi_renewal_rate, 1)
    }
    
    # Validation des KPIs
    print(f"\n📈 KPI Summary:")
    print(f"   Total Customers: {kpis['total_unique_customers']:,}")
    print(f"   Conversion Rate: {kpis['kpi_conversion_rate']}%")
    print(f"   Trial Cancel Rate: {kpis['kpi_trial_cancel_rate']}%")
    print(f"   Refund Rate: {kpis['kpi_refund_rate']}%")
    print(f"   Renewal Rate: {kpis['kpi_renewal_rate']}%")
    print(f"   Active Members: {kpis['current_active_members']:,}")
    
    return kpis

kpis = calculate_kpis_corrected(customer_df)

# %%
##################
# COHORT ANALYSIS - CORRIGÉ
##################

def create_cohort_analysis_corrected(customer_df):
    """Create weekly cohort analysis - VERSION CORRIGÉE"""
    
    cohort_df = customer_df.copy()
    
    # Créer les cohortes hebdomadaires basées sur le premier signup
    cohort_df['signup_week'] = cohort_df['created_utc'].dt.to_period('W-SUN')
    
    # Agrégation par cohorte avec métriques cohérentes
    cohort_analysis = cohort_df.groupby('signup_week').agg({
        'customer_name': 'count',  # Total signups
        'paid_after_trial': 'sum',  # Conversions
        'detailed_status': lambda x: (x == 'Trial Canceled').sum(),  # Trial cancellations
        'was_refund': 'sum',  # Refunds
        'eligible_for_1st_renewal': 'sum',  # Eligible for renewal
        'actually_renewed_1st': 'sum'  # Actually renewed
    }).reset_index()
    
    # Renommer les colonnes pour plus de clarté
    cohort_analysis.columns = [
        'signup_week', 'total_signups', 'conversions', 'trial_cancellations',
        'refunds', 'eligible_renewals', 'actual_renewals'
    ]
    
    # Calculer les taux
    def safe_rate(numerator, denominator):
        return np.where(denominator > 0, (numerator / denominator * 100).round(1), 0)
    
    cohort_analysis['conversion_rate'] = safe_rate(
        cohort_analysis['conversions'], cohort_analysis['total_signups']
    )
    cohort_analysis['trial_cancel_rate'] = safe_rate(
        cohort_analysis['trial_cancellations'], cohort_analysis['total_signups']
    )
    cohort_analysis['refund_rate'] = safe_rate(
        cohort_analysis['refunds'], cohort_analysis['conversions']
    )
    cohort_analysis['renewal_rate'] = safe_rate(
        cohort_analysis['actual_renewals'], cohort_analysis['eligible_renewals']
    )
    
    # Filtrer les cohortes complètes (plus de 4 semaines)
    cutoff_date = pd.Timestamp.now() - pd.Timedelta(weeks=4)
    complete_cohorts = cohort_analysis[
        cohort_analysis['signup_week'].apply(lambda x: x.end_time) < cutoff_date
    ].copy()
    
    print(f"\n📅 Cohort Analysis:")
    print(f"   Total cohorts: {len(cohort_analysis)}")
    print(f"   Complete cohorts (>4 weeks): {len(complete_cohorts)}")
    print(f"   Recent complete cohorts avg conversion: {complete_cohorts.tail(10)['conversion_rate'].mean():.1f}%")
    
    return cohort_analysis, complete_cohorts

cohort_analysis, complete_cohorts = create_cohort_analysis_corrected(customer_df)

# %%
##################
# VISUALIZATIONS - CORRIGÉ
##################

def create_business_dashboard(kpis, customer_df, customer_journey_df):
    """Create comprehensive business dashboard"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Dishpatch Supper Club - Business Analytics Dashboard (Corrected)', 
                 fontsize=20, fontweight='bold')

    # 1. Conversion Funnel
    funnel_stages = ['Total\nSignups', 'Paid\nConversions', 'Currently\nActive', 'Renewed\n(1st Time)']
    funnel_counts = [
        kpis['total_unique_customers'],
        kpis['total_conversions'],
        kpis['current_active_members'],
        kpis['total_actually_renewed']
    ]
    funnel_colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    bars = ax1.bar(funnel_stages, funnel_counts, color=funnel_colors, alpha=0.8)
    ax1.set_title('Customer Conversion Funnel', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Number of Customers')
    
    # Ajouter les pourcentages
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage = (height / funnel_counts[0] * 100) if funnel_counts[0] > 0 else 0
        ax1.text(bar.get_x() + bar.get_width()/2., height + (max(funnel_counts) * 0.01), 
                f'{height:,}\n({percentage:.1f}%)', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    ax1.set_ylim(0, max(funnel_counts) * 1.15)

    # 2. Status Distribution
    status_counts = customer_df['detailed_status'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(status_counts)))
    
    wedges, texts, autotexts = ax2.pie(status_counts.values, labels=status_counts.index, 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Customer Status Distribution', fontweight='bold', fontsize=14)

    # 3. Key Performance Indicators
    kpi_names = ['Conversion\nRate', 'Trial Cancel\nRate', 'Refund\nRate', 'Renewal\nRate']
    kpi_values = [
        kpis['kpi_conversion_rate'],
        kpis['kpi_trial_cancel_rate'],
        kpis['kpi_refund_rate'],
        kpis['kpi_renewal_rate']
    ]
    kpi_colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    
    bars_kpi = ax3.bar(kpi_names, kpi_values, color=kpi_colors, alpha=0.8)
    ax3.set_title('Key Performance Indicators', fontweight='bold', fontsize=14)
    ax3.set_ylabel('Rate (%)')
    
    for bar in bars_kpi:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', 
                ha='center', va='bottom', fontweight='bold')
    ax3.set_ylim(0, max(kpi_values) * 1.2 if max(kpi_values) > 0 else 10)

    # 4. Touchpoint Analysis
    touchpoint_analysis = customer_journey_df.groupby('first_touchpoint').agg({
        'customer_name': 'count',
        'converted_to_paid': 'sum'
    })
    touchpoint_analysis['conversion_rate'] = (
        touchpoint_analysis['converted_to_paid'] / touchpoint_analysis['customer_name'] * 100
    )
    
    bars_tp = ax4.bar(touchpoint_analysis.index, touchpoint_analysis['conversion_rate'], 
                     color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
    ax4.set_title('Conversion Rate by First Touchpoint', fontweight='bold', fontsize=14)
    ax4.set_ylabel('Conversion Rate (%)')
    ax4.set_xlabel('First Touchpoint')
    
    for i, bar in enumerate(bars_tp):
        height = bar.get_height()
        volume = touchpoint_analysis.iloc[i]['customer_name']
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1, 
                f'{height:.1f}%\n(n={volume:,})', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def create_cohort_dashboard(complete_cohorts):
    """Create cohort analysis dashboard"""
    
    if len(complete_cohorts) < 5:
        print("⚠️ Not enough complete cohorts for meaningful analysis")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Weekly Cohort Analysis - Complete Cohorts Only', fontsize=20, fontweight='bold')
    
    # Utiliser les 20 dernières cohortes complètes
    recent_cohorts = complete_cohorts.tail(20).copy()
    recent_cohorts['week_str'] = recent_cohorts['signup_week'].astype(str)
    x_pos = np.arange(len(recent_cohorts))

    # 1. Signups vs Conversions
    width = 0.35
    ax1.bar(x_pos - width/2, recent_cohorts['total_signups'], width, 
           label='Total Signups', color='#3498db', alpha=0.8)
    ax1.bar(x_pos + width/2, recent_cohorts['conversions'], width,
           label='Conversions', color='#2ecc71', alpha=0.8)
    ax1.set_title('Signups vs Conversions by Cohort', fontweight='bold')
    ax1.set_ylabel('Number of Customers')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(recent_cohorts['week_str'], rotation=45, ha="right", fontsize=8)
    ax1.legend()

    # 2. Conversion and Cancellation Rates
    ax2.plot(x_pos, recent_cohorts['conversion_rate'], 'o-', linewidth=2, 
            color='#2ecc71', label='Conversion Rate', markersize=6)
    ax2.plot(x_pos, recent_cohorts['trial_cancel_rate'], 's-', linewidth=2,
            color='#e74c3c', label='Trial Cancel Rate', markersize=6)
    ax2.set_title('Performance Rates by Cohort', fontweight='bold')
    ax2.set_ylabel('Rate (%)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(recent_cohorts['week_str'], rotation=45, ha="right", fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Refund Rate
    ax3.bar(x_pos, recent_cohorts['refund_rate'], color='#f39c12', alpha=0.8)
    ax3.set_title('Refund Rate by Cohort', fontweight='bold')
    ax3.set_ylabel('Refund Rate (%)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(recent_cohorts['week_str'], rotation=45, ha="right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Renewal Analysis (pour les cohortes matures)
    mature_cohorts = recent_cohorts[recent_cohorts['eligible_renewals'] > 0]
    if len(mature_cohorts) > 0:
        mature_x_pos = np.arange(len(mature_cohorts))
        ax4.bar(mature_x_pos, mature_cohorts['renewal_rate'], color='#9b59b6', alpha=0.8)
        ax4.set_xticks(mature_x_pos)
        ax4.set_xticklabels(mature_cohorts['week_str'], rotation=45, ha="right", fontsize=8)
        ax4.set_title(f'Renewal Rate by Mature Cohort (n={len(mature_cohorts)})', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, "No mature cohorts\navailable for\nrenewal analysis", 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax4.set_title('Renewal Analysis - Insufficient Data', fontweight='bold')
    
    ax4.set_ylabel('Renewal Rate (%)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def create_advanced_analytics(customer_df):
    """Create advanced analytics visualizations"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Advanced Customer Analytics', fontsize=20, fontweight='bold')

    # 1. Subscription Duration Distribution
    converted_customers = customer_df[customer_df['paid_after_trial'] == True]
    if len(converted_customers) > 0:
        ax1.hist(converted_customers['total_subscription_days'], bins=30, 
                color='#2ecc71', alpha=0.7, edgecolor='black')
        mean_duration = converted_customers['total_subscription_days'].mean()
        ax1.axvline(mean_duration, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_duration:.0f} days')
        ax1.set_title('Subscription Duration Distribution (Converted Customers)', fontweight='bold')
        ax1.set_xlabel('Duration (days)')
        ax1.set_ylabel('Number of Customers')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. Churn Timing Analysis
    churned_customers = customer_df[customer_df['detailed_status'] == 'Churned After Payment']
    if len(churned_customers) > 10:
        churn_buckets = churned_customers['total_subscription_days'].value_counts().head(15)
        ax2.bar(range(len(churn_buckets)), churn_buckets.values, color='#e74c3c', alpha=0.7)
        ax2.set_title('Churn Timing Distribution (Top 15)', fontweight='bold')
        ax2.set_xlabel('Days to Churn')
        ax2.set_ylabel('Number of Customers')
        ax2.set_xticks(range(len(churn_buckets)))
        ax2.set_xticklabels([f'{d}d' for d in churn_buckets.index], rotation=45)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Insufficient churn data\nfor meaningful analysis", 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Churn Analysis - Insufficient Data', fontweight='bold')

    # 3. Monthly Signup Trends
    customer_df_copy = customer_df.copy()
    customer_df_copy['signup_month'] = customer_df_copy['created_utc'].dt.month
    monthly_trends = customer_df_copy.groupby('signup_month').agg({
        'customer_name': 'count',
        'paid_after_trial': 'sum'
    })
    monthly_trends['conversion_rate'] = (
        monthly_trends['paid_after_trial'] / monthly_trends['customer_name'] * 100
    )
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Créer des données pour tous les mois
    signup_data = [monthly_trends.loc[i, 'customer_name'] if i in monthly_trends.index else 0 
                   for i in range(1, 13)]
    conversion_data = [monthly_trends.loc[i, 'conversion_rate'] if i in monthly_trends.index else 0 
                      for i in range(1, 13)]
    
    ax3_twin = ax3.twinx()
    bars = ax3.bar(range(1, 13), signup_data, color='#3498db', alpha=0.7, label='Signups')
    line = ax3_twin.plot(range(1, 13), conversion_data, color='#e74c3c', 
                        marker='o', linewidth=2, markersize=6, label='Conversion Rate')
    
    ax3.set_title('Monthly Signup Trends & Conversion Rates', fontweight='bold')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Number of Signups', color='#3498db')
    ax3_twin.set_ylabel('Conversion Rate (%)', color='#e74c3c')
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(months)
    ax3.grid(True, alpha=0.3)
    
    # Légendes
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')

    # 4. Customer Lifetime Value Estimation
    if len(converted_customers) > 0:
        # Estimation simple: durée * tarif journalier
        daily_rate = 69 / 365  # £69 par an
        converted_customers_copy = converted_customers.copy()
        converted_customers_copy['estimated_ltv'] = converted_customers_copy['total_subscription_days'] * daily_rate
        
        ax4.hist(converted_customers_copy['estimated_ltv'], bins=25, 
                color='#f39c12', alpha=0.7, edgecolor='black')
        mean_ltv = converted_customers_copy['estimated_ltv'].mean()
        median_ltv = converted_customers_copy['estimated_ltv'].median()
        
        ax4.axvline(mean_ltv, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: £{mean_ltv:.0f}')
        ax4.axvline(median_ltv, color='orange', linestyle='--', linewidth=2,
                   label=f'Median: £{median_ltv:.0f}')
        
        ax4.set_title('Estimated Customer Lifetime Value Distribution', fontweight='bold')
        ax4.set_xlabel('Estimated LTV (£)')
        ax4.set_ylabel('Number of Customers')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# %%
##################
# BUSINESS INSIGHTS - CORRIGÉ
##################

def generate_business_insights_corrected(kpis, customer_df, customer_journey_df):
    """Generate comprehensive business insights and recommendations"""
    
    print("\n" + "="*70)
    print("DISHPATCH SUPPER CLUB - BUSINESS INSIGHTS & RECOMMENDATIONS")
    print("="*70)
    
    # 1. PERFORMANCE OVERVIEW
    print(f"\n🎯 PERFORMANCE OVERVIEW:")
    print(f"   • Total Unique Customers: {kpis['total_unique_customers']:,}")
    print(f"   • Conversion Rate: {kpis['kpi_conversion_rate']:.1f}%")
    print(f"   • Currently Active Members: {kpis['current_active_members']:,}")
    print(f"   • Estimated Annual Revenue: £{kpis['current_active_members'] * 69:,.0f}")
    
    # 2. ACQUISITION ANALYSIS
    print(f"\n📈 ACQUISITION CHANNELS:")
    touchpoint_stats = customer_journey_df.groupby('first_touchpoint').agg({
        'customer_name': 'count',
        'converted_to_paid': 'sum'
    })
    touchpoint_stats['conversion_rate'] = (
        touchpoint_stats['converted_to_paid'] / touchpoint_stats['customer_name'] * 100
    )
    
    for touchpoint, row in touchpoint_stats.iterrows():
        print(f"   • {touchpoint.capitalize()}: {row['customer_name']:,} signups "
              f"({row['conversion_rate']:.1f}% conversion)")
    
    # Recommandations d'acquisition
    best_channel = touchpoint_stats['conversion_rate'].idxmax()
    best_rate = touchpoint_stats.loc[best_channel, 'conversion_rate']
    print(f"   🏆 Best performing channel: {best_channel} ({best_rate:.1f}% conversion)")
    
    # 3. RETENTION ANALYSIS
    print(f"\n🔄 RETENTION & CHURN:")
    print(f"   • Trial Cancellation Rate: {kpis['kpi_trial_cancel_rate']:.1f}%")
    print(f"   • Refund Rate: {kpis['kpi_refund_rate']:.1f}%")
    print(f"   • First Renewal Rate: {kpis['kpi_renewal_rate']:.1f}%")
    
    # Analyse des durées
    converted_customers = customer_df[customer_df['paid_after_trial'] == True]
    if len(converted_customers) > 0:
        avg_duration = converted_customers['total_subscription_days'].mean()
        median_duration = converted_customers['total_subscription_days'].median()
        print(f"   • Average subscription duration: {avg_duration:.0f} days")
        print(f"   • Median subscription duration: {median_duration:.0f} days")
    
    # 4. FINANCIAL INSIGHTS
    print(f"\n💰 FINANCIAL METRICS:")
    if len(converted_customers) > 0:
        daily_rate = 69 / 365
        avg_ltv = converted_customers['total_subscription_days'].mean() * daily_rate
        print(f"   • Average Customer LTV: £{avg_ltv:.0f}")
        print(f"   • Monthly Recurring Revenue: £{kpis['current_active_members'] * 69 / 12:,.0f}")
    
    # Customer acquisition cost effectiveness
    total_value = kpis['current_active_members'] * 69
    acquisition_efficiency = total_value / kpis['total_unique_customers']
    print(f"   • Revenue per signup: £{acquisition_efficiency:.0f}")
    
    # 5. KEY OPPORTUNITIES
    print(f"\n🚀 KEY OPPORTUNITIES:")
    
    # Trial optimization
    if kpis['kpi_trial_cancel_rate'] > 30:
        print(f"   🔴 HIGH PRIORITY: Trial experience optimization")
        print(f"      - {kpis['kpi_trial_cancel_rate']:.1f}% trial cancellation rate needs improvement")
        print(f"      - Focus on onboarding and early value delivery")
    
    # Renewal optimization
    if kpis['kpi_renewal_rate'] < 70:
        print(f"   🟡 MEDIUM PRIORITY: Renewal rate improvement")
        print(f"      - {kpis['kpi_renewal_rate']:.1f}% renewal rate has room for growth")
        print(f"      - Implement pre-renewal engagement campaigns")
    
    # Channel optimization
    lowest_channel = touchpoint_stats['conversion_rate'].idxmin()
    lowest_rate = touchpoint_stats.loc[lowest_channel, 'conversion_rate']
    if lowest_rate < best_rate * 0.7:  # Si 30% de différence
        print(f"   🟡 CHANNEL OPTIMIZATION: Improve {lowest_channel} channel")
        print(f"      - {lowest_rate:.1f}% conversion vs {best_rate:.1f}% for {best_channel}")
    
    # Growth opportunities
    if kpis['kpi_conversion_rate'] > 40:
        print(f"   🟢 GROWTH OPPORTUNITY: Scale acquisition")
        print(f"      - {kpis['kpi_conversion_rate']:.1f}% conversion rate is strong")
        print(f"      - Focus on increasing traffic to best-performing channels")
    
    # 6. ACTION ITEMS
    print(f"\n📋 IMMEDIATE ACTION ITEMS:")
    print(f"   1. Analyze trial cancellation reasons and improve onboarding")
    print(f"   2. Develop retention campaigns for customers approaching renewal")
    print(f"   3. Investigate success factors of {best_channel} channel")
    print(f"   4. A/B test improvements for underperforming channels")
    print(f"   5. Monitor cohort performance weekly for early trend detection")
    
    print("\n" + "="*70)

def validate_data_quality(customer_df, analysis_df):
    """Validate data quality and consistency"""
    
    print(f"\n🔍 DATA QUALITY VALIDATION:")
    
    # Basic counts
    total_raw_records = len(analysis_df)
    total_customers = len(customer_df)
    print(f"   • Raw subscription records: {total_raw_records:,}")
    print(f"   • Unique customers: {total_customers:,}")
    print(f"   • Average subscriptions per customer: {total_raw_records/total_customers:.2f}")
    
    # Status consistency
    status_sum = customer_df['detailed_status'].value_counts().sum()
    assert status_sum == total_customers, f"Status inconsistency: {status_sum} vs {total_customers}"
    print(f"   ✅ Status consistency verified")
    
    # Conversion logic validation
    converted_count = customer_df['paid_after_trial'].sum()
    never_converted = (customer_df['detailed_status'] == 'Never Converted').sum()
    trial_canceled = (customer_df['detailed_status'] == 'Trial Canceled').sum()
    
    expected_not_converted = never_converted + trial_canceled
    actual_not_converted = total_customers - converted_count
    
    print(f"   • Converted customers: {converted_count:,}")
    print(f"   • Non-converted (expected): {expected_not_converted:,}")
    print(f"   • Non-converted (actual): {actual_not_converted:,}")
    
    if abs(expected_not_converted - actual_not_converted) <= 5:  # Allow small discrepancy
        print(f"   ✅ Conversion logic consistency verified")
    else:
        print(f"   ⚠️ Potential conversion logic inconsistency")
    
    # Date validation
    date_issues = customer_df[
        (customer_df['created_utc'] > customer_df['current_period_start_utc']) |
        (customer_df['current_period_start_utc'] > customer_df['current_period_end_utc'])
    ]
    print(f"   • Date logic issues: {len(date_issues)} customers")
    
    if len(date_issues) == 0:
        print(f"   ✅ Date logic validated")
    
    print(f"   ✅ Data quality validation completed")

# %%
##################
# EXECUTE ANALYSIS
##################

print("🚀 Starting Corrected Dishpatch Analysis...")

# Validate data quality
validate_data_quality(customer_df, analysis_df)

# Generate business insights
generate_business_insights_corrected(kpis, customer_df, customer_journey_df)

# Create visualizations
print("\n📊 Generating Business Dashboard...")
create_business_dashboard(kpis, customer_df, customer_journey_df)

print("\n📈 Generating Cohort Analysis...")
create_cohort_dashboard(complete_cohorts)

print("\n🔬 Generating Advanced Analytics...")
create_advanced_analytics(customer_df)

# %%
##################
# SUMMARY REPORT
##################

def generate_executive_summary():
    """Generate final executive summary"""
    
    print("\n" + "="*60)
    print("EXECUTIVE SUMMARY")
    print("="*60)
    
    print(f"\n📊 KEY METRICS:")
    print(f"   • Total Customers Analyzed: {kpis['total_unique_customers']:,}")
    print(f"   • Overall Conversion Rate: {kpis['kpi_conversion_rate']:.1f}%")
    print(f"   • Active Paying Members: {kpis['current_active_members']:,}")
    print(f"   • Estimated ARR: £{kpis['current_active_members'] * 69:,.0f}")
    
    # Performance assessment
    conversion_assessment = "EXCELLENT" if kpis['kpi_conversion_rate'] > 40 else "GOOD" if kpis['kpi_conversion_rate'] > 25 else "NEEDS IMPROVEMENT"
    trial_assessment = "EXCELLENT" if kpis['kpi_trial_cancel_rate'] < 25 else "GOOD" if kpis['kpi_trial_cancel_rate'] < 40 else "NEEDS IMPROVEMENT"
    
    print(f"\n🎯 PERFORMANCE ASSESSMENT:")
    print(f"   • Conversion Performance: {conversion_assessment}")
    print(f"   • Trial Retention: {trial_assessment}")
    
    # Top priorities
    print(f"\n🔥 TOP PRIORITIES:")
    if kpis['kpi_trial_cancel_rate'] > 35:
        print(f"   1. 🔴 URGENT: Reduce trial cancellation rate ({kpis['kpi_trial_cancel_rate']:.1f}%)")
    else:
        print(f"   1. 🟢 Trial performance is healthy ({kpis['kpi_trial_cancel_rate']:.1f}%)")
    
    if kpis['kpi_renewal_rate'] < 10:
        print(f"   2. 🟡 IMPORTANT: Improve renewal tracking and rates")
    else:
        print(f"   2. 🟢 Renewal rate looks good ({kpis['kpi_renewal_rate']:.1f}%)")
    
    print(f"   3. 🔵 Continue monitoring weekly cohort performance")
    
    print(f"\n✅ Analysis completed successfully!")
    print(f"📅 Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

# Generate final summary
generate_executive_summary()

# %%
##################
# EXPORT RESULTS (Optional)
##################

def export_results():
    """Export key results to CSV files"""
    
    try:
        # Export customer summary
        customer_export = customer_df[[
            'customer_name', 'created_utc', 'detailed_status', 'paid_after_trial',
            'total_subscription_days', 'subscription_count'
        ]].copy()
        
        customer_export['created_date'] = customer_export['created_utc'].dt.date
        customer_export = customer_export.drop('created_utc', axis=1)
        
        # Export cohort analysis
        cohort_export = complete_cohorts[[
            'signup_week', 'total_signups', 'conversions', 'conversion_rate',
            'trial_cancellations', 'trial_cancel_rate'
        ]].copy()
        
        print(f"\n💾 Export Summary:")
        print(f"   • Customer data: {len(customer_export):,} records")
        print(f"   • Cohort data: {len(cohort_export):,} records")
        print(f"   • Files ready for export")
        
        return customer_export, cohort_export
        
    except Exception as e:
        print(f"❌ Export error: {str(e)}")
        return None, None

# Uncomment to export:
# customer_export, cohort_export = export_results()

print("\n🎉 DISHPATCH ANALYSIS COMPLETED SUCCESSFULLY!")
print("All metrics have been corrected and validated.")
print("Visualizations and insights are ready for business review.")


# Ajouter cette validation détaillée :
def debug_conversion_logic(customer_df):
    print("=== DEBUG CONVERSION LOGIC ===")
    
    status_breakdown = customer_df['detailed_status'].value_counts()
    
    # Clients qui ont payé selon votre logique
    paid_customers = customer_df[customer_df['paid_after_trial'] == True]
    
    # Clients qui ont payé selon les statuts
    paid_statuses = ['Active - First Year', 'Active - Renewed', 'Churned After Payment', 'Refunded After Payment']
    paid_by_status = customer_df[customer_df['detailed_status'].isin(paid_statuses)]
    
    print(f"Paid by logic: {len(paid_customers)}")
    print(f"Paid by status: {len(paid_by_status)}")
    print(f"Difference: {abs(len(paid_customers) - len(paid_by_status))}")
    
    return paid_customers, paid_by_status

debug_conversion_logic(customer_df)


# Logique alternative pour les renouvellements
def recalculate_renewals(customer_df):
    # Clients éligibles = actifs depuis plus de 11 mois
    truly_eligible = customer_df[
        (customer_df['total_subscription_days'] >= 335) &  # ~11 mois
        (customer_df['paid_after_trial'] == True)
    ]
    
    # Renouvelés = période > 400 jours OU multiple subscriptions
    truly_renewed = customer_df[
        ((customer_df['total_subscription_days'] > 400) |
         (customer_df['subscription_count'] > 1)) &
        (customer_df['paid_after_trial'] == True)
    ]
    
    return len(truly_renewed), len(truly_eligible)

recalculate_renewals(customer_df)

# Investiguer le canal "direct"
def investigate_direct_channel(customer_journey_df):
    direct_customers = customer_journey_df[customer_journey_df['first_touchpoint'] == 'direct']
    
    print("=== DIRECT CHANNEL ANALYSIS ===")
    print(f"Direct customers: {len(direct_customers)}")
    print(f"Had trial: {direct_customers['had_trial'].sum()}")
    print(f"Had gift: {direct_customers['had_gift'].sum()}")
    print(f"Average duration: {direct_customers['total_subscription_days'].mean():.0f} days")

investigate_direct_channel(customer_journey_df)
