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

    # # Set Canceled At (UTC) to Current Period End (UTC) for past_due without cancellation date
    # past_due_no_cancel_date = (
    #     df['canceled_at_utc'].isna() & 
    #     (df['status'] == 'past_due'))
    #
    # df.loc[past_due_no_cancel_date, 'canceled_at_utc'] = df.loc[past_due_no_cancel_date, 
    #                                                             'current_period_end_utc'].fillna(
    # df.loc[past_due_no_cancel_date, 'trial_end_utc']).fillna(df.loc[past_due_no_cancel_date, 
    #                                                                     'created_utc'])
    #
    # # Consolidate status
    # df.loc[df['status'].isin(['past_due', 'incomplete_expired']), 'status'] = 'canceled'
    #
    return df

df = preprocess_data(df_raw)

# %%

# Removing customers with more than 7 subscriptions (Probably testing accounts)
def remove_high_volume_customers(df, threshold=7):
    """Remove customers with more than a specified number of subscriptions"""
    
    original_count = len(df)

    customer_counts = df['customer_name'].value_counts()
    high_volume_customers = customer_counts[customer_counts > threshold].index
    
    print(f"Removing {len(high_volume_customers)} customers with more than {threshold} subscriptions")
    
    df = df[~df['customer_name'].isin(high_volume_customers)]
    
    print(f'{original_count - len(df)} subscriptions removed from {len(high_volume_customers)} customers')

    return df

df = remove_high_volume_customers(df)

# %%

# Adding end date for subscriptions that are canceled but have no ended_at_utc date.
def add_ended_at_for_canceled(df):
    """Set ended_at_utc to current_period_end_utc for canceled subscriptions without ended_at_utc"""
    
    # Ensure ended_at_utc is set to current_period_end_utc for canceled subscriptions
    mask = (df['ended_at_utc'].isna()) & (df['canceled_at_utc'].notna())
    
    df.loc[mask, 'ended_at_utc'] = df['current_period_end_utc']
    
    return df


df = add_ended_at_for_canceled(df)

# %%

def calculate_duration(df):
    """Calculate various duration in days"""
    
    df['trial_duration'] = \
            (df['trial_end_utc'] - df['trial_start_utc']).dt.days.fillna(0)
    
    df['current_period_duration'] = \
            (df['current_period_end_utc'] - df['current_period_start_utc']).dt.days
    
    df['trial_only_subscription'] = (
        df['trial_start_utc'].notna() & 
        df['trial_end_utc'].notna() & 
        (df['trial_duration'] == df['current_period_duration'])
    )

    df['gift_duration'] = df['current_period_duration'].where(df['is_gifted_member'].notna(), 0)

    df['end_in'] = \
            ((df['current_period_end_utc'] - reference_date).dt.days).where(df['status'] == 'active', np.nan)

    df['total_duration_days'] = (df['ended_at_utc'] - df['created_utc']).dt.days

    return df


df = calculate_duration(df)

# %%

# Calucate real duration of the subscription (filter out trial only subscriptions)
def real_duration(df):
    """Calculate the real duration of the subscription"""
    print('------------------------------------------------------')

    print(f"Number of row in df before filtering {len(df)}") 
    filtered_df = df[
            (df['trial_only_subscription'] == False) &
            (df['total_duration_days'] > 14)
            ].copy()
        
    print(f"Number of row in df after filtering {len(filtered_df)}")
    return filtered_df

real_duration(df)


# %%

def calculate_duration_advanced(df):
    """Calculate subscription duration with better trial detection"""
    
    # Durées de base
    df['trial_duration_days'] = (df['trial_end_utc'] - df['trial_start_utc']).dt.days
    df['current_period_duration_days'] = (df['current_period_end_utc'] - df['current_period_start_utc']).dt.days
    df['total_duration_days'] = (df['ended_at_utc'] - df['created_utc']).dt.days
    
    # Détection améliorée des trial-only subscriptions
    df['trial_only_subscription'] = (
        df['trial_start_utc'].notna() & 
        df['trial_end_utc'].notna() & 
        (df['trial_duration_days'] == df['current_period_duration_days']) &
        (df['trial_duration_days'].between(10, 40))  # Période trial typique
    )
    
    # Calcul de la durée réelle (post-trial)
    df['real_duration_days'] = df['total_duration_days'].copy()
    
    # Pour ceux qui ont eu un trial, soustraire la durée du trial
    has_trial = df['trial_start_utc'].notna() & df['trial_end_utc'].notna()
    df.loc[has_trial, 'real_duration_days'] = (
        df.loc[has_trial, 'total_duration_days'] - df.loc[has_trial, 'trial_duration_days']
    )
    
    # Durée depuis la fin du trial (pour les abonnements actifs)
    df['days_since_trial_end'] = (reference_date - df['trial_end_utc']).dt.days
    
    return df

def filter_real_subscriptions(df, min_duration_days=14, include_refund_period=True):
    """Filter to get real subscriptions excluding trials and refund period"""
    
    print('=' * 60)
    print('FILTRAGE DES ABONNEMENTS RÉELS')
    print('=' * 60)
    
    print(f"📊 Nombre total d'abonnements : {len(df)}")
    
    # 1. Exclure les trial-only subscriptions
    non_trial_only = df[df['trial_only_subscription'] == False].copy()
    print(f"📊 Après exclusion trial-only : {len(non_trial_only)} ({len(df) - len(non_trial_only)} exclus)")
    
    # 2. Exclure les abonnements trop courts (période de remboursement)
    if include_refund_period:
        min_real_duration = non_trial_only[non_trial_only['real_duration_days'] >= min_duration_days].copy()
        print(f"📊 Après exclusion période remboursement (<{min_duration_days}j) : {len(min_real_duration)} ({len(non_trial_only) - len(min_real_duration)} exclus)")
    else:
        min_real_duration = non_trial_only.copy()
    
    # 3. Statistiques des durées
    if len(min_real_duration) > 0:
        print(f"\n📈 STATISTIQUES DES DURÉES RÉELLES :")
        print(f"   • Durée moyenne : {min_real_duration['real_duration_days'].mean():.1f} jours")
        print(f"   • Durée médiane : {min_real_duration['real_duration_days'].median():.1f} jours")
        print(f"   • Durée min : {min_real_duration['real_duration_days'].min()} jours")
        print(f"   • Durée max : {min_real_duration['real_duration_days'].max()} jours")
    
    return min_real_duration

def analyze_subscription_patterns(df):
    """Analyse détaillée des patterns d'abonnement"""
    
    print('\n' + '=' * 60)
    print('ANALYSE DES PATTERNS D\'ABONNEMENT')
    print('=' * 60)
    
    # Répartition par statut
    print(f"\n📋 RÉPARTITION PAR STATUT :")
    status_counts = df['status'].value_counts()
    for status, count in status_counts.items():
        pct = (count / len(df)) * 100
        print(f"   • {status}: {count} ({pct:.1f}%)")
    
    # Durées de trial
    trial_data = df[df['trial_duration_days'].notna()]
    if len(trial_data) > 0:
        print(f"\n🧪 DURÉES DE TRIAL :")
        print(f"   • Trial moyen : {trial_data['trial_duration_days'].mean():.1f} jours")
        print(f"   • Trial médian : {trial_data['trial_duration_days'].median():.1f} jours")
        trial_counts = trial_data['trial_duration_days'].value_counts().head(5)
        print(f"   • Durées les plus fréquentes :")
        for days, count in trial_counts.items():
            print(f"     - {days} jours: {count} abonnements")
    
    # Abonnements cadeaux
    if 'is_gifted_member' in df.columns:
        gifted_count = df['is_gifted_member'].sum()
        print(f"\n🎁 ABONNEMENTS CADEAUX : {gifted_count} ({(gifted_count/len(df)*100):.1f}%)")


# Utilisation améliorée :
df = calculate_duration_advanced(df)

# Analyse complète
analyze_subscription_patterns(df)

# Filtrage pour les abonnements réels
real_subs = filter_real_subscriptions(df, min_duration_days=14)



def analyze_date_columns(df):
    """Analyse détaillée des colonnes de dates pour comprendre leur signification"""
    
    print('=' * 80)
    print('ANALYSE DES COLONNES DE DATES')
    print('=' * 80)
    
    date_columns = [
        'created_utc', 'start_utc', 'current_period_start_utc', 'current_period_end_utc',
        'trial_start_utc', 'trial_end_utc', 'canceled_at_utc', 'ended_at_utc'
    ]
    
    # 1. Statistiques de base pour chaque colonne
    print("\n📅 STATISTIQUES DE BASE PAR COLONNE :")
    print("-" * 50)
    
    for col in date_columns:
        if col in df.columns:
            non_null = df[col].notna().sum()
            null_count = df[col].isna().sum()
            pct_non_null = (non_null / len(df)) * 100
            
            print(f"\n{col}:")
            print(f"  • Valeurs non-nulles: {non_null}/{len(df)} ({pct_non_null:.1f}%)")
            
            if non_null > 0:
                min_date = df[col].min()
                max_date = df[col].max()
                print(f"  • Période: {min_date.strftime('%Y-%m-%d')} → {max_date.strftime('%Y-%m-%d')}")
    
    # 2. Comparaison des ordres chronologiques
    print("\n\n⏰ ANALYSE DE L'ORDRE CHRONOLOGIQUE :")
    print("-" * 50)
    
    # Sous-ensemble avec toutes les dates principales
    complete_data = df[
        df['created_utc'].notna() & 
        df['start_utc'].notna()
    ].copy()
    
    if len(complete_data) > 0:
        # created_utc vs start_utc
        same_created_start = (complete_data['created_utc'] == complete_data['start_utc']).sum()
        created_before_start = (complete_data['created_utc'] < complete_data['start_utc']).sum()
        created_after_start = (complete_data['created_utc'] > complete_data['start_utc']).sum()
        
        print(f"\n🔍 CREATED_UTC vs START_UTC ({len(complete_data)} abonnements):")
        print(f"  • created_utc == start_utc: {same_created_start} ({same_created_start/len(complete_data)*100:.1f}%)")
        print(f"  • created_utc < start_utc:  {created_before_start} ({created_before_start/len(complete_data)*100:.1f}%)")
        print(f"  • created_utc > start_utc:  {created_after_start} ({created_after_start/len(complete_data)*100:.1f}%)")
        
        if created_before_start > 0:
            diff_days = (complete_data['start_utc'] - complete_data['created_utc']).dt.days
            diff_positive = diff_days[diff_days > 0]
            if len(diff_positive) > 0:
                print(f"  • Écart moyen quand created < start: {diff_positive.mean():.1f} jours")
                print(f"  • Écart médian: {diff_positive.median():.1f} jours")
    
    # 3. Relation avec les trials
    print("\n\n🧪 ANALYSE DES TRIALS :")
    print("-" * 50)
    
    trial_data = df[
        df['trial_start_utc'].notna() & 
        df['trial_end_utc'].notna()
    ].copy()
    
    if len(trial_data) > 0:
        print(f"\nAbonnements avec trial: {len(trial_data)}")
        
        # trial_start vs created/start
        if len(trial_data[trial_data['created_utc'].notna()]) > 0:
            trial_vs_created = trial_data[trial_data['created_utc'].notna()]
            same_trial_created = (trial_vs_created['trial_start_utc'] == trial_vs_created['created_utc']).sum()
            print(f"  • trial_start == created: {same_trial_created}/{len(trial_vs_created)} ({same_trial_created/len(trial_vs_created)*100:.1f}%)")
        
        if len(trial_data[trial_data['start_utc'].notna()]) > 0:
            trial_vs_start = trial_data[trial_data['start_utc'].notna()]
            same_trial_start = (trial_vs_start['trial_start_utc'] == trial_vs_start['start_utc']).sum()
            print(f"  • trial_start == start: {same_trial_start}/{len(trial_vs_start)} ({same_trial_start/len(trial_vs_start)*100:.1f}%)")
    
    # 4. Échantillons pour inspection manuelle
    print("\n\n🔍 ÉCHANTILLONS POUR INSPECTION MANUELLE :")
    print("-" * 50)
    
    # Cas où created != start
    different_dates = df[
        df['created_utc'].notna() & 
        df['start_utc'].notna() & 
        (df['created_utc'] != df['start_utc'])
    ].copy()
    
    if len(different_dates) > 0:
        print(f"\n📋 ÉCHANTILLON - created_utc ≠ start_utc (5 premiers):")
        sample_cols = ['customer_name', 'status', 'created_utc', 'start_utc', 'trial_start_utc', 'trial_end_utc']
        sample_cols = [col for col in sample_cols if col in different_dates.columns]
        print(different_dates[sample_cols].head().to_string(index=False))
    
    # Cas avec trial
    trial_sample = df[
        df['trial_start_utc'].notna() & 
        df['created_utc'].notna() & 
        df['start_utc'].notna()
    ].copy()
    
    if len(trial_sample) > 0:
        print(f"\n📋 ÉCHANTILLON - Avec trial (5 premiers):")
        sample_cols = ['customer_name', 'status', 'created_utc', 'start_utc', 'trial_start_utc', 'trial_end_utc']
        sample_cols = [col for col in sample_cols if col in trial_sample.columns]
        print(trial_sample[sample_cols].head().to_string(index=False))
    
    # 5. Recommandations basées sur l'analyse
    print("\n\n💡 RECOMMANDATIONS :")
    print("-" * 50)
    
    if len(complete_data) > 0:
        same_pct = (same_created_start / len(complete_data)) * 100
        if same_pct > 80:
            print("✅ created_utc et start_utc sont identiques dans >80% des cas")
            print("   → Probablement safe d'utiliser l'un ou l'autre")
        elif created_before_start > created_after_start:
            print("⚠️  created_utc est souvent antérieur à start_utc")
            print("   → created_utc pourrait être la création du compte/prospect")
            print("   → start_utc pourrait être le début effectif de l'abonnement")
        else:
            print("🔍 Pattern incertain - inspection manuelle recommandée")
    
    if len(trial_data) > 0:
        print(f"\n🧪 {len(trial_data)} abonnements avec trial détectés")
        print("   → Vérifier si trial_start correspond à start ou created")
    
    return {
        'complete_data_count': len(complete_data) if len(complete_data) > 0 else 0,
        'same_created_start': same_created_start if len(complete_data) > 0 else 0,
        'trial_count': len(trial_data),
        'recommendation': 'start_utc' if created_before_start > same_created_start else 'created_utc'
    }

def suggest_duration_calculation(df):
    """Suggère la meilleure façon de calculer les durées basé sur l'analyse"""
    
    analysis_result = analyze_date_columns(df)
    
    print("\n\n🎯 SUGGESTION POUR LE CALCUL DE DURÉE :")
    print("=" * 50)
    
    recommended_start = analysis_result['recommendation']
    
    print(f"\n📊 Recommandation basée sur l'analyse :")
    print(f"   • Date de début recommandée: {recommended_start}")
    print(f"   • Date de fin: ended_at_utc (ou reference_date pour actifs)")
    
    # Code suggéré
    print(f"\n💻 CODE SUGGÉRÉ :")
    print("-" * 20)
    print(f"""
# Calcul de durée recommandé
df['subscription_start_date'] = df['{recommended_start}']

# Pour les abonnements terminés
df['total_duration_days'] = (df['ended_at_utc'] - df['subscription_start_date']).dt.days

# Pour inclure les abonnements actifs
df['duration_to_today'] = (reference_date - df['subscription_start_date']).dt.days

# Durée effective (en utilisant ended_at_utc si disponible, sinon aujourd'hui)
df['effective_duration_days'] = np.where(
    df['ended_at_utc'].notna(),
    (df['ended_at_utc'] - df['subscription_start_date']).dt.days,
    (reference_date - df['subscription_start_date']).dt.days
)
""")

# Lancer l'analyse
analysis_result = analyze_date_columns(df)
suggest_duration_calculation(df)




