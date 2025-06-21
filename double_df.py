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
    sub_raw = pd.read_csv(file1_path, low_memory=False)
    inv_raw = pd.read_csv(file2_path, low_memory=False)
    print(f"\nSuccessfully loaded:")
    print(f"  sub_raw: {sub_raw.shape[0]} rows, {sub_raw.shape[1]} columns")
    print(f"  inv_raw: {inv_raw.shape[0]} rows, {inv_raw.shape[1]} columns")
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

print("\nDataFrames available as: sub_raw, inv_raw")
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

df1 = preprocess_data(sub_raw)


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
        'Date (UTC)' : 'date_utc',
        'Description': 'description',
        'Paid At (UTC)': 'paid_at_utc',
        'Amount Paid': 'amount_paid',
        'Subscription': 'subscription_id',
    }, inplace=True)

    return df

df2 = preprocess_data_invoice(inv_raw)

# %%
###################################################################################

#df1['subscription_id'].value_counts().head(60)

def add_invoice_columns_optimized(df1, df2, max_invoices=None):
    """
    Version optimisée utilisant merge pour de meilleures performances
    """
    
    # Préparation des données
    df2_clean = df2.dropna(subset=['subscription_id']).copy()
    df2_clean = df2_clean.sort_values([
        'subscription_id', 
        'Minimum Line Item Period Start (UTC)'
    ])
    
    # Déterminer le nombre max d'invoices
    if max_invoices is None:
        max_invoices = df2_clean.groupby('subscription_id').size().max()
        print(f"Nombre maximum d'invoices: {max_invoices}")
    
    # Méthode corrigée pour éviter le conflit de colonnes
    # Ajouter le numéro d'invoice directement
    df2_clean['invoice_number'] = df2_clean.groupby('subscription_id').cumcount() + 1
    
    # Filtrer pour garder seulement jusqu'à max_invoices
    df2_numbered = df2_clean[df2_clean['invoice_number'] <= max_invoices].copy()
    
    # Créer les colonnes pivotées
    dates_pivot = df2_numbered.pivot_table(
        index='subscription_id',
        columns='invoice_number',
        values='Minimum Line Item Period Start (UTC)',
        aggfunc='first'
    )
    
    prices_pivot = df2_numbered.pivot_table(
        index='subscription_id',
        columns='invoice_number',
        values='amount_paid',
        aggfunc='first'
    )
    
    # Renommer les colonnes
    date_columns = {col: f'invoice_{col}_date' for col in dates_pivot.columns}
    price_columns = {col: f'invoice_{col}_price' for col in prices_pivot.columns}
    
    dates_pivot = dates_pivot.rename(columns=date_columns)
    prices_pivot = prices_pivot.rename(columns=price_columns)
    
    # Merger avec df1
    df1_enriched = df1.merge(dates_pivot, on='subscription_id', how='left')
    df1_enriched = df1_enriched.merge(prices_pivot, on='subscription_id', how='left')
    
    print(f"✅ Version optimisée terminée!")
    return df1_enriched

# Maintenant ça devrait fonctionner
df1_enriched = add_invoice_columns_optimized(df1, df2, max_invoices=5)
df1_final = analyze_invoice_distribution(df1_enriched)


# Fonction pour analyser les résultats
def analyze_invoice_distribution(df_enriched):
    """Analyse la distribution des invoices"""
    
    print("=== ANALYSE DE LA DISTRIBUTION DES INVOICES ===")
    
    # Compter les invoices par subscription
    invoice_counts = []
    invoice_cols = [col for col in df_enriched.columns if col.startswith('invoice_') and col.endswith('_date')]
    
    for _, row in df_enriched.iterrows():
        count = sum(1 for col in invoice_cols if pd.notna(row[col]))
        invoice_counts.append(count)
    
    df_enriched['total_invoices'] = invoice_counts
    
    print("Distribution du nombre d'invoices:")
    print(df_enriched['total_invoices'].value_counts().sort_index())
    
    # Statistiques sur les prix
    price_cols = [col for col in df_enriched.columns if col.startswith('invoice_') and col.endswith('_price')]
    
    print(f"\nColonnes d'invoices créées:")
    print(f"📅 Dates: {len(invoice_cols)} colonnes")
    print(f"💰 Prix: {len(price_cols)} colonnes")
    
    return df_enriched


# ===== UTILISATION =====

# Version standard (recommandée pour la plupart des cas)
print("=== ENRICHISSEMENT DU DATAFRAME ===")
df1_enriched = add_invoice_columns_optimized(df1, df2, max_invoices=5)  # Limiter à 5 invoices max




df1_0_invoice_2 = df1_enriched[df1_enriched['invoice_2_price'].notna()]
df1_0_invoice_2['invoice_2_price'].value_counts()

df1_0_invoice_1 = df1_enriched[df1_enriched['invoice_1_price'].notna()]
df1_0_invoice_1['invoice_1_price'].value_counts()


df1_0_invoice_3 = df1_enriched[df1_enriched['invoice_3_price'].notna()]
df1_0_invoice_3['invoice_3_price'].value_counts()



df2['subscription_id'].value_counts()


# Vérification des subscription_id dans df2

# 1. Compter les occurrences de chaque subscription_id
subscription_counts = df2['subscription_id'].value_counts()

print("=== ANALYSE DES SUBSCRIPTION_ID DANS DF2 ===")
print(f"Total d'entrées dans df2: {len(df2)}")
print(f"Subscription_id uniques: {df2['subscription_id'].nunique()}")
print(f"Subscription_id null/NaN: {df2['subscription_id'].isnull().sum()}")

# 2. Maximum d'occurrences
max_count = subscription_counts.max()
print(f"\n📈 MAXIMUM d'invoices par subscription_id: {max_count}")

# 3. Subscription_id avec le plus d'occurrences
max_subscription = subscription_counts.idxmax()
print(f"🎯 Subscription_id avec le plus d'invoices: {max_subscription}")

# 4. Distribution des nombres d'invoices
print(f"\n=== DISTRIBUTION DU NOMBRE D'INVOICES ===")
count_distribution = subscription_counts.value_counts().sort_index()
print("Nombre d'invoices -> Combien de subscriptions:")
for invoices, subscriptions in count_distribution.items():
    print(f"  {invoices} invoices: {subscriptions} subscriptions")

# 5. Top 10 des subscription_id avec le plus d'invoices
print(f"\n=== TOP 10 SUBSCRIPTION_ID AVEC LE PLUS D'INVOICES ===")
top_10 = subscription_counts.head(10)
for sub_id, count in top_10.items():
    print(f"  {sub_id}: {count} invoices")

# 6. Statistiques détaillées
print(f"\n=== STATISTIQUES DÉTAILLÉES ===")
print(f"Moyenne d'invoices par subscription: {subscription_counts.mean():.2f}")
print(f"Médiane d'invoices par subscription: {subscription_counts.median():.2f}")
print(f"Écart-type: {subscription_counts.std():.2f}")

# 7. Subscription_id avec beaucoup d'invoices (>= 5)
high_invoice_subs = subscription_counts[subscription_counts >= 5]
print(f"\nSubscriptions avec 5+ invoices: {len(high_invoice_subs)}")
if len(high_invoice_subs) > 0:
    print("Exemples:")
    for sub_id, count in high_invoice_subs.head().items():
        print(f"  {sub_id}: {count} invoices")

# 8. Vérification des subscription_id les plus problématiques
print(f"\n=== EXAMINATION DU SUBSCRIPTION_ID AVEC LE PLUS D'INVOICES ===")
if max_count > 1:
    max_sub_data = df2[df2['subscription_id'] == max_subscription]
    print(f"Subscription: {max_subscription}")
    print(f"Nombre d'invoices: {len(max_sub_data)}")
    print("\nDétails des invoices:")
    relevant_cols = ['invoice_id', 'amount_paid', 'Minimum Line Item Period Start (UTC)', 'Status']
    available_cols = [col for col in relevant_cols if col in max_sub_data.columns]
    print(max_sub_data[available_cols].sort_values('Minimum Line Item Period Start (UTC)' if 'Minimum Line Item Period Start (UTC)' in available_cols else available_cols[0]))

# 9. Graphique simple de la distribution (optionnel)
try:
    import matplotlib.pyplot as plt
    
    # Distribution des nombres d'invoices (limité aux 20 premiers)
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Distribution du nombre d'invoices
    plt.subplot(1, 2, 1)
    count_dist_plot = subscription_counts.value_counts().sort_index().head(20)
    plt.bar(count_dist_plot.index, count_dist_plot.values)
    plt.xlabel('Nombre d\'invoices par subscription')
    plt.ylabel('Nombre de subscriptions')
    plt.title('Distribution du nombre d\'invoices')
    plt.xticks(rotation=45)
    
    # Subplot 2: Top 10 des subscription_id
    plt.subplot(1, 2, 2)
    top_10_plot = subscription_counts.head(10)
    plt.bar(range(len(top_10_plot)), top_10_plot.values)
    plt.xlabel('Subscription ID (top 10)')
    plt.ylabel('Nombre d\'invoices')
    plt.title('Top 10 des subscription_id')
    plt.xticks(range(len(top_10_plot)), [f'Sub {i+1}' for i in range(len(top_10_plot))], rotation=45)
    
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("\n(Matplotlib non disponible pour les graphiques)")

# 10. Retourner les informations principales
result_summary = {
    'max_invoices_per_subscription': max_count,
    'subscription_with_max_invoices': max_subscription,
    'total_unique_subscriptions': df2['subscription_id'].nunique(),
    'avg_invoices_per_subscription': subscription_counts.mean(),
    'subscriptions_with_multiple_invoices': len(subscription_counts[subscription_counts > 1])
}

print(f"\n=== RÉSUMÉ ===")
for key, value in result_summary.items():
    print(f"{key}: {value}")




no_sub_df2 = df2[df2['subscription_id'].isna()]
no_sub_df2



# Analyse des customer_id dans df2

print("=== ANALYSE DES CUSTOMER_ID DANS DF2 ===")

# 1. Compter les occurrences de chaque customer_id
customer_counts = df2['customer_id'].value_counts()

print(f"Total d'entrées dans df2: {len(df2)}")
print(f"Customer_id uniques: {df2['customer_id'].nunique()}")
print(f"Customer_id null/NaN: {df2['customer_id'].isnull().sum()}")

# 2. Maximum d'occurrences
max_count = customer_counts.max()
print(f"\n📈 MAXIMUM d'invoices par customer_id: {max_count}")

# 3. Customer_id avec le plus d'occurrences
max_customer = customer_counts.idxmax()
print(f"🎯 Customer_id avec le plus d'invoices: {max_customer}")

# 4. Distribution des nombres d'invoices par customer
print(f"\n=== DISTRIBUTION DU NOMBRE D'INVOICES PAR CUSTOMER ===")
count_distribution = customer_counts.value_counts().sort_index()
print("Nombre d'invoices -> Combien de customers:")
for invoices, customers in count_distribution.head(20).items():  # Limiter à 20 pour éviter trop d'output
    print(f"  {invoices} invoices: {customers} customers")

if len(count_distribution) > 20:
    print(f"... et {len(count_distribution) - 20} autres valeurs")

# 5. Top 10 des customer_id avec le plus d'invoices
print(f"\n=== TOP 10 CUSTOMER_ID AVEC LE PLUS D'INVOICES ===")
top_10 = customer_counts.head(10)
for customer_id, count in top_10.items():
    print(f"  {customer_id}: {count} invoices")

# 6. Statistiques détaillées
print(f"\n=== STATISTIQUES DÉTAILLÉES ===")
print(f"Moyenne d'invoices par customer: {customer_counts.mean():.2f}")
print(f"Médiane d'invoices par customer: {customer_counts.median():.2f}")
print(f"Écart-type: {customer_counts.std():.2f}")

# 7. Customers avec beaucoup d'invoices (>= 5)
high_invoice_customers = customer_counts[customer_counts >= 5]
print(f"\nCustomers avec 5+ invoices: {len(high_invoice_customers)}")
if len(high_invoice_customers) > 0:
    print("Exemples (top 10):")
    for customer_id, count in high_invoice_customers.head(10).items():
        print(f"  {customer_id}: {count} invoices")

# 8. Customers avec énormément d'invoices (>= 10)
very_high_invoice_customers = customer_counts[customer_counts >= 10]
print(f"\nCustomers avec 10+ invoices: {len(very_high_invoice_customers)}")
if len(very_high_invoice_customers) > 0:
    print("Liste complète:")
    for customer_id, count in very_high_invoice_customers.items():
        print(f"  {customer_id}: {count} invoices")

# 9. Analyse des invoices sans customer_id
invoices_without_customer = df2[df2['customer_id'].isnull()]
invoices_with_customer = df2[df2['customer_id'].notna()]

print(f"\n=== INVOICES SANS CUSTOMER_ID ===")
print(f"📊 Invoices SANS customer_id: {len(invoices_without_customer)}")
print(f"📊 Invoices AVEC customer_id: {len(invoices_with_customer)}")
print(f"📊 Pourcentage sans customer_id: {len(invoices_without_customer) / len(df2) * 100:.2f}%")

# 10. Examination du customer avec le plus d'invoices
if max_count > 1:
    print(f"\n=== EXAMINATION DU CUSTOMER AVEC LE PLUS D'INVOICES ===")
    max_customer_data = df2[df2['customer_id'] == max_customer]
    print(f"Customer: {max_customer}")
    print(f"Nombre d'invoices: {len(max_customer_data)}")
    print("\nDétails des invoices:")
    relevant_cols = ['invoice_id', 'subscription_id', 'amount_paid', 'Minimum Line Item Period Start (UTC)', 'Status']
    available_cols = [col for col in relevant_cols if col in max_customer_data.columns]
    
    # Trier par date si disponible
    if 'Minimum Line Item Period Start (UTC)' in available_cols:
        max_customer_data_sorted = max_customer_data.sort_values('Minimum Line Item Period Start (UTC)')
    else:
        max_customer_data_sorted = max_customer_data
    
    print(max_customer_data_sorted[available_cols])
    
    # Analyser ses subscription_ids
    customer_subscriptions = max_customer_data['subscription_id'].dropna().unique()
    print(f"\nSubscription_ids uniques pour ce customer: {len(customer_subscriptions)}")
    for sub_id in customer_subscriptions:
        sub_invoices = max_customer_data[max_customer_data['subscription_id'] == sub_id]
        print(f"  {sub_id}: {len(sub_invoices)} invoices")

# 11. Analyser les caractéristiques des invoices sans customer_id (si elles existent)
if len(invoices_without_customer) > 0:
    print(f"\n=== CARACTÉRISTIQUES DES INVOICES SANS CUSTOMER_ID ===")
    
    # Status
    if 'Status' in invoices_without_customer.columns:
        print("Status des invoices sans customer_id:")
        print(invoices_without_customer['Status'].value_counts())
    
    # Montants
    if 'amount_paid' in invoices_without_customer.columns:
        print(f"\nMontants des invoices sans customer_id:")
        print(f"Somme totale: {invoices_without_customer['amount_paid'].sum():.2f}")
        print(f"Moyenne: {invoices_without_customer['amount_paid'].mean():.2f}")
    
    # Échantillon
    print(f"\nÉchantillon des invoices sans customer_id:")
    sample_cols = ['invoice_id', 'subscription_id', 'amount_paid', 'Status']
    available_cols = [col for col in sample_cols if col in invoices_without_customer.columns]
    print(invoices_without_customer[available_cols].head())

# 12. Comparaison avec df1 (subscriptions)
if 'customer_id' in df1.columns:
    print(f"\n=== COMPARAISON AVEC DF1 (SUBSCRIPTIONS) ===")
    
    customers_in_df2 = set(df2['customer_id'].dropna().unique())
    customers_in_df1 = set(df1['customer_id'].unique())
    
    print(f"Customers uniques dans df2 (invoices): {len(customers_in_df2)}")
    print(f"Customers uniques dans df1 (subscriptions): {len(customers_in_df1)}")
    
    # Intersection et différences
    customers_in_both = customers_in_df2 & customers_in_df1
    customers_only_df2 = customers_in_df2 - customers_in_df1
    customers_only_df1 = customers_in_df1 - customers_in_df2
    
    print(f"Customers dans les DEUX: {len(customers_in_both)}")
    print(f"Customers SEULEMENT dans df2 (invoices): {len(customers_only_df2)}")
    print(f"Customers SEULEMENT dans df1 (subscriptions): {len(customers_only_df1)}")
    
    if len(customers_only_df2) > 0:
        print(f"\nExemples de customers avec invoices mais sans subscriptions:")
        for customer in list(customers_only_df2)[:5]:
            customer_invoices = df2[df2['customer_id'] == customer]
            print(f"  {customer}: {len(customer_invoices)} invoices")
    
    if len(customers_only_df1) > 0:
        print(f"\nExemples de customers avec subscriptions mais sans invoices:")
        for customer in list(customers_only_df1)[:5]:
            customer_subs = df1[df1['customer_id'] == customer]
            print(f"  {customer}: {len(customer_subs)} subscriptions")

# 13. Résumé final
result_summary = {
    'max_invoices_per_customer': max_count,
    'customer_with_max_invoices': max_customer,
    'total_unique_customers_in_df2': df2['customer_id'].nunique(),
    'avg_invoices_per_customer': customer_counts.mean(),
    'customers_with_multiple_invoices': len(customer_counts[customer_counts > 1]),
    'customers_with_5plus_invoices': len(high_invoice_customers),
    'invoices_without_customer_id': len(invoices_without_customer)
}

print(f"\n=== RÉSUMÉ CUSTOMER_ID ===")
for key, value in result_summary.items():
    if isinstance(value, float):
        print(f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value}")

# 14. Graphique simple de la distribution (optionnel)
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Distribution du nombre d'invoices par customer (limité)
    plt.subplot(2, 2, 1)
    count_dist_plot = customer_counts.value_counts().sort_index().head(20)
    plt.bar(count_dist_plot.index, count_dist_plot.values)
    plt.xlabel('Nombre d\'invoices par customer')
    plt.ylabel('Nombre de customers')
    plt.title('Distribution du nombre d\'invoices par customer')
    plt.xticks(rotation=45)
    
    # Subplot 2: Top 15 des customers
    plt.subplot(2, 2, 2)
    top_15_plot = customer_counts.head(15)
    plt.bar(range(len(top_15_plot)), top_15_plot.values)
    plt.xlabel('Customer ID (top 15)')
    plt.ylabel('Nombre d\'invoices')
    plt.title('Top 15 des customers avec le plus d\'invoices')
    plt.xticks(range(len(top_15_plot)), [f'C{i+1}' for i in range(len(top_15_plot))], rotation=45)
    
    # Subplot 3: Comparaison df1 vs df2 (si disponible)
    if 'customer_id' in df1.columns:
        plt.subplot(2, 2, 3)
        comparison_data = [len(customers_in_both), len(customers_only_df1), len(customers_only_df2)]
        labels = ['Dans les deux', 'Seulement df1\n(subscriptions)', 'Seulement df2\n(invoices)']
        plt.pie(comparison_data, labels=labels, autopct='%1.1f%%')
        plt.title('Répartition des customers entre df1 et df2')
    
    # Subplot 4: Distribution des montants payés par customer avec beaucoup d'invoices
    plt.subplot(2, 2, 4)
    if len(high_invoice_customers) > 0:
        high_customers_amounts = []
        for customer in high_invoice_customers.head(10).index:
            total_amount = df2[df2['customer_id'] == customer]['amount_paid'].sum()
            high_customers_amounts.append(total_amount)
        
        plt.bar(range(len(high_customers_amounts)), high_customers_amounts)
        plt.xlabel('Top customers (5+ invoices)')
        plt.ylabel('Montant total payé')
        plt.title('Montants totaux des top customers')
        plt.xticks(range(len(high_customers_amounts)), [f'C{i+1}' for i in range(len(high_customers_amounts))], rotation=45)
    
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("\n(Matplotlib non disponible pour les graphiques)")
