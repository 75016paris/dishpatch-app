# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DISHPATCH import preprocess_data, remove_multi_subscriptions, remove_high_volume_customers, clean_inconsistent_statuses, custom_multisub_aggregation, prepare_multisub_for_integration, integrate_with_subdf, cancel_during_trial, refund_period_end_utc, canceled_during_refund_period, full_member_status  # Importez vos fonctions
from DISHPATCH import paying_members, add_ended_at_utc, calculate_duration, get_full_members_count, get_iso_week_bounds, get_weeks_in_iso_year, calculate_target_iso_week, get_new_trial_last_week, get_conversion_rate_last_weeks, get_churn_members_last_week, cus_renewal
import matplotlib
matplotlib.use('Agg')

# Configuration de la page
st.set_page_config(
    page_title="DISHPATCH Analytics",
    page_icon="📊",
    layout="wide"
)

# Titre de l'application
st.title("📊 DISHPATCH Subscription Analytics")
st.markdown("""
**Dashboard d'analyse des abonnements clients**
""")

# Téléchargement du fichier
uploaded_file = st.file_uploader("Télécharger le fichier CSV", type="csv")

if uploaded_file:
    today_date = pd.Timestamp('2025-05-23', tz='UTC') # For testing purposes
    today_iso = pd.to_datetime(today_date).isocalendar()

    # Set REFUND PERDIOD DURATION
    REFUND_PERIOD_DAYS = 14  # Duration of the refund period in days

    # Set thresholds for cleaning
    HIGH_VOLUME_THRESHOLD = 4
    DUPLICATE_THRESHOLD_MINUTES = 15


    # Traitement des données
    sub_raw = pd.read_csv(uploaded_file)
    sub_df = preprocess_data(sub_raw)
    sub_df, multisub_df = remove_multi_subscriptions(sub_df)
    multisub_df = remove_high_volume_customers(multisub_df)
    multisub_df = custom_multisub_aggregation(multisub_df)
    combined_df = integrate_with_subdf(multisub_df, sub_df)
    sub_df = combined_df.copy()
    sub_df = cancel_during_trial(sub_df)
    sub_df = refund_period_end_utc(sub_df, REFUND_PERIOD_DAYS)
    sub_df = canceled_during_refund_period(sub_df)
    sub_df = full_member_status(sub_df)
    sub_df = paying_members(sub_df)
    sub_df = add_ended_at_utc(sub_df, today_date)
    sub_df = calculate_duration(sub_df, today_date)
    dict_full_member = get_full_members_count(sub_df)
    new_trial_last_week = get_new_trial_last_week(sub_df, today_iso, weeks_back=1)
    new_trial_prev_week = get_new_trial_last_week(sub_df, today_iso, weeks_back=2)
    last_week_conversion_rate = get_conversion_rate_last_weeks(sub_df, today_iso, weeks_back=1)
    prev_week_conversion_rate = get_conversion_rate_last_weeks(sub_df, today_iso, weeks_back=2)
    last_week_churned_members = get_churn_members_last_week(sub_df, today_iso, weeks_back=1)
    prev_week_churned_members = get_churn_members_last_week(sub_df, today_iso, weeks_back=2)

    # Métriques clés
    full_members = get_full_members_count(sub_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Membres Actifs", full_members['active'])
    col2.metric("Taux de Conversion", f"{renewal_dict['conversion_rate']}%")
    col3.metric("Nouveaux Essais", new_trial_last_week['trials_count'])

    # Visualisations
    st.header("Performances Hebdomadaires")
    fig, ax = plt.subplots(figsize=(10, 6))
    # Ajoutez vos visualisations ici
    st.pyplot(fig)

    # Affichage des données
    st.header("Données Brutes")
    st.dataframe(sub_df.head())
