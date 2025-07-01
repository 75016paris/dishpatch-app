# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DISHPATCH import preprocess_data, remove_multi_subscriptions, remove_high_volume_customers, clean_inconsistent_statuses, custom_multisub_aggregation, prepare_multisub_for_integration, integrate_with_subdf, cancel_during_trial, refund_period_end_utc, canceled_during_refund_period, full_member_status
from DISHPATCH import paying_members, add_ended_at_utc, calculate_duration, get_full_members_count, get_iso_week_bounds, get_weeks_in_iso_year, calculate_target_iso_week, get_new_trial_last_week, get_conversion_rate_last_weeks, get_churn_members_last_week, cus_renewal, get_new_full_members_last_week
from DISHPATCH import plot_weekly_trials_8_weeks, plot_weekly_trials_all_time, weekly_flow_8_weeks, weekly_flow_all_time, weekly_renewal_flow_8_weeks, weekly_renewal_flow_all_time, plot_cohort_conversion_funnel, plot_cohort_conversion_funnel_comparison
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
**Subscription Analytics Dashboard**
""")

# Download file
uploaded_file = st.file_uploader("Uploqd your CSV", type="csv")

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
    renewal_dict = cus_renewal(sub_df)
    last_week_new_full_member = get_new_full_members_last_week(sub_df, today_iso, 1, REFUND_PERIOD_DAYS)
    prev_week_new_full_member = get_new_full_members_last_week(sub_df, today_iso, 2, REFUND_PERIOD_DAYS)

    fig_trials_8w, trials_metrics_8w = plot_weekly_trials_8_weeks(sub_df, today_date, today_iso, num_weeks=8)
    fig_trials_all_time, trials_metrics_all = plot_weekly_trials_all_time(sub_df, today_date, today_iso)
    fig_flow_8w, metrics_8w = weekly_flow_8_weeks(sub_df, today_date, today_iso, num_weeks=8)
    fig_flow_all_time, weekly_flow_all_time_result = weekly_flow_all_time(sub_df, today_date, today_iso)
    fig_renewal_8w, renewal_metrics_8w = weekly_renewal_flow_8_weeks(sub_df, today_date, today_iso, num_weeks=8)
    fig_renewal_all_time, renewal_flow_results = weekly_renewal_flow_all_time(sub_df, today_date, today_iso)
    fig_cohort, last_cohort_dict = plot_cohort_conversion_funnel(sub_df, today_date, today_iso)
    fig_cohort_comparison, last_cohort_comparison = plot_cohort_conversion_funnel_comparison(sub_df, today_date, today_iso, last_cohort_dict)



    col1, col2, col3 = st.columns(3)
    col1.metric("Full Active member:", dict_full_member['active'])
    col2.metric("Active Full Member in 1st year", renewal_dict['active_in_y1'])
    col3.metric("Active Full Member in 2nd year", renewal_dict['active_in_y2'])

    st.metric("Conversion Rate (from Trial to Full Member):", f"{renewal_dict['conversion_rate']}%")
    st.markdown(f"*To be a full member a user must complete their trial, not request a refund, and not be gifted. (refund period {REFUND_PERIOD_DAYS} days)*")

    st.metric("Renewal Rate:", f"{renewal_dict['renewal_rate_y1_to_y2']}%")
    st.markdown(f"*Renewal rate from 1st year to 2nd year:* **{renewal_dict['renewal_rate_y1_to_y2']}% - {renewal_dict['refund_rate_y2']}%** *from 2nd year to 3rd year:* **{renewal_dict['renewal_rate_y2_to_y3']}% - {renewal_dict['refund_rate_y3']}%**")

    col1, col2 = st.columns(2)
    col1.metric("New trial last week:", {new_trial_last_week['trials_count']})
    col2.metric("Active Full Member in 1st year", renewal_dict['active_in_y1'])

    # Visualisations
    st.header("Performances Hebdomadaires")
    fig, ax = plt.subplots(figsize=(10, 6))
    # Ajoutez vos visualisations ici
    st.pyplot(fig)

    # Affichage des données
    st.header("Données Brutes")
    st.dataframe(sub_df.head())
