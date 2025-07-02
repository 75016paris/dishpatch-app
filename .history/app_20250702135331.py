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
from matplotlib.backends.backend_pdf import PdfPages
import io
from datetime import datetime



st.set_page_config(
    page_title="DISHPATCH Analytics",
    page_icon="📊",
    layout="wide"
)


st.markdown("""
<style>
/* Labels des métriques */
[data-testid="stMetricLabel"] p {
    font-size: 20px !important;
}

/* Valeurs des métriques */
[data-testid="stMetricValue"] {
    font-size: 23px !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)



st.title("📊 DISHPATCH")
st.markdown("""
**Subscription Analytics Dashboard**
""")

# Download file
uploaded_file = st.file_uploader("Upload the subscription csv", type="csv")

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




    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("📄 Download PDF Report", type="primary", use_container_width=True):
            with st.spinner("Generating PDF..."):
                pdf_buffer = generate_pdf_report(
                    sub_df, today_date, dict_full_member, renewal_dict,
                    new_trial_last_week, new_trial_prev_week,
                    last_week_new_full_member, prev_week_new_full_member,
                    last_week_churned_members, prev_week_churned_members,
                    trials_metrics_8w, trials_metrics_all, metrics_8w,
                    weekly_flow_all_time_result, renewal_metrics_8w,
                    renewal_flow_results, last_cohort_dict, REFUND_PERIOD_DAYS,
                    fig_trials_8w, fig_trials_all_time, fig_flow_8w, fig_flow_all_time,
                    fig_renewal_8w, fig_renewal_all_time, fig_cohort, fig_cohort_comparison
                )

                # File name with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"DISHPATCH_Analytics_Report_{timestamp}.pdf"

                st.download_button(
                    label="⬇️ Click here to download",
                    data=pdf_buffer,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True
                )

                st.success("✅ PDF generated successfully!")

    st.markdown("---")

    target_year, target_week, target_week_key = calculate_target_iso_week(today_iso, weeks_back=1)
    last_week_monday, last_week_sunday = get_iso_week_bounds(target_year, target_week)
    week_label = f"{last_week_monday.strftime('%d-%m-%y')} > {last_week_sunday.strftime('%d-%m-%y')}"

    target_year2, target_week2, target_week_key2 = calculate_target_iso_week(today_iso, weeks_back=2)
    last_week_monday2, last_week_sunday2 = get_iso_week_bounds(target_year2, target_week2)
    week_label2 = f"{last_week_monday2.strftime('%d-%m-%y')} > {last_week_sunday2.strftime('%d-%m-%y')}"


    st.header(f"{today_date.strftime('%B %d, %Y')}")
    st.markdown(f"**Last week :** {week_label} - **Previous week :** {week_label2}")

    st.markdown(" ")

    col1, col2, col3 = st.columns(3)
    col1.metric("Full Active Members :", dict_full_member['active'])
    st.markdown(f"*Active Full Member in 1st year :* **{renewal_dict['active_in_y1']}** *Active Full Member in 2nd year :* **{renewal_dict['active_in_y2']}**")
    # col2.metric("Active Full Member in 1st year :", renewal_dict['active_in_y1'])
    # col3.metric("Active Full Member in 2nd year :", renewal_dict['active_in_y2'])

    st.markdown(" ")

    st.metric("Conversion Rate (from Trial to Full Member):", f"{renewal_dict['conversion_rate']}%")
    st.markdown(f"*To be a full member a user must complete their trial, not request a refund, and not be gifted. (refund period {REFUND_PERIOD_DAYS} days)*")

    st.markdown(" ")

    st.metric("Renewal Rate:", f"{renewal_dict['renewal_rate_y1_to_y2']}%")
    st.markdown(f"*Renewal rate from 1st year to 2nd year:* **{renewal_dict['renewal_rate_y1_to_y2']}% - {renewal_dict['refund_rate_y2']}%** *ask for refund, from 2nd year to 3rd year:* **{renewal_dict['renewal_rate_y2_to_y3']}% - {renewal_dict['refund_rate_y3']}%** *ask for refund*")

    st.markdown(" ")
    st.markdown(" ")


    st.markdown(f"**Last week :** {week_label}")

    col1, col2, col3 = st.columns(3)
    col1.metric("New trial last week:", new_trial_last_week['trials_count'])
    col2.metric("New full member last week:", last_week_new_full_member['count'])
    col3.metric("Churn full member last week:", last_week_churned_members['count'])

    st.markdown(" ")

    st.markdown(f"**Previous week :** {week_label2}")
    col1, col2, col3 = st.columns(3)
    col1.metric("New trial previous week:", new_trial_prev_week['trials_count'])
    col2.metric("New full member previous week:", prev_week_new_full_member['count'])
    col3.metric("Churn full member previous week:", prev_week_churned_members['count'])






    st.markdown("---")


    # Visualisations
    st.header("NEW TRIALS")
    st.pyplot(fig_trials_8w)

    col1, col2, col3 = st.columns(3)
    col1.metric("Last week trial :", trials_metrics_8w['latest_week'])
    col2.metric("Previous week trial :", trials_metrics_8w['previous_week'])
    col3.metric("Week-over-week change :", f"{trials_metrics_8w['week_over_week_change']} ({trials_metrics_8w['week_over_week_pct']:.1f}%)")


    col1, col2 = st.columns(2)
    col1.metric("Average trial per week (8 weeks) :", f"{trials_metrics_8w['average_per_week']:.0f}")
    col2.metric("Recent trial average (4 weeks) :", f"{trials_metrics_8w['recent_4w_avg']:.0f}")

    col1, col2 = st.columns(2)
    col1.metric("Max Trial per week (8 weeks):", f"{trials_metrics_8w['max_week']} - ({trials_metrics_8w['max_week_label']})")
    col1.markdown(f"({trials_metrics_8w['max_week_label']})")
    col2.metric("Min week:", f"{trials_metrics_8w['min_week']} - ({trials_metrics_8w['min_week_label']})")

    st.markdown(" ")

    st.pyplot(fig_trials_all_time)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total trials (all time):", f"{trials_metrics_all['total_trials']:.0f}")
    col2.metric("Average per week:", f"{trials_metrics_all['average_per_week']:.0f}")
    col3.metric("Recent 4-week average:", f"{trials_metrics_8w['recent_4w_avg']:.0f}")

    col1, col2 = st.columns(2)
    col1.metric("Max Trial week:", f"{trials_metrics_all['max_week']} - ({trials_metrics_all['max_week_label']})")
    col2.metric("Min Trial week:", f"{trials_metrics_all['min_week']} - ({trials_metrics_all['min_week_label']})")

    st.markdown("---")

    st.header("FULL MEMBERS FLOW")
    st.pyplot(fig_flow_8w)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total conversions (8 weeks):", f"{metrics_8w['conversions']}")
    col2.metric("Total churn (8 weeks):", f"{metrics_8w['churn']}")
    col3.metric("Net growth (8 weeks):", f"{metrics_8w['net_growth']}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Average conversions per week (8 weeks):", f"{metrics_8w['avg_conversions_per_week']:.1f}")
    col2.metric("Average churn per week (8 weeks):", f"{metrics_8w['avg_churn_per_week']:.1f}")
    col3.metric("Average Net growth (8 weeks):", f"{metrics_8w['avg_net_growth']:.1f}")

    col1, col2 = st.columns(2)
    col1.metric("Max conversions week:", f"{metrics_8w['max_conv_value']} - ({metrics_8w['max_conv_label']})")
    col2.metric("Min conversions week:", f"{metrics_8w['min_conv_value']} - ({metrics_8w['min_conv_label']})")

    st.markdown(" ")
    st.pyplot(fig_flow_all_time)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total conversions (all time):", f"{weekly_flow_all_time_result['total_conversions']}")
    col2.metric("Churn (all time):", f"{weekly_flow_all_time_result['total_churn']}")
    col3.metric("Net growth (all time):", f"{weekly_flow_all_time_result['net_growth']}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Average conversions per week (all time)", f"{weekly_flow_all_time_result['avg_conversions_per_week']:.1f}")
    col2.metric("Average churn per week (all time):", f"{weekly_flow_all_time_result['avg_churn_per_week']:.1f}")
    col3.metric("Net growth average (all time):", f"{weekly_flow_all_time_result['avg_net_per_week']:.1f}")

    col1, col2 = st.columns(2)
    col1.metric("Max conversions week:", f"{weekly_flow_all_time_result['max_conv_value']} - ({weekly_flow_all_time_result['max_conv_label']})")
    col2.metric("Min conversions week:", f"{weekly_flow_all_time_result['min_conv_value']} - ({weekly_flow_all_time_result['min_conv_label']})")


    st.markdown("---")


    st.header("RENEWAL FLOW")
    st.pyplot(fig_renewal_8w)

    col1, col2 = st.columns(2)
    col1.metric("Total renewals (8 weeks):", f"{renewal_metrics_8w['total_renewals']}")
    col2.metric("Total churn (8 weeks):", f"{renewal_metrics_8w['total_churn']}")

    col1, col2 = st.columns(2)
    col1.metric("Post-Renewal Churn:", f"{renewal_metrics_8w['churn_post_renewal']}")
    col2.metric("Refund Churn:", f"{renewal_metrics_8w['churn_refund_renewal']}")

    st.pyplot(fig_renewal_all_time)

    col1, col2 = st.columns(2)
    col1.metric("Total renewals (8 weeks):", f"{renewal_flow_results['total_renewals']}")
    col2.metric("Total churn (8 weeks):", f"{renewal_flow_results['total_churn']}")

    col1, col2 = st.columns(2)
    col1.metric("Post-Renewal Churn:", f"{renewal_flow_results['total_churn_post_renewal']}")
    col2.metric("Refund Churn:", f"{renewal_flow_results['total_churn_refund_renewal']}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Average renewals per week (all time):", f"{renewal_flow_results['avg_renewals_per_week']:.1f}")
    col2.metric("Average Post-Renewal Churn per week (all time):", f"{renewal_flow_results['avg_post_churn_per_week']:.1f}")
    col3.metric("Average Refund per week (all time):", f"{renewal_flow_results['avg_refund_per_week']:.1f}")

    st.markdown("---")

    st.header("CONVERSION FUNNEL")
    st.pyplot(fig_cohort)

    col1, col2, col3 = st.columns(3)
    col1.metric("Drop-off during trial:", f"{last_cohort_dict['drop_off_trial']:.1f}%")
    col2.metric("Drop-off during refund:", f"{last_cohort_dict['drop_off_refund']:.1f}%")
    col3.metric("Total drop-off:", f"{last_cohort_dict['total_drop_off']:.1f}%")

    st.pyplot(fig_cohort_comparison)
