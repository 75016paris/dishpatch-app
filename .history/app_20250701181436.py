# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DISHPATCH import preprocess_data, calculate_duration, get_full_members_count  # Importez vos fonctions
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
    # Traitement des données
    sub_raw = pd.read_csv(uploaded_file)
    sub_df = preprocess_data(sub_raw)
    sub_df = calculate_duration(sub_df, pd.Timestamp.now(tz='UTC'))

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
