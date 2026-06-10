import os
import warnings
warnings.filterwarnings('ignore')

try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="FamineGuard AI", layout="wide", page_icon="🌾")
st.title("🌾 FamineGuard: Spatiotemporal GNN & Agentic RAG")
st.markdown("AIMS Senegal - Decision Support System")

@st.cache_resource
def load_resources():
    # Chargement du CSV
    try:
        df = pd.read_csv('ipc_sen_area_long_latest.csv')
        cols = df.columns.tolist()
        for c in ['Area', 'area_name', 'zone', 'title', 'ADM2_FR']:
            if c in cols:
                df = df.rename(columns={c: 'nom_zone'})
                break
        if 'nom_zone' not in df.columns:
            df['nom_zone'] = df.iloc[:, 0]
    except:
        df = pd.DataFrame({'nom_zone': ["Bakel", "Matam", "Podor", "Dakar", "Kolda", "Kanel"]})

    # Chargement du GeoJSON
    s_map = None
    if os.path.exists('ipc_sen.geojson'):
        try:
            s_map = gpd.read_file('ipc_sen.geojson')
            if 'title' not in s_map.columns:
                for c in s_map.columns:
                    if c.lower() in ['name', 'reg', 'admin', 'area']:
                        s_map = s_map.rename(columns={c: 'title'})
                        break
        except:
            s_map = None

    # Chargement du RAG
    v_store = None
    if os.path.exists('mon_index_chroma'):
        try:
            from langchain_community.embeddings import FakeEmbeddings
            v_store = Chroma(
                persist_directory="mon_index_chroma",
                embedding_function=FakeEmbeddings(size=384)
            )
        except:
            v_store = None

    return df, s_map, v_store

nodes_df, senegal_map, vectorstore = load_resources()

def agent_famine_guard(zone, phase, prix, ndvi, langue):
    try:
        rag_context = "Pas de données historiques trouvées dans l'index."
        if vectorstore:
            search_query = f"Intervention and crisis lessons for {zone} Senegal phase {phase}"
            docs = vectorstore.similarity_search(search_query, k=2)
            if docs:
                rag_context = "\n\n".join([d.page_content for d in docs])

        api_key = st.secrets["GROQ_API_KEY"]
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key)

        system_prompt = "Tu es un expert humanitaire de l'ONU au Sahel."
        user_prompt = f"""Réponds en {'FRANÇAIS' if langue=='French' else 'ENGLISH'}.
        ANALYSE GNN : La zone de {zone} est en Phase {phase}. Choc de prix x{prix} et NDVI x{ndvi}.
        DOCUMENTS HISTORIQUES : {rag_context}
        Donne une analyse du risque et 3 recommandations concrètes."""

        res = llm.invoke([("system", system_prompt), ("human", user_prompt)])
        return res.content
    except Exception as e:
        return f"⚠️ Erreur agent : {str(e)}"

with st.sidebar:
    st.header("🎮 Contrôles")
    villes = sorted(nodes_df['nom_zone'].dropna().unique())
    zone_choisie = st.selectbox("Sélectionnez une zone", villes)
    langue_choisie = st.radio("Langue", ["French", "English"])
    prix_slider = st.slider("Hausse Prix (Facteur)", 1.0, 5.0, 2.5)
    ndvi_slider = st.slider("État Végétation (NDVI)", 0.1, 1.0, 0.4)
    run = st.button("🚀 LANCER L'ANALYSE")

col1, col2 = st.columns([1, 1.3])

if run:
    ph = 4 if prix_slider > 2.8 or ndvi_slider < 0.2 else 3 if prix_slider > 2.0 else 2

    with st.spinner("Analyse Agentique en cours..."):
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')

        if senegal_map is not None:
            senegal_map["color"] = "#2ECC71"
            mask = senegal_map['title'].str.lower().str.contains(zone_choisie.lower(), na=False)
            senegal_map.loc[mask, "color"] = "#E74C3C"
            senegal_map.plot(color=senegal_map["color"], edgecolor="white", ax=ax)
        ax.set_axis_off()

        rapport = agent_famine_guard(zone_choisie, ph, prix_slider, ndvi_slider, langue_choisie)

        with col1:
            st.subheader("📍 Carte de Risque GNN")
            st.pyplot(fig)
            st.metric("Phase IPC Prédite", f"Phase {ph}", delta="CRITIQUE" if ph >= 4 else "ALERTE")

        with col2:
            st.subheader("🤖 Rapport de l'Agentic RAG")
            st.markdown(rapport)
else:
    st.info("Choisissez une ville et configurez un choc à gauche.")
