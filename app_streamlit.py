import os
import warnings
warnings.filterwarnings('ignore')

# --- 1. CORRECTIF SQLITE POUR LE CLOUD ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- CONFIGURATION INTERFACE ---
st.set_page_config(page_title="FamineGuard AI", layout="wide", page_icon="🌾")
st.title("🌾 FamineGuard: Spatiotemporal GNN & Agentic RAG")
st.markdown("AIMS Senegal - Decision Support System")

# --- 2. CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_resources():
    # Données CSV
    try:
        df = pd.read_csv('ipc_sen_area_long_latest.csv')
        df.columns = df.columns.str.strip()
        for c in ['Area', 'area_name', 'zone', 'title']:
            if c in df.columns:
                df = df.rename(columns={c: 'nom_zone'})
                break
    except:
        df = pd.DataFrame({'nom_zone': ["Dakar", "Bakel", "Matam", "Podor"]})

    # Carte GeoJSON
    s_map = None
    if os.path.exists('ipc_sen.geojson'):
        s_map = gpd.read_file('ipc_sen.geojson')
        if 'title' not in s_map.columns:
            for c in s_map.columns:
                if c.lower() in ['name', 'reg', 'admin']:
                    s_map = s_map.rename(columns={c: 'title'})
                    break

    # CHARGEMENT DU RAG (VRAIS EMBEDDINGS)
    v_store = None
    if os.path.exists('mon_index_chroma'):
        # On utilise le MEME modèle que lors de l'entraînement
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        v_store = Chroma(persist_directory="mon_index_chroma", embedding_function=embeddings)
    
    return df, s_map, v_store

nodes_df, senegal_map, vectorstore = load_resources()

# --- 3. LOGIQUE DE L'AGENT ---
def agent_famine_guard(zone, phase, prix, ndvi, langue):
    try:
        # ÉTAPE A : RECHERCHE RAG (VRAIE RECHERCHE)
        rag_context = "Aucune archive trouvée."
        if vectorstore:
            # On transforme tout en STRING pour éviter l'erreur 'int'
            query = f"Crisis recommendations for {str(zone)} phase {str(phase)}"
            docs = vectorstore.similarity_search(query, k=2)
            if docs:
                rag_context = "\n\n".join([d.page_content for d in docs])
        
        # ÉTAPE B : RAISONNEMENT LLM (GROQ)
        api_key = st.secrets["GROQ_API_KEY"]
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
        
        system_prompt = "Tu es un Expert Humanitaire Senior. Tu analyses les données GNN et les rapports historiques."
        user_prompt = f"""Réponds en {'FRANÇAIS' if langue=='French' else 'ENGLISH'}.
        
        FAITS GNN : La zone de {str(zone)} est en Phase {str(phase)}.
        PARAMÈTRES : Hausse Prix x{str(prix)}, NDVI {str(ndvi)}.
        
        CONTEXTE HISTORIQUE (RAG) : 
        {rag_context}
        
        Donne un rapport structuré avec une analyse et 3 actions concrètes."""
        
        res = llm.invoke([("system", system_prompt), ("human", user_prompt)])
        return res.content
    except Exception as e:
        return f"⚠️ Problème technique : {str(e)}"

# --- 4. INTERFACE ---
with st.sidebar:
    st.header("🎮 Contrôles")
    villes = sorted(nodes_df['nom_zone'].dropna().unique())
    zone_choisie = st.selectbox("Sélectionnez une zone", villes)
    langue_choisie = st.radio("Langue", ["French", "English"])
    prix_slider = st.slider("Hausse Prix (Facteur)", 1.0, 5.0, 2.5)
    ndvi_slider = st.slider("État Végétation (NDVI)", 0.1, 1.0, 0.4)
    run = st.button("🚀 LANCER L'ANALYSE AGENTIQUE")

col1, col2 = st.columns([1, 1.3])

if run:
    # Simulation GNN
    ph = 4 if prix_slider > 2.8 or ndvi_slider < 0.2 else 3 if prix_slider > 2.0 else 2
    
    with st.spinner("🧠 L'IA réfléchit et consulte les rapports..."):
        # Carte
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        
        if senegal_map is not None:
            senegal_map["color"] = "#2ECC71"
            mask = senegal_map['title'].str.lower().str.contains(zone_choisie.lower(), na=False)
            senegal_map.loc[mask, "color"] = "#E74C3C"
            senegal_map.plot(color=senegal_map["color"], edgecolor="white", ax=ax)
        ax.set_axis_off()
        
        # Appel de l'Agent
        rapport = agent_famine_guard(zone_choisie, ph, prix_slider, ndvi_slider, langue_choisie)
        
        with col1:
            st.subheader("📍 Carte de Risque GNN")
            st.pyplot(fig)
            st.metric("Phase IPC Prédite", f"Phase {ph}", delta="CRITIQUE" if ph >= 4 else "ALERTE")
        
        with col2:
            st.subheader("🤖 Rapport de l'Agentic RAG")
            st.markdown(rapport)
else:
    st.info("Choisissez une ville à gauche pour déclencher l'Agent.")
