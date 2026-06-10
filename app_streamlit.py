import os
import warnings
warnings.filterwarnings('ignore')

# --- 1. CORRECTIF OBLIGATOIRE POUR CHROMADB SUR LE CLOUD ---
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
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION INTERFACE ---
st.set_page_config(page_title="FamineGuard AI", layout="wide", page_icon="🌾")
st.title("🌾 FamineGuard: GNN & Agentic RAG Platform")
st.markdown("AIMS Senegal - Decision Support System")

# --- 2. CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_resources():
    # Charger les zones depuis le CSV
    try:
        df = pd.read_csv('ipc_sen_area_long_latest.csv')
        df.columns = df.columns.str.strip()
        for col in ['Area', 'zone', 'title', 'region']:
            if col in df.columns:
                df = df.rename(columns={col: 'zone'})
                break
    except:
        df = pd.DataFrame({'zone': ["Dakar", "Bakel", "Matam", "Podor", "Thies"]})
    
    # Charger la carte GeoJSON
    s_map = None
    if os.path.exists('ipc_sen.geojson'):
        s_map = gpd.read_file('ipc_sen.geojson')
        if 'title' not in s_map.columns:
            for c in s_map.columns:
                if c.lower() in ['name', 'reg', 'admin']:
                    s_map = s_map.rename(columns={c: 'title'})
                    break

    # Charger l'index RAG (Chroma)
    v_store = None
    if os.path.exists('mon_index_chroma'):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        v_store = Chroma(persist_directory="mon_index_chroma", embedding_function=embeddings)
    
    return df, s_map, v_store

nodes_df, senegal_map, vectorstore = load_resources()

# --- 3. LOGIQUE DE L'AGENT EXPERT (AGENTIC RAG) ---
def agent_famine_guard_logic(zone, phase, prix, ndvi, langue):
    """
    Cette fonction remplace l'AgentExecutor instable. 
    Elle réalise la boucle de raisonnement : Faits -> Recherche -> Décision.
    """
    # Étape 1 : On récupère le contexte du RAG
    if vectorstore:
        search_query = f"Food security emergency response recommendations for phase {phase} in Senegal"
        docs = vectorstore.similarity_search(search_query, k=2)
        rag_context = "\n\n".join([d.page_content for d in docs])
    else:
        rag_context = "Historical reports unavailable. Using general knowledge for Sahel crisis (2012)."

    # Étape 2 : L'IA raisonne et produit le rapport
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
        
        lang_prompt = "Réponds en FRANÇAIS." if langue == "French" else "Respond in ENGLISH."
        
        prompt = f"""
        {lang_prompt}
        Tu es un Expert Humanitaire Senior du système FamineGuard.
        
        CONTEXTE DU GNN (Layer 2) :
        - Zone : {zone}
        - Phase IPC prédite : {phase}
        - Hausse Prix : x{prix}
        - Baisse Végétation : x{ndvi}
        
        ARCHIVES HISTORIQUES (Layer 3 - RAG) :
        {rag_context}
        
        TON RAISONNEMENT AGENTIQUE :
        1. Analyse les faits du GNN.
        2. Compare avec les preuves historiques des rapports cités ci-dessus.
        3. Formule 3 actions prioritaires.
        
        Structure ta réponse de manière professionnelle.
        """
        
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"⚠️ Erreur de connexion à l'IA : {e}"

# --- 4. INTERFACE UTILISATEUR ---
with st.sidebar:
    st.header("🎮 Contrôles")
    zone_test = st.selectbox("Sélectionnez une zone", nodes_df['zone'].unique())
    langue = st.radio("Langue de l'Agent", ["French", "English"])
    prix_slider = st.slider("Hausse des Prix (Facteur)", 1.0, 5.0, 3.0)
    ndvi_slider = st.slider("État Végétation (NDVI)", 0.1, 1.0, 0.2)
    run = st.button("🚀 LANCER L'ANALYSE")

col1, col2 = st.columns([1, 1.3])

if run:
    # Calcul de la Phase (Simulation GNN)
    # On rend la phase dynamique pour la démo
    if prix_slider > 3.0 or ndvi_slider < 0.2: phase_val = 4
    elif prix_slider > 2.0 or ndvi_slider < 0.4: phase_val = 3
    else: phase_val = 2

    with st.spinner("🧠 L'agent analyse les données et les rapports..."):
        # 1. Carte
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        
        if senegal_map is not None:
            senegal_map["color"] = "#2ECC71" # Tout vert par défaut
            # On cherche la zone (insensible à la casse et aux espaces)
            mask = senegal_map['title'].str.lower().str.strip().str.contains(zone_test.lower().strip(), na=False)
            color_code = "#E74C3C" if phase_val >= 4 else "#E67E22" if phase_val == 3 else "#F1C40F"
            senegal_map.loc[mask, "color"] = color_code
            senegal_map.plot(color=senegal_map["color"], edgecolor="white", linewidth=0.4, ax=ax)
        ax.set_axis_off()

        # 2. Rapport de l'Agent
        report = agent_famine_guard_logic(zone_test, phase_val, prix_slider, ndvi_slider, langue)
        
        with col1:
            st.subheader("📍 Carte de Risque GNN")
            st.pyplot(fig)
            st.metric("Phase IPC Prédite", f"Phase {phase_val}", delta="CRITIQUE" if phase_val >= 4 else "ALERTE")
        
        with col2:
            st.subheader("🤖 Rapport de l'Agentic RAG")
            st.markdown(report)
else:
    st.info("Configurez un scénario à gauche et cliquez sur le bouton pour lancer l'Analyse.")
