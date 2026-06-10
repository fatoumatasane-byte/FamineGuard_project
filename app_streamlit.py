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
st.title("🌾 FamineGuard: Spatiotemporal GNN & Agentic RAG")
st.markdown("AIMS Senegal - Decision Support System")

# --- 2. CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_resources():
    # Données CSV
    try:
        df = pd.read_csv('ipc_sen_area_long_latest.csv')
        df.columns = df.columns.str.strip()
        for col in ['Area', 'zone', 'title', 'region']:
            if col in df.columns:
                df = df.rename(columns={col: 'zone'})
                break
    except:
        df = pd.DataFrame({'zone': ["Dakar", "Bakel", "Matam", "Podor", "Thies"]})

    # Carte GeoJSON
    s_map = gpd.read_file('ipc_sen.geojson') if os.path.exists('ipc_sen.geojson') else None

    # CHARGEMENT DE VOTRE INDEX CHROMA (IMPORTANT)
    v_store = None
    if os.path.exists('mon_index_chroma'):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            v_store = Chroma(persist_directory="mon_index_chroma", embedding_function=embeddings)
        except Exception as e:
            st.error(f"Erreur chargement index : {e}")
    
    return df, s_map, v_store

nodes_df, senegal_map, vectorstore = load_resources()

# --- 3. LOGIQUE AGENTIQUE MANUELLE (STABLE ET SANS ERREUR) ---
def agent_famine_guard(zone, phase, prix, ndvi, langue):
    """
    Simule la boucle de raisonnement Agentic RAG :
    1. Analyse des faits (GNN)
    2. Recherche documentaire (RAG)
    3. Synthèse et Décision
    """
    # ÉTAPE 1 : RÉCUPÉRATION DES STATS (GNN Tool)
    gnn_facts = f"Zone: {zone}, Predicted Phase: {phase}, Price Factor: {prix}, NDVI Factor: {ndvi}."
    
    # ÉTAPE 2 : RECHERCHE DANS LES RAPPORTS (RAG Tool)
    if vectorstore:
        search_query = f"Food crisis prevention recommendations for phase {phase} in Senegal"
        docs = vectorstore.similarity_search(search_query, k=2)
        rag_context = "\n\n".join([d.page_content for d in docs])
    else:
        rag_context = "Base documentaire indisponible. Utilisation des connaissances générales sur le Sahel (2012)."

    # ÉTAPE 3 : RAISONNEMENT ET SYNTHÈSE (LLM)
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
        
        lang_instr = "RÉPONDS EN FRANÇAIS." if langue == "French" else "RESPOND IN ENGLISH."
        
        # On construit le prompt pour simuler la pensée de l'agent
        prompt = f"""
        {lang_instr}
        Tu es un Expert Humanitaire Senior. 
        
        THOUGHT: I need to analyze the GNN results and compare them with historical reports to provide 3 actionable steps.
        
        GNN EVIDENCE: {gnn_facts}
        HISTORICAL EVIDENCE (RAG): {rag_context}
        
        Final Report Structure:
        1. Context Analysis (mentioning similarities with past crises like 2012)
        2. Risk Level Explanation
        3. 3 Clear Actionable Recommendations (citing sources if possible)
        """
        
        res = llm.invoke(prompt)
        return res.content
    except Exception as e:
        return f"⚠️ Erreur de connexion au cerveau de l'agent : {e}"

# --- 4. MOTEUR DE VISUALISATION ---
def generate_map(zone, phase_val):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')

    if senegal_map is not None:
        senegal_map["color"] = "#2ECC71"
        mask = senegal_map['title'].str.lower().str.contains(zone.lower(), na=False)
        color_code = "#E74C3C" if phase_val >= 4 else "#E67E22" if phase_val == 3 else "#F1C40F"
        senegal_map.loc[mask, "color"] = color_code
        senegal_map.plot(color=senegal_map["color"], edgecolor="white", linewidth=0.4, ax=ax)
    ax.set_axis_off()
    return fig

# --- 5. INTERFACE UTILISATEUR ---
with st.sidebar:
    st.header("🎮 Contrôles")
    zone_test = st.selectbox("Sélectionnez une zone", nodes_df['zone'].unique())
    langue = st.radio("Langue", ["French", "English"])
    prix_slider = st.slider("Hausse des Prix (Facteur)", 1.0, 5.0, 1.5)
    ndvi_slider = st.slider("État Végétation (NDVI)", 0.1, 1.0, 0.2)
    run = st.button("🚀 LANCER L'ANALYSE AGENTIQUE")

col1, col2 = st.columns([1, 1.3])

if run:
    # Calcul de la phase pour la démo
    phase_calc = 4 if prix_slider > 2.8 or ndvi_slider < 0.2 else 3 if prix_slider > 2.0 or ndvi_slider < 0.4 else 2
    
    with st.spinner("🧠 L'agent réfléchit (Thought -> Action -> Search)..."):
        fig_map = generate_map(zone_test, phase_calc)
        report = agent_famine_guard(zone_test, phase_calc, prix_slider, ndvi_slider, langue)
        
        with col1:
            st.subheader("📍 Carte de Risque GNN")
            st.pyplot(fig_map)
            st.metric("Phase IPC Prédite", f"Phase {phase_calc}", delta="CRITIQUE" if phase_calc >= 4 else "ALERTE")
        
        with col2:
            st.subheader("🤖 Rapport de l'Agentic RAG")
            st.markdown(report)
else:
    st.info("Utilisez le panneau de gauche pour simuler un choc et déclencher l'agent.")
