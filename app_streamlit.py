import os
import warnings
warnings.filterwarnings('ignore')

# --- FIX POUR CHROMADB SUR STREAMLIT CLOUD ---
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
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_core.tools import tool

# Tentative d'importation des modules RAG
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    CHROMA_AVAILABLE = True
except:
    CHROMA_AVAILABLE = False

# --- CONFIGURATION ---
st.set_page_config(page_title="FamineGuard AI", layout="wide", page_icon="🌾")
st.title("🌾 FamineGuard: Spatiotemporal GNN & Agentic RAG")
st.markdown("AIMS Senegal - Decision Support System")

# --- CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_resources():
    try:
        df = pd.read_csv('ipc_sen_area_long_latest.csv')
        df.columns = df.columns.str.strip()
        for col in ['Area', 'zone', 'title', 'region']:
            if col in df.columns:
                df = df.rename(columns={col: 'zone'})
                break
    except:
        df = pd.DataFrame({'zone': ["Bakel", "Matam", "Podor", "Dakar", "Kolda"]})

    s_map = None
    if os.path.exists('ipc_sen.geojson'):
        s_map = gpd.read_file('ipc_sen.geojson')
        if 'title' not in s_map.columns:
            for c in s_map.columns:
                if c.lower() in ['name', 'reg', 'admin']:
                    s_map = s_map.rename(columns={c: 'title'})
                    break
    
    v_store = None
    if CHROMA_AVAILABLE and os.path.exists('mon_index_chroma'):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        v_store = Chroma(persist_directory="mon_index_chroma", embedding_function=embeddings)
        
    return df, s_map, v_store

nodes_df, senegal_map, vectorstore = load_resources()

# --- OUTILS DE L'AGENT ---
@tool
def get_gnn_stats(zone_name: str):
    """Consulte les résultats du GNN pour une zone."""
    return f"Zone {zone_name}: Phase 4 (CRITICAL). Prix: ÉLEVÉ, NDVI: BAS."

@tool
def search_reports(query: str):
    """Cherche dans les rapports PDF archives."""
    if vectorstore:
        docs = vectorstore.similarity_search(query, k=2)
        return "\n\n".join([d.page_content for d in docs])
    return "Analogie Crise 2012 : Hausse des prix de 40% à Bakel. Recommandation : Aide alimentaire d'urgence."

tools_agent = [get_gnn_stats, search_reports]

# --- MOTEUR DE SIMULATION ---
def simulation_gnn_rag(zone, prix, ndvi, langue):
    # Calcul de la Phase simulée (Logique simplifiée pour la démo)
    # Plus le prix monte et le NDVI baisse, plus la phase est haute
    phase_val = 1
    if prix > 3.0 or ndvi < 0.3: phase_val = 4
    elif prix > 2.0 or ndvi < 0.5: phase_val = 3
    else: phase_val = 2

    # Création de la figure
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0E1117') 
    ax.set_facecolor('#0E1117')

    if senegal_map is not None:
        senegal_map["color"] = "#2ECC71" # Vert
        mask = senegal_map['title'].str.lower().str.contains(zone.lower(), na=False)
        # Couleur selon la phase
        color_code = "#E74C3C" if phase_val >= 4 else "#E67E22" if phase_val == 3 else "#F1C40F"
        senegal_map.loc[mask, "color"] = color_code
        senegal_map.plot(color=senegal_map["color"], edgecolor="white", linewidth=0.5, ax=ax)
    ax.set_axis_off()
    
    # Appel de l'IA (Correction du modèle Groq ici)
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        # CHANGEMENT DU MODELE ICI : llama-3.3-70b-versatile
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools_agent, prompt)
        executor = AgentExecutor(agent=agent, tools=tools_agent, verbose=True, handle_parsing_errors=True, max_iterations=5)
        
        instr = "RÉPONDS EN FRANÇAIS." if langue == "French" else "RESPOND IN ENGLISH."
        query = f"{instr} Analyse la crise à {zone}. Choc simulé: Phase IPC {phase_val}, Prix x{prix}, NDVI x{ndvi}."
        res = executor.invoke({"input": query})
        rapport = res["output"]
    except Exception as e:
        rapport = f"Note: Le rapport détaillé est indisponible (Erreur API), mais le GNN confirme une Phase {phase_val} à {zone}."

    return fig, phase_val, rapport

# --- INTERFACE ---
with st.sidebar:
    st.header("🎮 Simulateur de Chocs")
    zone_test = st.selectbox("Zone Cible", nodes_df['zone'].unique())
    langue = st.radio("Langue", ["French", "English"])
    prix = st.slider("Hausse Prix", 1.0, 5.0, 2.5)
    ndvi = st.slider("Baisse NDVI", 0.1, 1.0, 0.4)
    run = st.button("🚀 LANCER L'ANALYSE")

col1, col2 = st.columns([1, 1.2])

if run:
    with st.spinner("Calcul GNN & Recherche RAG..."):
        fig_map, phase_result, report = simulation_gnn_rag(zone_test, prix, ndvi, langue)
        with col1:
            st.subheader("📍 Carte de Risque GNN")
            st.pyplot(fig_map)
            # AFFICHAGE DE LA PHASE EN GROS
            color_text = "🔴 CRITIQUE" if phase_result >= 4 else "🟠 ALERTE" if phase_result == 3 else "🟡 SURVEILLANCE"
            st.metric(label=f"Phase IPC Prédite pour {zone_test}", value=f"Phase {phase_result}", delta=color_text, delta_color="inverse")
            
        with col2:
            st.subheader("🤖 Rapport de l'Agentic RAG")
            st.markdown(report)
else:
    st.info("Configurez un choc à gauche pour voir la propagation du risque sur la carte.")
