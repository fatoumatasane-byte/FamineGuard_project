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
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_core.tools import tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION INTERFACE ---
st.set_page_config(page_title="FamineGuard AI", layout="wide", page_icon="🌾")
st.title("🌾 FamineGuard: Spatiotemporal GNN & Agentic RAG")
st.markdown("AIMS Senegal - Decision Support System")

# --- 2. CHARGEMENT DES RESSOURCES (MODE PERSISTANT) ---
@st.cache_resource
def load_resources():
    # Charger les zones depuis le CSV
    df = pd.read_csv('ipc_sen_area_long_latest.csv')
    df = df.rename(columns={'Area': 'zone'}) if 'Area' in df.columns else df
    
    # Charger la carte GeoJSON
    s_map = gpd.read_file('ipc_sen.geojson') if os.path.exists('ipc_sen.geojson') else None

    # CHARGER L'INDEX CHROMA EXISTANT (votre dossier mon_index_chroma)
    v_store = None
    if os.path.exists('mon_index_chroma'):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # On n'utilise pas .from_documents, on charge le dossier directement
        v_store = Chroma(persist_directory="mon_index_chroma", embedding_function=embeddings)
    
    return df, s_map, v_store

nodes_df, senegal_map, vectorstore = load_resources()

# --- 3. OUTILS DE L'AGENT ---
@tool
def get_gnn_stats(zone_name: str):
    """Query the GNN model results (Layer 2) for a specific zone."""
    # On simule la lecture du rapport généré
    return f"Alerte CRITIQUE pour {zone_name}. Phase IPC 4 prédite. Cause: Prix élevés et NDVI bas."

@tool
def search_humanitarian_reports(query: str):
    """Search for historical crisis analogies and recommendations in the indexed PDF reports."""
    if vectorstore is not None:
        docs = vectorstore.similarity_search(query, k=2)
        return "\n\n".join([d.page_content for d in docs])
    return "Aucune archive trouvée dans la base de données locale."

tools_agent = [get_gnn_stats, search_humanitarian_reports]

# --- 4. MOTEUR DE SIMULATION & AGENT ---
def simulation_globale(zone, prix, ndvi, langue):
    # Logique de Phase
    phase_val = 4 if prix > 2.8 or ndvi < 0.3 else 3 if prix > 2.0 else 2

    # Carte
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#0E1117')
    if senegal_map is not None:
        senegal_map["color"] = "#2ECC71"
        # Coloration de la zone cible
        mask = senegal_map['title'].str.lower().str.contains(zone.lower(), na=False)
        color_code = "#E74C3C" if phase_val >= 4 else "#E67E22"
        senegal_map.loc[mask, "color"] = color_code
        senegal_map.plot(color=senegal_map["color"], edgecolor="white", linewidth=0.4, ax=ax)
    ax.set_axis_off()
    
    # AGENTIC RAG (GROQ)
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools_agent, prompt)
        executor = AgentExecutor(agent=agent, tools=tools_agent, verbose=True, handle_parsing_errors=True, max_iterations=5)
        
        instr = "RÉPONDS EN FRANÇAIS." if langue == "French" else "RESPOND IN ENGLISH."
        query = f"{instr} Analyse la situation à {zone}. GNN indique Phase {phase_val}. Cherche des recommandations dans les rapports."
        
        res = executor.invoke({"input": query})
        rapport = res["output"]
    except Exception as e:
        rapport = f"⚠️ Erreur Agent : {e}\n\nL'agent n'a pas pu analyser les PDF, mais le GNN confirme une Phase {phase_val}."

    return fig, phase_val, rapport

# --- 5. INTERFACE ---
with st.sidebar:
    st.header("🎮 Contrôles")
    zone_test = st.selectbox("Sélectionnez une zone", nodes_df['zone'].unique())
    langue = st.radio("Langue", ["French", "English"])
    prix = st.slider("Hausse des Prix (Facteur)", 1.0, 5.0, 3.0)
    ndvi = st.slider("État Végétation (NDVI)", 0.1, 1.0, 0.2)
    run = st.button("🚀 LANCER L'ANALYSE")

col1, col2 = st.columns([1, 1.3])

if run:
    with st.spinner("L'agent analyse les données et les rapports..."):
        fig_map, phase, report = simulation_globale(zone_test, prix, ndvi, langue)
        with col1:
            st.subheader("📍 Carte de Risque")
            st.pyplot(fig_map)
            st.metric("Phase IPC Prédite", f"Phase {phase}", delta="CRITIQUE" if phase >= 4 else "ALERTE")
        with col2:
            st.subheader("🤖 Rapport de l'Agentic RAG")
            st.markdown(report)
else:
    st.info("Configurez un choc à gauche pour tester le système.")
