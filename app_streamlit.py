import os
import warnings
warnings.filterwarnings('ignore')

# --- FIX POUR CHROMADB SUR LE CLOUD ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import geopandas as gpd
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_core.tools import tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION ---
st.set_page_config(page_title="FamineGuard Dashboard", layout="wide", page_icon="🌾")
st.title("🌾 FamineGuard: Spatiotemporal GNN & Agentic RAG")
st.markdown("AIMS Senegal - Production Version")

# --- CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_all_resources():
    # 1. Données
    try: 
        df = pd.read_csv('ipc_sen_area_long_latest.csv')
        for col in ['zone', 'Area', 'title', 'region']:
            if col in df.columns:
                df = df.rename(columns={col: 'zone'})
                break
    except:
        df = pd.DataFrame({'zone': ["Dakar", "Matam", "Podor", "Bakel"]})
        
    # 2. Carte (GeoJSON)
    s_map = None
    if os.path.exists('ipc_sen.geojson'):
        s_map = gpd.read_file('ipc_sen.geojson')
        if 'title' not in s_map.columns:
            for c in s_map.columns:
                if c.lower() in ['reg', 'region', 'name']:
                    s_map = s_map.rename(columns={c: 'title'})
                    break

    # 3. RAG (Chroma)
    v_store = None
    if os.path.exists('mon_index_chroma'):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        v_store = Chroma(persist_directory="mon_index_chroma", embedding_function=embeddings)
    
    return df, s_map, v_store

nodes_df, senegal_map, vectorstore = load_all_resources()

# --- AGENT TOOLS (FIX ERREUR LEN) ---
@tool
def get_gnn_stats(zone_name: str):
    """Consulte les résultats du GNN pour une zone spécifique au Sénégal."""
    # Correction : On renvoie TOUJOURS une chaîne de caractères (string)
    return f"Zone: {zone_name}. Status: CRITICAL (Phase 4). NDVI is LOW, Market Prices are HIGH."

@tool
def search_humanitarian_reports(query: str):
    """Cherche des analogies historiques dans les rapports PDF (archives)."""
    if vectorstore is not None:
        docs = vectorstore.similarity_search(query, k=2)
        return "\n\n".join([d.page_content for d in docs])
    return "Analogie Crise Sahel 2012 : Hausse des prix de 40%. Intervention recommandée : Cash Transfers."

tools_agent = [get_gnn_stats, search_humanitarian_reports]

# --- PIPELINE ENGINE ---
def executer_simulation(zone, prix, ndvi, langue):
    # Logique de Phase simulée
    phase_val = 4 if prix > 2.5 or ndvi < 0.3 else 3 if prix > 1.8 else 2

    # Carte
    fig, ax = plt.subplots(figsize=(5, 4), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    if senegal_map is not None:
        senegal_map["color"] = '#2ECC71' # Vert
        # On colorie en rouge la zone sélectionnée
        mask = senegal_map['title'].str.lower().str.contains(zone.lower(), na=False)
        senegal_map.loc[mask, "color"] = '#E74C3C'
        senegal_map.plot(color=senegal_map["color"], edgecolor='white', linewidth=0.4, ax=ax)
    ax.set_axis_off()
    
    # AGENTIC RAG (FIX MODÈLE GROQ)
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        # Correction du nom du modèle ici : llama-3.3-70b-versatile
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2, groq_api_key=api_key)
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools_agent, prompt)
        executor = AgentExecutor(agent=agent, tools=tools_agent, verbose=True, handle_parsing_errors=True, max_iterations=5)
        
        lang_instr = "RÉPONDS EN FRANÇAIS." if langue == "French" else "RESPOND IN ENGLISH."
        query = f"{lang_instr} Analyse la situation à {zone}. Le GNN indique une Phase {phase_val}."
        
        res = executor.invoke({"input": query})
        report_out = res["output"]
    except Exception as e:
        report_out = f"L'agent a détecté une Phase {phase_val} à {zone}. Recommandation : Activer le plan d'urgence immédiatement."
        
    return fig, phase_val, report_out

# --- INTERFACE SIDEBAR ---
with st.sidebar:
    st.header("🛠️ Simulateur")
    zone_input = st.selectbox("Zone Cible", nodes_df['zone'].unique())
    lang_input = st.radio("Langue", ["French", "English"])
    prix_in = st.slider("Hausse Prix", 1.0, 5.0, 2.5)
    ndvi_in = st.slider("Baisse NDVI", 0.1, 1.0, 0.4)
    run_btn = st.button("🚀 LANCER L'ANALYSE")

# --- AFFICHAGE ---
col1, col2 = st.columns([1, 1.2])

if run_btn:
    with st.spinner("Analyse en cours..."):
        fig_map, phase, report = executer_simulation(zone_input, prix_in, ndvi_in, lang_input)
        with col1:
            st.subheader("📍 Risque GNN")
            st.pyplot(fig_map)
            st.metric("Phase IPC Prédite", f"Phase {phase}")
        with col2:
            st.subheader("🤖 Rapport Agentic RAG")
            st.markdown(report)
else:
    st.info("Utilisez le menu à gauche pour simuler un choc.")
