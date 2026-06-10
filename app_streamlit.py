import os
import warnings
warnings.filterwarnings('ignore')

# 1. FIX POUR CHROMADB (OBLIGATOIRE TOUT EN HAUT)
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

# --- CHARGEMENT SÉCURISÉ DES LIBS ---
try:
    import geopandas as gpd
    from langchain_groq import ChatGroq
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    LIBS_OK = True
except Exception as e:
    st.error(f"Erreur chargement bibliothèques : {e}")
    LIBS_OK = False

st.set_page_config(page_title="FamineGuard AI", layout="wide")
st.title("🌾 FamineGuard: GNN & Agentic RAG")

# --- CHARGEMENT DES FICHIERS ---
@st.cache_resource
def load_data():
    df = pd.DataFrame({'zone': ["Dakar", "Bakel", "Matam", "Podor", "Thies"]})
    if os.path.exists('ipc_sen_area_long_latest.csv'):
        df = pd.read_csv('ipc_sen_area_long_latest.csv')
    
    s_map = None
    if os.path.exists('ipc_sen.geojson'):
        s_map = gpd.read_file('ipc_sen.geojson')
    
    v_store = None
    if os.path.exists('mon_index_chroma'):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            v_store = Chroma(persist_directory="mon_index_chroma", embedding_function=embeddings)
        except: pass
    return df, s_map, v_store

if LIBS_OK:
    nodes_df, senegal_map, vectorstore = load_data()

    # --- LOGIQUE AGENT ---
    def run_agent(zone, phase, prix, ndvi, langue):
        try:
            rag_context = ""
            if vectorstore:
                docs = vectorstore.similarity_search(f"crisis recommendations phase {phase}", k=2)
                rag_context = "\n\n".join([d.page_content for d in docs])
            
            api_key = st.secrets["GROQ_API_KEY"]
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
            
            p = f"Tu es un expert. Réponds en {'Français' if langue=='French' else 'Anglais'}. Zone:{zone}, Phase:{phase}, Choc Prix:x{prix}. Rapports:{rag_context}. Donne 3 conseils."
            res = llm.invoke(p)
            return res.content
        except Exception as e:
            return f"L'agent est en maintenance. Recommandation pour {zone} (Phase {phase}) : Activer l'aide d'urgence."

    # --- UI ---
    with st.sidebar:
        st.header("🎮 Configuration")
        # Nettoyage des noms de zone
        col_zone = 'zone' if 'zone' in nodes_df.columns else nodes_df.columns[0]
        z_list = nodes_df[col_zone].unique()
        target = st.selectbox("Zone", z_list)
        lang = st.radio("Langue", ["French", "English"])
        p_val = st.slider("Prix", 1.0, 5.0, 2.5)
        n_val = st.slider("NDVI", 0.1, 1.0, 0.4)
        go = st.button("LANCER L'ANALYSE")

    if go:
        # Calcul phase simple pour la démo
        ph = 4 if p_val > 3.0 or n_val < 0.2 else 3 if p_val > 2.0 else 2
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📍 Carte")
            fig, ax = plt.subplots(facecolor='#0E1117')
            if senegal_map is not None:
                senegal_map.plot(color='#2ECC71', ax=ax)
                # Coloration simplifiée
                try:
                    target_map = senegal_map[senegal_map['title'].str.lower().str.contains(target.lower())]
                    target_map.plot(color='red', ax=ax)
                except: pass
            ax.set_axis_off()
            st.pyplot(fig)
            st.metric("Risque Prédit", f"Phase {ph}")
        
        with c2:
            st.subheader("🤖 Rapport Agent")
            st.write(run_agent(target, ph, p_val, n_val, lang))
    else:
        st.info("Prêt pour la simulation. Choisissez une zone à gauche.")
else:
    st.error("L'application n'a pas pu charger les bibliothèques d'IA. Vérifiez le fichier 'requirements.txt'.")
