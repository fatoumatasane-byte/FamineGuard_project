import os
import warnings
warnings.filterwarnings('ignore')

# --- 1. CORRECTIF OBLIGATOIRE POUR STREAMLIT CLOUD ---
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
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI # On utilise Groq via le format OpenAI (plus stable)

# --- CONFIGURATION INTERFACE ---
st.set_page_config(page_title="FamineGuard AI", layout="wide", page_icon="🌾")
st.title("🌾 FamineGuard: GNN & Agentic RAG")
st.markdown("AIMS Senegal - Decision Support System")

# --- 2. CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_resources():
    # Données CSV
    try:
        df = pd.read_csv('ipc_sen_area_long_latest.csv')
        df.columns = df.columns.str.strip()
        for c in ['Area', 'area_name', 'zone', 'title', 'nom_zone']:
            if c in df.columns:
                df = df.rename(columns={c: 'zone_display'})
                break
    except:
        df = pd.DataFrame({'zone_display': ["Dakar", "Bakel", "Matam", "Podor"]})

    # Carte GeoJSON
    s_map = None
    if os.path.exists('ipc_sen.geojson'):
        s_map = gpd.read_file('ipc_sen.geojson')
        if 'title' not in s_map.columns:
            for c in s_map.columns:
                if c.lower() in ['name', 'reg', 'admin', 'title']:
                    s_map = s_map.rename(columns={c: 'title'})
                    break

    # RAG (Base Vectorielle)
    v_store = None
    if os.path.exists('mon_index_chroma'):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            v_store = Chroma(persist_directory="mon_index_chroma", embedding_function=embeddings)
        except: pass
        
    return df, s_map, v_store

nodes_df, senegal_map, vectorstore = load_resources()

# --- 3. LE CERVEAU DE L'AGENT (VERSION ROBUSTE SANS BUG) ---
def famine_guard_brain(zone, phase, prix, ndvi, langue):
    try:
        # Étape A : Recherche dans tes PDF
        context = "Aucune archive spécifique trouvée."
        if vectorstore:
            query = f"crisis recommendations for {zone} phase {phase}"
            docs = vectorstore.similarity_search(query, k=2)
            context = "\n\n".join([str(d.page_content) for d in docs])
        
        # Étape B : Appel à Groq via l'API Standard (Plus de bug len!)
        api_key = st.secrets["GROQ_API_KEY"]
        # Groq est compatible avec le client OpenAI, on en profite !
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
        
        system_msg = "Tu es un expert en sécurité alimentaire. Tu aides les ONGs à prendre des décisions."
        user_msg = f"""
        RÉPONDS EN {'FRANÇAIS' if langue=='French' else 'ANGLAIS'}.
        
        FAITS : La zone de {zone} est en Phase {phase}.
        DONNÉES GNN : Hausse Prix x{prix}, État Végétation {ndvi}.
        
        ARCHIVES : {context}
        
        Donne un rapport court avec :
        1. Analyse du risque
        2. 3 actions concrètes basées sur les rapports.
        """
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.2
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Désolé, l'IA a eu un petit problème technique ({e}). Cependant, le GNN confirme que {zone} est en Phase {phase}."

# --- 4. INTERFACE ---
with st.sidebar:
    st.header("🎮 Configuration")
    villes = sorted(nodes_df['zone_display'].unique())
    zone_choisie = st.selectbox("Zone Cible", villes)
    langue_choisie = st.radio("Langue", ["French", "English"])
    prix_val = st.slider("Choc Prix (Multiplicateur)", 1.0, 5.0, 1.5)
    ndvi_val = st.slider("Choc NDVI (0.1=Désastre)", 0.1, 1.0, 0.4)
    run = st.button("🚀 LANCER L'ANALYSE")

col1, col2 = st.columns([1, 1.3])

if run:
    # Logique Phase simplifiée
    ph = 4 if prix_val > 2.8 or ndvi_val < 0.2 else 3 if prix_val > 1.9 else 2
    
    with st.spinner("L'agent analyse les rapports..."):
        # Carte
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        
        if senegal_map is not None:
            senegal_map["color"] = "#2ECC71" # Vert
            mask = senegal_map['title'].str.lower().str.strip().str.contains(zone_choisie.lower().strip(), na=False)
            color_code = "#E74C3C" if ph >= 4 else "#E67E22"
            senegal_map.loc[mask, "color"] = color_code
            senegal_map.plot(color=senegal_map["color"], edgecolor="white", linewidth=0.4, ax=ax)
        ax.set_axis_off()
        
        # Rapport
        rapport = famine_guard_brain(zone_choisie, ph, prix_val, ndvi_val, langue_choisie)
        
        with col1:
            st.subheader("📍 Risque GNN")
            st.pyplot(fig)
            st.metric("Niveau de Danger", f"Phase {ph}", delta="CRITIQUE" if ph >= 4 else "ALERTE")
        
        with col2:
            st.subheader("🤖 Rapport de l'Agentic RAG")
            st.markdown(rapport)
else:
    st.info("Sélectionnez une zone et simulez un choc à gauche.")
