import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from openai import OpenAI

warnings.filterwarnings('ignore')

# --- 1. CORRECTIF SQLITE3 (STREAMLIT CLOUD) ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# --- CONFIGURATION INTERFACE ---
st.set_page_config(page_title="FamineGuard AI", layout="wide", page_icon="🌾")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { border: 2px solid #e74c3c; padding: 15px; border_radius: 10px; background-color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CHARGEMENT DES RESSOURCES (Géo + RAG) ---
@st.cache_resource
def load_resources():
    # Carte GeoJSON
    gdf = gpd.read_file('ipc_sen.geojson')
    if 'title' not in gdf.columns:
        gdf['title'] = gdf['ADM2_FR'] if 'ADM2_FR' in gdf.columns else gdf.index.astype(str)
    
    # Correction des géométries pour les calculs de centres
    gdf['centroid'] = gdf.geometry.centroid

    # Chargement du Vectorstore Chroma avec le FIX 'int has no len'
    v_store = None
    chroma_path = "mon_index_chroma"
    if os.path.exists(chroma_path):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_community.vectorstores import Chroma
            
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # FIX CRUCIAL : collection_metadata empêche le bug 'int' len()
            v_store = Chroma(
                persist_directory=chroma_path, 
                embedding_function=embeddings,
                collection_metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            st.sidebar.error(f"Erreur RAG : {e}")
    
    return gdf, v_store

gdf, vectorstore = load_resources()

# --- 3. INITIALISATION SESSION ---
if 'selected_zone' not in st.session_state:
    st.session_state.selected_zone = gdf['title'].iloc[0]

# --- 4. LOGIQUE AGENTIC RAG (Tools + Brain) ---
def tool_search_rag(query):
    if vectorstore is None: return "Archives non disponibles."
    try:
        docs = vectorstore.similarity_search(query, k=3)
        return "\n\n".join([f"[SOURCE: {d.metadata.get('source','PDF')}] {d.page_content}" for d in docs])
    except: return "Erreur lors de la recherche documentaire."

def famine_guard_agent(zone, phase, prix, ndvi, langue):
    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=st.secrets["GROQ_API_KEY"])
        
        # Récupération des archives
        rag_data = tool_search_rag(f"food security crisis {zone} Senegal history interventions")
        
        system_prompt = "Tu es un expert senior du PAM et de FEWS NET. Réponds avec précision et cite tes sources."
        user_prompt = f"""
        LANGUE : {langue}
        ZONE : {zone}
        GNN PREDICTION : Phase {phase} (Prix x{prix}, NDVI {ndvi})
        ARCHIVES PDF : {rag_data}
        
        Produis un rapport : 1. Analyse du Choc | 2. Analogies Historiques | 3. Recommandations prioritaires.
        """
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur de l'agent : {e}"

# --- 5. INTERFACE SIDEBAR ---
with st.sidebar:
    st.title("🌾 FamineGuard")
    st.write(f"📍 Zone : **{st.session_state.selected_zone}**")
    prix_val = st.slider("Choc Prix", 1.0, 5.0, 1.8)
    ndvi_val = st.slider("Végétation (NDVI)", 0.1, 1.0, 0.35)
    langue = st.selectbox("Langue", ["Français", "English"])
    
    # Calcul Phase IPC
    phase = 4 if prix_val > 2.5 or ndvi_val < 0.2 else (3 if prix_val > 1.8 or ndvi_val < 0.35 else 2)
    color_hex = "#E74C3C" if phase >= 4 else ("#E67E22" if phase == 3 else "#2ECC71")
    
    st.metric("Risque Prédit", f"Phase {phase}", delta="CRITIQUE" if phase >= 4 else "STABLE")

# --- 6. VISUALISATION : CARTE + GRAPHE ---
col1, col2 = st.columns([1.3, 1])

with col1:
    st.subheader("🌐 Graphe de Propagation Spatio-Temporelle")
    
    # Carte claire pour meilleure visibilité
    m = folium.Map(location=[14.5, -14.5], zoom_start=7, tiles="CartoDB positron")

    # 1. Dessiner le GRAPHE (Edges) en orange vif
    for i, row in gdf.iterrows():
        p1 = [row.centroid.y, row.centroid.x]
        # On relie aux voisins (ceux qui touchent la bordure)
        neighbors = gdf[gdf.geometry.touches(row.geometry)]
        for _, neighbor in neighbors.iterrows():
            p2 = [neighbor.centroid.y, neighbor.centroid.x]
            folium.PolyLine([p1, p2], color="#F39C12", weight=1.5, opacity=0.6).add_to(m)

    # 2. Dessiner les ZONES (Polygones)
    folium.GeoJson(
        gdf,
        style_function=lambda x: {
            'fillColor': color_hex if x['properties']['title'] == st.session_state.selected_zone else '#BDC3C7',
            'color': 'black', 'weight': 1, 'fillOpacity': 0.5
        },
        tooltip=folium.GeoJsonTooltip(fields=['title'])
    ).add_to(m)

    # 3. Dessiner les NOEUDS (Markers)
    for _, row in gdf.iterrows():
        folium.CircleMarker(
            location=[row.centroid.y, row.centroid.x],
            radius=3, color="#2980B9", fill=True, weight=2
        ).add_to(m)

    # Capture du clic
    map_data = st_folium(m, width=750, height=550, key="main_map")
    
    if map_data.get("last_active_drawing"):
        clicked = map_data["last_active_drawing"]["properties"].get("title")
        if clicked and clicked != st.session_state.selected_zone:
            st.session_state.selected_zone = clicked
            st.rerun()

with col2:
    st.subheader("🤖 Rapport de l'Expert Agentic")
    if st.button("🚀 Générer l'Analyse pour " + st.session_state.selected_zone):
        with st.spinner("L'IA analyse le graphe et les rapports..."):
            rapport = famine_guard_brain(st.session_state.selected_zone, phase, prix_val, ndvi_val, langue)
            st.markdown(rapport)
    else:
        st.info("Cliquez sur une zone sur la carte, puis sur 'Générer' pour voir l'analyse RAG.")

st.caption("FamineGuard v1.2 | AIMS Senegal | Decision Support System")
