import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI

warnings.filterwarnings('ignore')

# --- 1. CORRECTIF SQLITE3 POUR LE CLOUD ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# --- CONFIGURATION INTERFACE ---
st.set_page_config(page_title="FamineGuard Interactive", layout="wide", page_icon="🌾")

# Custom CSS pour un look "Dark Tech"
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { border: 1px solid #30363d; padding: 10px; border_radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CHARGEMENT DES DONNÉES ---
@st.cache_resource
def load_data():
    # Carte GeoJSON
    gdf = gpd.read_file('ipc_sen.geojson')
    # On s'assure d'avoir une colonne 'title' propre
    if 'title' not in gdf.columns:
        gdf['title'] = gdf['ADM2_FR'] if 'ADM2_FR' in gdf.columns else gdf.index.astype(str)
    
    # RAG - Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # RAG - Vectorstore
    v_store = None
    if os.path.exists("mon_index_chroma"):
        try:
            v_store = Chroma(persist_directory="mon_index_chroma", embedding_function=embeddings)
        except:
            v_store = None
    return gdf, v_store

gdf, vectorstore = load_data()

# --- 3. SESSION STATE POUR LE CLIC ---
if 'selected_zone' not in st.session_state:
    st.session_state.selected_zone = "Dakar"

# --- 4. LOGIQUE DE CALCUL DU RISQUE (GNN Simulé) ---
def get_risk_level(prix, ndvi):
    if prix > 2.8 or ndvi < 0.2: return 4, "🔴 CRITIQUE", "#E74C3C"
    if prix > 1.9 or ndvi < 0.35: return 3, "🟠 ALERTE", "#E67E22"
    return 2, "🟢 STABLE", "#2ECC71"

# --- 5. INTERFACE SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Simulation")
    st.info(f"📍 Zone sélectionnée : **{st.session_state.selected_zone}**")
    
    prix_val = st.slider("Choc Prix (multiplicateur)", 1.0, 5.0, 1.5)
    ndvi_val = st.slider("Indice Végétation (NDVI)", 0.1, 1.0, 0.4)
    langue = st.radio("Langue", ["Français", "English"])
    
    phase, label, color_hex = get_risk_level(prix_val, ndvi_val)
    
    st.markdown("---")
    st.metric("Risque GNN", f"Phase {phase}", delta=label)

# --- 6. CARTE INTERACTIVE (FOLIUM) ---
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("🗺️ Graphe Spatio-Temporel (Cliquez sur une zone)")
    
    # Création de la carte
    m = folium.Map(location=[14.5, -14.5], zoom_start=7, tiles="CartoDB dark_matter")

    # On ajoute le GeoJSON avec des événements de clic
    geojson = folium.GeoJson(
        gdf,
        style_function=lambda feature: {
            'fillColor': color_hex if feature['properties']['title'] == st.session_state.selected_zone else '#2c3e50',
            'color': 'white',
            'weight': 1,
            'fillOpacity': 0.7,
        },
        tooltip=folium.GeoJsonTooltip(fields=['title'], aliases=['Zone: '])
    ).add_to(m)

    # Affichage et capture du clic
    map_output = st_folium(m, width=700, height=500, key="senegal_map")

    # Si l'utilisateur clique sur une région, on met à jour la zone sélectionnée
    if map_output.get("last_active_drawing"):
        clicked_zone = map_output["last_active_drawing"]["properties"].get("title")
        if clicked_zone and clicked_zone != st.session_state.selected_zone:
            st.session_state.selected_zone = clicked_zone
            st.rerun()

with col2:
    st.subheader("🤖 Rapport de l'Agent Expert")
    
    if st.button("🚀 Générer l'analyse pour " + st.session_state.selected_zone):
        with st.spinner("L'expert consulte le GNN et les archives..."):
            try:
                # 1. Recherche RAG (Correctif du bug 'int')
                context = "Aucune archive trouvée."
                if vectorstore is not None:
                    # On force le type string pour la recherche
                    query_text = str(f"food security crisis {st.session_state.selected_zone} Senegal recommendations")
                    docs = vectorstore.similarity_search(query_text, k=2)
                    if docs:
                        context = "\n\n".join([doc.page_content for doc in docs])

                # 2. Appel Groq
                client = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=st.secrets["GROQ_API_KEY"]
                )
                
                prompt = f"""
                Réponds en {langue}. 
                Tu es un expert FEWS NET. 
                ZONE : {st.session_state.selected_zone}
                RISQUE GNN : Phase {phase} (Prix x{prix_val}, NDVI {ndvi_val})
                ARCHIVES : {context}
                
                Rédige un rapport bref : Analyse du choc et 3 recommandations prioritaires.
                """
                
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                )
                
                st.markdown(f"### 📋 Rapport : {st.session_state.selected_zone}")
                st.write(chat_completion.choices[0].message.content)
                
            except Exception as e:
                st.error(f"Erreur API : {str(e)}")

# --- BAS DE PAGE ---
st.markdown("---")
st.caption("FamineGuard Project | AIMS Senegal | SDGs Challenge 2026")
