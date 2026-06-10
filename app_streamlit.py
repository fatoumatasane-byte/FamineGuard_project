import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import networkx as nx

warnings.filterwarnings('ignore')

# --- 1. CORRECTIF SQLITE3 POUR STREAMLIT CLOUD ---
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
    .main { background-color: #0E1117; color: white; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌾 FamineGuard: Spatiotemporal GNN & Agentic RAG")
st.markdown("### AIMS Senegal - SDGs Innovation Challenge 2026")

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
        df = pd.DataFrame({'zone_display': ["Dakar", "Bakel", "Matam", "Podor", "Saint-Louis", "Tambacounda", "Kaolack", "Ziguinchor"]})

    # Carte GeoJSON
    s_map = None
    if os.path.exists('ipc_sen.geojson'):
        s_map = gpd.read_file('ipc_sen.geojson')
        # Normalisation des colonnes de nom
        if 'title' not in s_map.columns:
            for c in s_map.columns:
                if c.lower() in ['name', 'reg', 'admin', 'title', 'nom']:
                    s_map = s_map.rename(columns={c: 'title'})
                    break
    
    # --- CHARGEMENT DU VECTORSTORE RAG ---
    v_store = None
    chroma_path = "mon_index_chroma"
    if os.path.exists(chroma_path):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_community.vectorstores import Chroma
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            v_store = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
        except Exception as e:
            v_store = None
    
    return df, s_map, v_store

nodes_df, senegal_map, vectorstore = load_resources()

# --- 3. CERVEAU AGENTIC (GROQ) ---
def famine_guard_brain(zone, phase, prix, ndvi, langue):
    try:
        from openai import OpenAI
        # Récupération de la clé API depuis les secrets ou l'environnement
        api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)

        # Recherche RAG
        context_docs = ""
        if vectorstore:
            docs = vectorstore.similarity_search(f"food security crisis in {zone} Senegal", k=2)
            context_docs = "\n\n".join([d.page_content for d in docs])

        lang_inst = "RÉPONDS EN FRANÇAIS." if langue == "French" else "RESPOND IN ENGLISH."
        
        prompt = f"""
        {lang_inst}
        Tu es un expert senior du PAM et de FEWS NET. 
        ANALYSE GNN : La zone {zone} est en Phase {phase} (Prix x{prix}, NDVI {ndvi}).
        CONTEXTE ARCHIVES : {context_docs if context_docs else 'Pas d archives spécifiques.'}
        
        Produis un rapport structuré :
        1. Analyse du risque (facteurs prix/climat)
        2. Recommandations prioritaires pour les ONGs
        3. Note sur la propagation vers les zones voisines.
        """

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Expert en sécurité alimentaire au Sahel."},
                      {"role": "user", "content": prompt}],
            temperature=0.2
        )
        return completion.choices[0].message.content, "✅ RAG Actif" if context_docs else "⚠️ Mode Base de Connaissances"
    except Exception as e:
        return f"Erreur : {e}", "❌ Erreur API"

# --- 4. INTERFACE SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Simulation GNN")
    villes = sorted(nodes_df['zone_display'].unique())
    zone_choisie = st.selectbox("Cible de l'analyse", villes)
    langue_choisie = st.radio("Langue", ["French", "English"])
    st.markdown("---")
    prix_val = st.slider("Choc Prix (x)", 1.0, 5.0, 1.5)
    ndvi_val = st.slider("Indice Végétation (NDVI)", 0.1, 1.0, 0.4)
    run = st.button("🚀 LANCER L'ANALYSE AGENTIQUE", use_container_width=True)

# --- 5. LOGIQUE D'AFFICHAGE ---
col1, col2 = st.columns([1, 1.2])

if run:
    # Calcul de la Phase (Logique GNN simplifiée)
    if prix_val > 2.8 or ndvi_val < 0.2: ph = 4
    elif prix_val > 1.9 or ndvi_val < 0.3: ph = 3
    else: ph = 2

    with col1:
        st.subheader("📍 Visualisation du Graphe GNN")
        
        # -- DESSIN DE LA CARTE + GRAPHE --
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')

        if senegal_map is not None:
            # Nettoyage des noms pour le match
            senegal_map['clean_title'] = senegal_map['title'].str.lower().str.strip()
            target_clean = zone_choisie.lower().strip()
            
            # 1. Dessin des arêtes (Réseau de voisinage)
            centroids = senegal_map.geometry.centroid
            for i, row in senegal_map.iterrows():
                geom = row.geometry
                p1 = centroids.iloc[i]
                # On connecte les voisins (ceux qui partagent une bordure)
                neighbors = senegal_map[senegal_map.geometry.touches(geom)]
                for j, neighbor in neighbors.iterrows():
                    p2 = centroids.iloc[j]
                    ax.plot([p1.x, p2.x], [p1.y, p2.y], color='#4b6584', linestyle='--', linewidth=0.6, alpha=0.4, zorder=1)

            # 2. Dessin des zones (Polygones)
            color_map = []
            for idx, row in senegal_map.iterrows():
                if target_clean in row['clean_title']:
                    color_map.append("#E74C3C" if ph >= 4 else "#E67E22" if ph == 3 else "#F1C40F")
                else:
                    color_map.append("#2d3436") # Zones neutres
            
            senegal_map.plot(color=color_map, edgecolor="#636e72", linewidth=0.5, ax=ax, alpha=0.7, zorder=2)

            # 3. Dessin des nœuds (Villes)
            ax.scatter(centroids.x, centroids.y, color='#00cec9', s=25, edgecolors='white', linewidth=0.5, zorder=3)
            
            # Focus sur la zone sélectionnée
            target_geo = senegal_map[senegal_map['clean_title'].str.contains(target_clean)]
            if not target_geo.empty:
                c = target_geo.geometry.centroid.iloc[0]
                ax.scatter(c.x, c.y, color='white', s=120, marker='*', zorder=4)

        ax.set_axis_off()
        st.pyplot(fig)
        
        # Métriques
        st.metric("Niveau de Risque Prédit", f"Phase IPC {ph}", delta="ALERTE" if ph >= 3 else "STABLE")

    with col2:
        st.subheader("🤖 Rapport de l'Agent Expert")
        rapport, badge = famine_guard_brain(zone_choisie, ph, prix_val, ndvi_val, langue_choisie)
        st.caption(badge)
        st.markdown(rapport)

else:
    with col1:
        st.info("👈 Sélectionnez une zone et configurez les paramètres dans la barre latérale.")
        # Affichage d'une carte vide par défaut
        if senegal_map is not None:
            fig, ax = plt.subplots()
            fig.patch.set_facecolor('#0E1117')
            senegal_map.plot(color="#2d3436", edgecolor="#636e72", ax=ax)
            ax.set_axis_off()
            st.pyplot(fig)
    with col2:
        st.markdown("""
        ### Bienvenue sur FamineGuard
        Cette interface simule l'interaction entre notre modèle **GNN Spatiotemporel** et notre **Agent RAG**.
        
        - **À gauche** : Visualisation du graphe de propagation (les lignes représentent les dépendances spatiales entre régions).
        - **À droite** : Rapport généré par l'IA fusionnant les données de simulation et les archives humanitaires.
        """)
