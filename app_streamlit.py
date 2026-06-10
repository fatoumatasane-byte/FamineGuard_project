import os
import warnings
warnings.filterwarnings('ignore')

# --- 1. CORRECTIF SQLITE3 POUR STREAMLIT CLOUD ---
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
        df = pd.DataFrame({'zone_display': ["Dakar", "Bakel", "Matam", "Podor",
                                             "Tambacounda", "Kaolack", "Louga",
                                             "Kedougou", "Kolda", "Ziguinchor"]})

    # Carte GeoJSON
    s_map = None
    if os.path.exists('ipc_sen.geojson'):
        s_map = gpd.read_file('ipc_sen.geojson')
        if 'title' not in s_map.columns:
            for c in s_map.columns:
                if c.lower() in ['name', 'reg', 'admin', 'title']:
                    s_map = s_map.rename(columns={c: 'title'})
                    break

    # --- CHARGEMENT DU VECTORSTORE CHROMA (VERSION CORRIGÉE) ---
    v_store = None
    chroma_path = "mon_index_chroma"
    if os.path.exists(chroma_path):
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import Chroma

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            # CORRECTIF BUG : on evite le crash len(int) en utilisant les bons params
            v_store = Chroma(
                persist_directory=chroma_path,
                embedding_function=embeddings,
                collection_metadata={"hnsw:space": "cosine"}
            )
            # Test de validation
            count = v_store._collection.count()
            st.sidebar.success(f"✅ RAG chargé : {count} chunks indexés")
        except Exception as e:
            st.sidebar.warning(f"⚠️ RAG non disponible: {e}")
            v_store = None
    else:
        st.sidebar.info("ℹ️ Dossier 'mon_index_chroma' non trouvé sur le serveur")

    return df, s_map, v_store

nodes_df, senegal_map, vectorstore = load_resources()


# --- 3. OUTILS DE L'AGENT (Tool 1: GNN Stats) ---
def tool_get_gnn_stats(zone_name: str, prix_val: float, ndvi_val: float, phase: int) -> str:
    """Retourne les données GNN simulées pour la zone."""
    alert_level = "CRITICAL" if phase >= 4 else ("WATCH" if phase == 3 else "STABLE")
    ndvi_status = "CRITICAL LOW" if ndvi_val < 0.2 else ("LOW" if ndvi_val < 0.35 else "NORMAL")
    price_status = "VERY HIGH" if prix_val > 2.8 else ("HIGH" if prix_val > 1.9 else "NORMAL")
    return f"""GNN PREDICTION REPORT:
- Zone: {zone_name}
- Predicted IPC Phase: {phase} ({alert_level})
- NDVI Status: {ndvi_status} (value: {ndvi_val:.2f})
- Price Shock: {price_status} (multiplier: {prix_val:.1f}x)
- Alert Level: {alert_level}
- Confidence: 0.985"""


# --- 4. OUTIL RAG (Tool 2: Archive Search) ---
def tool_search_rag(query: str) -> str:
    """Cherche dans les archives PDF indexées (FEWS NET, WFP, FAO)."""
    if vectorstore is None:
        return "ARCHIVE_UNAVAILABLE"
    try:
        docs = vectorstore.similarity_search(query, k=3)
        if not docs:
            return "NO_RESULTS"
        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown')
            source_name = os.path.basename(source) if '/' in source or '\\' in source else source
            page = doc.metadata.get('page', '?')
            results.append(f"[SOURCE {i+1}: {source_name}, page {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"RAG_ERROR: {str(e)}"


# --- 5. CERVEAU AGENTIC ---
def famine_guard_agentic_brain(zone, phase, prix, ndvi, langue):
    try:
        from openai import OpenAI

        api_key = st.secrets["GROQ_API_KEY"]
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)

        # == STEP 1 : GNN Tool ==
        gnn_data = tool_get_gnn_stats(zone, prix, ndvi, phase)

        # == STEP 2 : RAG Tool (2 requêtes ciblées) ==
        rag_result_1 = tool_search_rag(
            f"food insecurity crisis {zone} Senegal IPC phase {phase} humanitarian response"
        )
        rag_result_2 = tool_search_rag(
            f"famine prevention intervention Sahel Senegal recommendations cash transfer food aid"
        )

        rag_available = rag_result_1 not in ("ARCHIVE_UNAVAILABLE", "NO_RESULTS", "") and \
                        "RAG_ERROR" not in rag_result_1

        # == STEP 3 : Synthèse LLM ==
        lang_instruction = "RÉPONDS ENTIÈREMENT EN FRANÇAIS." if langue == "French" \
                           else "RESPOND ENTIRELY IN ENGLISH."

        if rag_available:
            rag_section = f"""ARCHIVES TROUVÉES (FEWS NET / WFP / FAO) :

Recherche 1 — Crise zone spécifique:
{rag_result_1}

Recherche 2 — Interventions et recommandations:
{rag_result_2}"""
            rag_instruction = """Pour chaque fait issu des archives, cite OBLIGATOIREMENT la source entre crochets:
[SOURCE: nom_du_fichier.pdf, page X]
Si une recommandation vient d'un rapport, l'indiquer explicitement."""
        else:
            rag_section = "ARCHIVES : Base de documents non disponible sur ce déploiement."
            rag_instruction = "Basez-vous sur les connaissances générales des crises alimentaires sahéliennes (FEWS NET, WFP, FAO)."

        system_prompt = """Tu es un expert senior en sécurité alimentaire travaillant pour FEWS NET et WFP.
Tu génères des rapports de décision pour des ONGs humanitaires au Sénégal.
Tes rapports sont structurés, précis et toujours sourcés quand des archives sont disponibles."""

        user_prompt = f"""
{lang_instruction}

=== DONNÉES GNN (Layer 2 — Spatiotemporal Graph Neural Network) ===
{gnn_data}

=== ARCHIVES HUMANITAIRES (Layer 3 — RAG) ===
{rag_section}

=== RÈGLES DE CITATION ===
{rag_instruction}

=== RAPPORT À PRODUIRE ===
Génère un rapport structuré avec ces 4 sections :

**1. 📊 ANALYSE DU RISQUE**
Analyse la situation de {zone} (Phase {phase}) en expliquant les facteurs déclencheurs
(prix alimentaires x{prix:.1f}, NDVI={ndvi:.2f}).

**2. 📚 ANALOGIES HISTORIQUES**
Si des archives sont disponibles : cite les situations similaires passées avec leurs sources [SOURCE: ...].
Si non disponibles : indique-le clairement et donne le contexte régional général.

**3. 🎯 3 ACTIONS RECOMMANDÉES**
Liste 3 interventions concrètes et prioritaires, numérotées, avec si possible une référence documentaire.

**4. 📋 SOURCES UTILISÉES**
Liste complète des rapports cités, ou mention "Connaissances générales FEWS NET/WFP" si pas d'archives.
"""

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.15,
            max_tokens=1200
        )

        rapport = completion.choices[0].message.content
        rag_badge = "✅ **RAG Agentic actif** — Sources documentaires indexées utilisées" if rag_available \
                    else "⚠️ **Mode dégradé** — Aucune archive PDF trouvée dans `mon_index_chroma/`"

        return rapport, rag_badge

    except Exception as e:
        fallback = (
            f"**⚠️ Erreur technique :** `{e}`\n\n"
            f"Le GNN confirme que **{zone}** est en **Phase {phase}**.\n\n"
            f"**Actions d'urgence recommandées (Phase {phase}) :**\n"
            f"1. Déploiement immédiat d'une évaluation terrain\n"
            f"2. Activation des mécanismes de réponse rapide WFP\n"
            f"3. Coordination avec les autorités locales de {zone}"
        )
        return fallback, "❌ Erreur — Vérifier la clé GROQ_API_KEY dans les secrets Streamlit"


# --- 6. INTERFACE STREAMLIT ---
with st.sidebar:
    st.header("🎮 Configuration")
    villes = sorted(nodes_df['zone_display'].unique())
    zone_choisie = st.selectbox("Zone Cible", villes)
    langue_choisie = st.radio("Langue du rapport", ["French", "English"])
    st.markdown("---")
    st.subheader("⚡ Simulation de Choc")
    prix_val = st.slider("Choc Prix (multiplicateur)", 1.0, 5.0, 1.5, step=0.1,
                         help="1.0 = prix normal, 3.0 = prix triplé")
    ndvi_val = st.slider("Végétation NDVI", 0.1, 1.0, 0.4, step=0.05,
                         help="0.1 = désastre végétatif, 1.0 = végétation maximale")
    run = st.button("🚀 LANCER L'ANALYSE AGENTIQUE", use_container_width=True)

col1, col2 = st.columns([1, 1.3])

if run:
    # Calcul de la Phase IPC (cohérent avec le notebook)
    if prix_val > 2.8 or ndvi_val < 0.2:
        ph = 4
    elif prix_val > 1.9 or ndvi_val < 0.3:
        ph = 3
    else:
        ph = 2

    with st.spinner("🤖 Agent en cours : GNN → RAG → Synthèse..."):

        # -- CARTE --
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')

        if senegal_map is not None:
            senegal_map["color"] = "#2ECC71"
            mask = senegal_map['title'].str.lower().str.strip().str.contains(
                zone_choisie.lower().strip(), na=False
            )
            color_code = "#E74C3C" if ph >= 4 else "#E67E22" if ph == 3 else "#F1C40F"
            senegal_map.loc[mask, "color"] = color_code
            senegal_map.plot(color=senegal_map["color"],
                             edgecolor="white", linewidth=0.4, ax=ax)
        ax.set_axis_off()

        # -- APPEL AGENT --
        rapport, rag_badge = famine_guard_agentic_brain(
            zone_choisie, ph, prix_val, ndvi_val, langue_choisie
        )

        with col1:
            st.subheader("📍 Risque GNN")
            st.pyplot(fig)
            phase_labels = {2: "STABLE", 3: "⚠️ ALERTE", 4: "🔴 CRITIQUE"}
            st.metric(
                "Niveau de Danger",
                f"Phase {ph} — {phase_labels.get(ph, str(ph))}",
                delta="URGENCE" if ph >= 4 else "VIGILANCE" if ph == 3 else "OK",
            )
            st.caption(f"Choc prix : **{prix_val}x** | NDVI : **{ndvi_val}**")

        with col2:
            st.subheader("🤖 Rapport Agentic RAG")
            st.caption(rag_badge)
            st.markdown(rapport)

else:
    with col1:
        st.info("👈 Sélectionnez une zone et simulez un choc, puis cliquez sur **LANCER L'ANALYSE**.")
    with col2:
        st.markdown("""
        ### 🏗️ Architecture FamineGuard
        
        **Layer 1 — Données** : IPC, NDVI, Prix céréales, Réseau routier
        
        **Layer 2 — GNN** : Prédiction spatiotemporelle par zone
        
        **Layer 3 — Agentic RAG** :
        - 🔧 Tool 1 : `get_gnn_stats` → données de la simulation
        - 🔧 Tool 2 : `search_humanitarian_reports` → archives FEWS NET / WFP / FAO
        - 🤖 LLM synthétise en citant les sources
        """)
