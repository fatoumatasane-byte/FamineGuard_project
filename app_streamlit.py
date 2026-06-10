import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MiniMap
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0f1117; color: #f0f0f0; }
    .block-container { padding: 1.5rem 2rem; }
    section[data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2e3146;
    }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    [data-testid="stMetric"] {
        background: #1e2130;
        border-radius: 10px;
        padding: 12px 16px;
        border-left: 4px solid #e74c3c;
    }
    .legend-box {
        background: #1e2130;
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 10px;
        font-size: 13px;
        color: #ccc;
        border: 1px solid #2e3146;
    }
    .legend-item { display: flex; align-items: center; margin: 6px 0; gap: 10px; }
    .legend-dot { width: 14px; height: 14px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
    h3 { color: #ffffff !important; font-weight: 700; font-size: 1.1rem; }
    </style>
""", unsafe_allow_html=True)


# --- 2. CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_resources():
    gdf = gpd.read_file('ipc_sen.geojson')
    if 'title' not in gdf.columns:
        gdf['title'] = gdf['ADM2_FR'] if 'ADM2_FR' in gdf.columns else gdf.index.astype(str)
    gdf['centroid'] = gdf.geometry.centroid

    v_store = None
    chroma_path = "mon_index_chroma"
    if os.path.exists(chroma_path):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_chroma import Chroma
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                encode_kwargs={"normalize_embeddings": True}
            )
            v_store = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
            count = v_store._collection.count()
            st.sidebar.success(f"✅ RAG chargé : {count} chunks indexés")
        except Exception as e:
            st.sidebar.warning(f"⚠️ RAG non disponible : {e}")
            v_store = None
    return gdf, v_store


gdf, vectorstore = load_resources()


# --- 3. CALCUL DU RISQUE PAR ZONE ---
def compute_zone_risk(prix_val, ndvi_val, zone_index):
    """Simule un risque par zone avec variation spatiale reproductible."""
    rng = np.random.default_rng(seed=int(zone_index) * 42)
    local_prix = prix_val * rng.uniform(0.7, 1.4)
    local_ndvi = ndvi_val * rng.uniform(0.6, 1.5)
    if local_prix > 2.5 or local_ndvi < 0.2:
        return 4
    elif local_prix > 1.8 or local_ndvi < 0.35:
        return 3
    elif local_prix > 1.3 or local_ndvi < 0.5:
        return 2
    else:
        return 1


PHASE_COLORS = {
    1: "#27AE60",
    2: "#F1C40F",
    3: "#E67E22",
    4: "#E74C3C",
    5: "#8E44AD",
}
PHASE_LABELS = {
    1: "Phase 1 — Minimal",
    2: "Phase 2 — Stress",
    3: "Phase 3 — Crise",
    4: "Phase 4 — Urgence",
    5: "Phase 5 — Famine",
}


# --- 4. INITIALISATION SESSION ---
if 'selected_zone' not in st.session_state:
    st.session_state.selected_zone = gdf['title'].iloc[0]


# --- 5. LOGIQUE AGENTIC RAG ---
def tool_search_rag(query):
    if vectorstore is None:
        return "Archives non disponibles."
    try:
        docs = vectorstore.similarity_search(query, k=3)
        return "\n\n".join(
            [f"[SOURCE: {d.metadata.get('source', 'PDF')}] {d.page_content}" for d in docs]
        )
    except Exception as e:
        return f"Erreur documentaire : {e}"


def famine_guard_agent(zone, phase, prix, ndvi, langue):
    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=st.secrets["GROQ_API_KEY"])
        rag_data = tool_search_rag(f"food security crisis {zone} Senegal history interventions")
        system_prompt = "Tu es un expert senior du PAM et de FEWS NET. Réponds avec précision et cite tes sources."
        user_prompt = f"""
LANGUE : {langue}
ZONE : {zone}
GNN PREDICTION : Phase {phase} (Prix x{prix:.2f}, NDVI {ndvi:.2f})
ARCHIVES PDF : {rag_data}

Produis un rapport structuré :
1. Analyse du Choc
2. Analogies Historiques
3. Recommandations prioritaires
        """
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Erreur de l'agent : {e}"


# --- 6. CALCUL GLOBAL (utilisé sidebar + col2) ---
all_phases = [compute_zone_risk(prix_val=1.8, ndvi_val=0.35, zone_index=i) for i in range(len(gdf))]
zone_phases = {gdf.iloc[i]['title']: all_phases[i] for i in range(len(gdf))}


# --- 7. SIDEBAR ---
with st.sidebar:
    st.markdown("## 🌾 FamineGuard AI")
    st.markdown("---")
    st.markdown(f"📍 **Zone sélectionnée**")
    st.markdown(f"### {st.session_state.selected_zone}")
    st.markdown("---")
    st.markdown("#### ⚙️ Paramètres de simulation")
    prix_val = st.slider("💰 Choc Prix (multiplicateur)", 1.0, 5.0, 1.8, step=0.1)
    ndvi_val = st.slider("🌿 Végétation NDVI", 0.1, 1.0, 0.35, step=0.05)
    langue = st.selectbox("🌐 Langue du rapport", ["Français", "English"])

    # Recalcul avec les sliders actuels
    all_phases = [compute_zone_risk(prix_val, ndvi_val, i) for i in range(len(gdf))]
    zone_phases = {gdf.iloc[i]['title']: all_phases[i] for i in range(len(gdf))}

    sel_idx = gdf[gdf['title'] == st.session_state.selected_zone].index
    global_phase = zone_phases.get(st.session_state.selected_zone, 1)
    phase_color = PHASE_COLORS[global_phase]

    st.markdown(f"""
    <div style="background:{phase_color}22; border-left:4px solid {phase_color};
         padding:12px; border-radius:8px; margin-top:8px;">
        <div style="font-size:11px; color:#aaa; text-transform:uppercase; letter-spacing:1px;">
            Risque zone sélectionnée
        </div>
        <div style="font-size:22px; font-weight:700; color:{phase_color}; margin-top:4px;">
            {PHASE_LABELS[global_phase]}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    n_critique = sum(1 for p in all_phases if p >= 4)
    n_crise    = sum(1 for p in all_phases if p == 3)
    n_stable   = sum(1 for p in all_phases if p <= 2)
    st.markdown("#### 📊 Aperçu national")
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Urgence", n_critique)
    c2.metric("🟠 Crise", n_crise)
    c3.metric("🟢 Stable", n_stable)

    st.markdown("""
    <div class="legend-box">
        <b>Légende IPC</b>
        <div class="legend-item"><span class="legend-dot" style="background:#27AE60"></span> Phase 1 — Minimal</div>
        <div class="legend-item"><span class="legend-dot" style="background:#F1C40F"></span> Phase 2 — Stress</div>
        <div class="legend-item"><span class="legend-dot" style="background:#E67E22"></span> Phase 3 — Crise</div>
        <div class="legend-item"><span class="legend-dot" style="background:#E74C3C"></span> Phase 4 — Urgence</div>
        <div class="legend-item"><span class="legend-dot" style="background:#8E44AD"></span> Phase 5 — Famine</div>
    </div>
    """, unsafe_allow_html=True)


# --- 8. CONTENU PRINCIPAL ---
st.markdown("## 🌍 FamineGuard — Système d'Alerte Précoce")
st.markdown("Ajustez les sliders pour voir les risques se propager sur la carte en **temps réel**.")

col1, col2 = st.columns([1.4, 1])

with col1:
    st.markdown("### 🗺️ Carte de Risque IPC — Graphe de Propagation")

    m = folium.Map(location=[14.5, -14.5], zoom_start=7, tiles="CartoDB dark_matter")

    # ── EDGES : lignes de propagation colorées selon risque max des deux zones ──
    drawn_edges = set()
    for i, row in gdf.iterrows():
        p1 = [row.centroid.y, row.centroid.x]
        neighbors = gdf[gdf.geometry.touches(row.geometry)]
        for j, neighbor in neighbors.iterrows():
            edge_key = tuple(sorted([i, j]))
            if edge_key in drawn_edges:
                continue
            drawn_edges.add(edge_key)
            p2 = [neighbor.centroid.y, neighbor.centroid.x]
            phase_i = zone_phases.get(row['title'], 1)
            phase_j = zone_phases.get(neighbor['title'], 1)
            max_phase = max(phase_i, phase_j)
            edge_color = PHASE_COLORS[max_phase]
            weight = 1.2 + max_phase * 0.9
            folium.PolyLine(
                [p1, p2],
                color=edge_color,
                weight=weight,
                opacity=0.75,
                tooltip=f"{row['title']} ↔ {neighbor['title']} | Phase max {max_phase}"
            ).add_to(m)

    # ── POLYGONES colorés par phase IPC ──
    def style_fn(feature):
        zone_name = feature['properties']['title']
        phase_i   = zone_phases.get(zone_name, 1)
        fill      = PHASE_COLORS[phase_i]
        selected  = (zone_name == st.session_state.selected_zone)
        return {
            'fillColor':   fill,
            'color':       '#ffffff' if selected else '#333333',
            'weight':      3 if selected else 0.8,
            'fillOpacity': 0.75 if selected else 0.55,
        }

    folium.GeoJson(
        gdf,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=['title'],
            aliases=['Zone :'],
            style="font-size:13px; font-weight:bold;"
        )
    ).add_to(m)

    # ── NOEUDS : taille et couleur proportionnelles au risque ──
    for i, row in gdf.iterrows():
        phase_i  = zone_phases.get(row['title'], 1)
        nc       = PHASE_COLORS[phase_i]
        selected = (row['title'] == st.session_state.selected_zone)

        folium.CircleMarker(
            location=[row.centroid.y, row.centroid.x],
            radius=5 + phase_i * 2,
            color='white',
            fill=True,
            fill_color=nc,
            fill_opacity=0.95,
            weight=2 if selected else 1,
            tooltip=f"<b>{row['title']}</b><br>{PHASE_LABELS[phase_i]}"
        ).add_to(m)

        # Halo sur la zone sélectionnée
        if selected:
            folium.CircleMarker(
                location=[row.centroid.y, row.centroid.x],
                radius=16, color=nc, fill=False, weight=2, opacity=0.5
            ).add_to(m)

    # MiniMap
    MiniMap(toggle_display=True, tile_layer="CartoDB dark_matter").add_to(m)

    # Légende inline dans la carte
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
         background:rgba(15,17,23,0.92); padding:12px 16px; border-radius:10px;
         border:1px solid #555; font-family:sans-serif; font-size:12px; color:white; line-height:1.8;">
        <b style="font-size:13px;">📊 Phases IPC</b><br>
        <span style="color:#27AE60">●</span> Phase 1 — Minimal &nbsp;
        <span style="color:#F1C40F">●</span> Phase 2 — Stress<br>
        <span style="color:#E67E22">●</span> Phase 3 — Crise &nbsp;&nbsp;
        <span style="color:#E74C3C">●</span> Phase 4 — Urgence<br>
        <span style="color:#8E44AD">●</span> Phase 5 — Famine<br>
        <hr style="border-color:#444; margin:6px 0;">
        <small>● Taille nœud ∝ intensité du risque<br>
        — Épaisseur arête ∝ propagation</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    map_data = st_folium(m, width=None, height=590, key="main_map")

    # Capture clic zone
    if map_data and map_data.get("last_object_clicked_tooltip"):
        raw = map_data["last_object_clicked_tooltip"]
        clicked = raw.split(":")[-1].strip() if ":" in raw else raw.strip()
        if clicked and clicked in gdf['title'].values and clicked != st.session_state.selected_zone:
            st.session_state.selected_zone = clicked
            st.rerun()


with col2:
    st.markdown("### 🤖 Analyse IA — Expert Agentic")

    sel_phase = zone_phases.get(st.session_state.selected_zone, 1)
    sel_color = PHASE_COLORS[sel_phase]

    st.markdown(f"""
    <div style="background:{sel_color}18; border:1px solid {sel_color}55;
         border-radius:10px; padding:14px 18px; margin-bottom:16px;">
        <div style="font-size:12px; color:#aaa; margin-bottom:4px;">Zone analysée</div>
        <div style="font-size:18px; font-weight:700; color:white;">📍 {st.session_state.selected_zone}</div>
        <div style="margin-top:8px;">
            <span style="background:{sel_color}; color:white; padding:3px 10px;
                  border-radius:12px; font-size:12px; font-weight:600;">
                {PHASE_LABELS[sel_phase]}
            </span>
        </div>
        <div style="margin-top:10px; font-size:12px; color:#bbb; line-height:1.6;">
            💰 Choc prix : <b>×{prix_val:.1f}</b> &nbsp;|&nbsp; 🌿 NDVI : <b>{ndvi_val:.2f}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Voisins à risque élevé
    sel_row = gdf[gdf['title'] == st.session_state.selected_zone]
    if not sel_row.empty:
        neighbors_sel = gdf[gdf.geometry.touches(sel_row.geometry.iloc[0])]
        high_risk = [n for _, n in neighbors_sel.iterrows() if zone_phases.get(n['title'], 1) >= 3]
        if high_risk:
            st.markdown("**⚠️ Zones voisines à risque élevé :**")
            for n in high_risk:
                nc = PHASE_COLORS[zone_phases[n['title']]]
                st.markdown(
                    f"<span style='color:{nc}; font-weight:600;'>● {n['title']}</span> — {PHASE_LABELS[zone_phases[n['title']]]}",
                    unsafe_allow_html=True
                )
            st.markdown("")

    if st.button(f"🚀 Générer le rapport pour {st.session_state.selected_zone}", use_container_width=True):
        with st.spinner("L'IA analyse le graphe et les archives..."):
            rapport = famine_guard_agent(
                st.session_state.selected_zone, sel_phase, prix_val, ndvi_val, langue
            )
            st.markdown(rapport)
    else:
        st.info("👆 Cliquez sur une zone, puis générez l'analyse pour le rapport RAG complet.")

    st.markdown("---")
    st.markdown("**📈 Distribution nationale des risques**")
    phase_counts = pd.Series(all_phases).value_counts().sort_index()
    chart_data = pd.DataFrame({
        'Phase': [PHASE_LABELS[p] for p in phase_counts.index],
        'Zones': phase_counts.values
    })
    st.bar_chart(chart_data.set_index('Phase'), use_container_width=True, height=200)


st.caption("FamineGuard v2.0 | AIMS Senegal | Decision Support System")
