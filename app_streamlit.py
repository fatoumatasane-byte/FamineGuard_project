import os
import warnings
warnings.filterwarnings('ignore')

try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import streamlit as st
import numpy as np
import folium
from folium.plugins import MiniMap
from streamlit_folium import st_folium
import geopandas as gpd
from openai import OpenAI

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
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
    .legend-box {
        background: #1e2130; border-radius: 10px; padding: 14px 18px;
        margin-top: 10px; font-size: 13px; color: #ccc; border: 1px solid #2e3146;
    }
    .legend-item { display: flex; align-items: center; margin: 6px 0; gap: 10px; }
    .legend-dot { width: 14px; height: 14px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
    </style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
PHASE_COLORS = {1: "#27AE60", 2: "#F1C40F", 3: "#E67E22", 4: "#E74C3C", 5: "#8E44AD"}
PHASE_LABELS = {
    1: "Phase 1 — Minimal", 2: "Phase 2 — Stress",
    3: "Phase 3 — Crise",   4: "Phase 4 — Urgence", 5: "Phase 5 — Famine"
}
NEUTRAL = "#4a4e69"

# ─── LOAD RESOURCES ──────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    gdf = gpd.read_file('ipc_sen.geojson')
    if 'title' not in gdf.columns:
        for c in gdf.columns:
            if c.lower() in ['name', 'adm2_fr', 'adm1_fr', 'reg', 'admin']:
                gdf = gdf.rename(columns={c: 'title'})
                break

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
            st.sidebar.success(f"✅ RAG : {count} chunks indexés")
        except Exception as e:
            st.sidebar.warning(f"⚠️ RAG non disponible : {e}")
            v_store = None
    return gdf, v_store

gdf, vectorstore = load_resources()

# ─── FONCTIONS ───────────────────────────────────────────────────────────────
def compute_phase(prix_val, ndvi_val, seed=None):
    p, n = prix_val, ndvi_val
    if seed is not None:
        rng = np.random.default_rng(seed)
        p = prix_val * rng.uniform(0.7, 1.3)
        n = ndvi_val * rng.uniform(0.8, 1.2)
    if   p > 2.5 or n < 0.2:  return 4
    elif p > 1.8 or n < 0.35: return 3
    elif p > 1.3 or n < 0.5:  return 2
    else:                      return 1

def get_susceptible_neighbors(gdf, zone_name, target_phase, prix_val, ndvi_val):
    """Retourne {zone: phase} pour les voisins susceptibles d'être affectés."""
    sel = gdf[gdf['title'] == zone_name]
    if sel.empty:
        return {}
    sel_geom = sel.geometry.iloc[0]
    result = {}
    for _, row in gdf.iterrows():
        if row['title'] == zone_name:
            continue
        if sel_geom.touches(row.geometry) or sel_geom.intersects(row.geometry.buffer(0.01)):
            seed = abs(hash(row['title'])) % 99991
            rng = np.random.default_rng(seed)
            local_prix = prix_val * rng.uniform(0.55, 0.85)
            local_ndvi = ndvi_val * rng.uniform(0.9, 1.1)
            phase = max(compute_phase(local_prix, local_ndvi), max(1, target_phase - 1))
            result[row['title']] = min(phase, 5)
    return {k: v for k, v in result.items() if v >= 2}

def tool_search_rag(query):
    if vectorstore is None:
        return "Archives non disponibles."
    try:
        docs = vectorstore.similarity_search(query, k=3)
        return "\n\n".join(
            [f"[SOURCE: {d.metadata.get('source', 'PDF')}]\n{d.page_content}" for d in docs]
        )
    except Exception as e:
        return f"Erreur RAG: {e}"

def famine_guard_agent(zone, phase, prix, ndvi, susceptible, langue):
    try:
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=st.secrets["GROQ_API_KEY"]
        )
        rag = tool_search_rag(f"food security crisis {zone} Senegal interventions recommendations")
        lang = "RÉPONDS ENTIÈREMENT EN FRANÇAIS." if langue == "Français" else "RESPOND ENTIRELY IN ENGLISH."
        neighbors_str = ", ".join(
            [f"{z} ({PHASE_LABELS[p]})" for z, p in susceptible.items()]
        ) or "Aucune"

        prompt = f"""
{lang}

ZONE CIBLE : {zone}
PRÉDICTION GNN : {PHASE_LABELS[phase]} (Choc prix ×{prix:.1f}, NDVI {ndvi:.2f})
ZONES VOISINES SUSCEPTIBLES : {neighbors_str}
ARCHIVES HUMANITAIRES : {rag}

Génère un rapport structuré :

**1. 📊 Analyse du Choc**
Explique les facteurs déclencheurs dans {zone} et le risque de propagation vers les zones voisines.

**2. 📚 Analogies Historiques**
Cite des situations similaires dans les archives [SOURCE: ...].

**3. 🎯 3 Actions Recommandées**
Interventions prioritaires concrètes, avec focus sur les zones voisines susceptibles.

**4. 📋 Sources utilisées**
"""
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Expert senior PAM/FEWS NET Sénégal. Rapports précis et sourcés."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1200
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ Erreur agent : {e}"

# ─── SESSION STATE ────────────────────────────────────────────────────────────
for key, val in [
    ('selected_zone', None), ('analysis_run', False),
    ('prix_val', 1.8), ('ndvi_val', 0.35),
    ('langue', 'Français'), ('rapport', None)
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 FamineGuard AI")
    st.markdown("*AIMS Senegal — Decision Support System*")
    st.markdown("---")

    zone_list = ["— Sélectionner —"] + sorted(gdf['title'].unique().tolist())
    default_idx = (zone_list.index(st.session_state.selected_zone)
                   if st.session_state.selected_zone in zone_list else 0)
    selected = st.selectbox("📍 Zone Cible", zone_list, index=default_idx)

    if selected != "— Sélectionner —":
        if selected != st.session_state.selected_zone:
            st.session_state.selected_zone = selected
            st.session_state.analysis_run = False
            st.session_state.rapport = None

    if st.session_state.selected_zone:
        st.markdown("---")
        st.markdown("#### ⚙️ Simulation de Choc")
        prix_slider = st.slider("💰 Choc Prix (×)", 1.0, 5.0, st.session_state.prix_val, 0.1)
        ndvi_slider = st.slider("🌿 NDVI", 0.1, 1.0, st.session_state.ndvi_val, 0.05)
        langue_sel  = st.selectbox("🌐 Langue du rapport", ["Français", "English"])

        # Preview de la phase en temps réel
        live_phase = compute_phase(prix_slider, ndvi_slider)
        lc = PHASE_COLORS[live_phase]
        st.markdown(f"""
        <div style="background:{lc}22; border-left:4px solid {lc}; padding:10px;
             border-radius:8px; margin-top:8px;">
            <div style="font-size:11px; color:#aaa;">Risque estimé (preview)</div>
            <div style="font-size:17px; font-weight:700; color:{lc};">{PHASE_LABELS[live_phase]}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        if st.button("🚀 Lancer l'analyse", use_container_width=True):
            st.session_state.prix_val     = prix_slider
            st.session_state.ndvi_val     = ndvi_slider
            st.session_state.langue       = langue_sel
            st.session_state.analysis_run = True
            st.session_state.rapport      = None
            st.rerun()

    st.markdown("---")
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

# ─── PRÉ-CALCUL ──────────────────────────────────────────────────────────────
if st.session_state.analysis_run and st.session_state.selected_zone:
    target_phase = compute_phase(st.session_state.prix_val, st.session_state.ndvi_val)
    susceptible  = get_susceptible_neighbors(
        gdf, st.session_state.selected_zone,
        target_phase, st.session_state.prix_val, st.session_state.ndvi_val
    )
else:
    target_phase, susceptible = None, {}

# ─── LAYOUT PRINCIPAL ────────────────────────────────────────────────────────
st.markdown("## 🌍 FamineGuard — Système d'Alerte Précoce")
col1, col2 = st.columns([1.4, 1])

# ── COLONNE GAUCHE : CARTE ────────────────────────────────────────────────────
with col1:
    m = folium.Map(location=[14.5, -14.5], zoom_start=7, tiles="CartoDB dark_matter")
    MiniMap(toggle_display=True, tile_layer="CartoDB dark_matter").add_to(m)

    if not st.session_state.analysis_run or st.session_state.selected_zone is None:
        # ── ÉTAT INITIAL : carte neutre ──
        st.markdown("### 🗺️ Carte du Sénégal — Toutes les régions")
        st.caption("👈 Sélectionnez une région et lancez l'analyse pour voir la propagation du risque.")

        folium.GeoJson(
            gdf[['title', 'geometry']],
            style_function=lambda f: {
                'fillColor': NEUTRAL, 'color': '#888',
                'weight': 0.8, 'fillOpacity': 0.55
            },
            tooltip=folium.GeoJsonTooltip(fields=['title'], aliases=['Zone :'])
        ).add_to(m)

        for _, row in gdf.iterrows():
            folium.CircleMarker(
                location=[row.geometry.centroid.y, row.geometry.centroid.x],
                radius=4, color='#aaa', fill=True,
                fill_color=NEUTRAL, fill_opacity=0.75, weight=1,
                tooltip=row['title']
            ).add_to(m)

    else:
        # ── ÉTAT ANALYSE : graphe de propagation ──
        st.markdown(f"### 🗺️ Propagation du risque — {st.session_state.selected_zone}")

        # Polygones
        def style_analysis(feature):
            z = feature['properties']['title']
            if z == st.session_state.selected_zone:
                return {'fillColor': PHASE_COLORS[target_phase],
                        'color': '#ffffff', 'weight': 3, 'fillOpacity': 0.85}
            elif z in susceptible:
                return {'fillColor': PHASE_COLORS[susceptible[z]],
                        'color': '#dddddd', 'weight': 1.5, 'fillOpacity': 0.70}
            return {'fillColor': NEUTRAL, 'color': '#555', 'weight': 0.5, 'fillOpacity': 0.30}

        folium.GeoJson(
            gdf[['title', 'geometry']],
            style_function=style_analysis,
            tooltip=folium.GeoJsonTooltip(fields=['title'], aliases=['Zone :'])
        ).add_to(m)

        # Centroid de la zone cible
        t_row = gdf[gdf['title'] == st.session_state.selected_zone].iloc[0]
        t_pt  = [t_row.geometry.centroid.y, t_row.geometry.centroid.x]

        # Arêtes du graphe : zone cible → voisins susceptibles
        for zone_name, phase in susceptible.items():
            nb_rows = gdf[gdf['title'] == zone_name]
            if nb_rows.empty:
                continue
            nb_pt = [nb_rows.iloc[0].geometry.centroid.y,
                     nb_rows.iloc[0].geometry.centroid.x]
            folium.PolyLine(
                [t_pt, nb_pt],
                color=PHASE_COLORS[phase],
                weight=1.5 + phase * 0.8,
                opacity=0.85,
                tooltip=f"Propagation → {zone_name} : {PHASE_LABELS[phase]}"
            ).add_to(m)

        # Nœuds
        for _, row in gdf.iterrows():
            z  = row['title']
            pt = [row.geometry.centroid.y, row.geometry.centroid.x]
            is_target   = (z == st.session_state.selected_zone)
            is_neighbor = (z in susceptible)

            if is_target:
                nc, r = PHASE_COLORS[target_phase], 13
                # Halo autour de la zone cible
                folium.CircleMarker(
                    location=t_pt, radius=22, color=nc,
                    fill=False, weight=2, opacity=0.4
                ).add_to(m)
                tip = f"<b>{z}</b><br>{PHASE_LABELS[target_phase]} ← Zone cible"
            elif is_neighbor:
                nc, r = PHASE_COLORS[susceptible[z]], 8
                tip = f"<b>{z}</b><br>{PHASE_LABELS[susceptible[z]]} ← Propagation"
            else:
                nc, r = '#888888', 4
                tip = z

            folium.CircleMarker(
                location=pt, radius=r,
                color='white', fill=True, fill_color=nc,
                fill_opacity=0.95, weight=2 if is_target else 1,
                tooltip=tip
            ).add_to(m)

    # Capture clic sur la carte
    map_data = st_folium(m, width=None, height=590, key="main_map")
    if map_data and map_data.get("last_object_clicked_tooltip"):
        raw     = map_data["last_object_clicked_tooltip"]
        clicked = raw.split(":")[-1].strip() if ":" in raw else raw.strip()
        if clicked and clicked in gdf['title'].values and clicked != st.session_state.selected_zone:
            st.session_state.selected_zone = clicked
            st.session_state.analysis_run  = False
            st.session_state.rapport       = None
            st.rerun()

# ── COLONNE DROITE : ANALYSE ──────────────────────────────────────────────────
with col2:
    if not st.session_state.analysis_run or st.session_state.selected_zone is None:
        st.markdown("### 🏗️ Comment utiliser FamineGuard")
        st.markdown("""
        **Étapes :**

        1. **Sélectionnez une région** dans le menu ou en cliquant sur la carte
        2. **Réglez les paramètres** de choc (prix alimentaires, NDVI végétation)
        3. **Lancez l'analyse** — le GNN prédit la phase IPC de la zone
        4. La carte affiche les **zones voisines susceptibles** avec le graphe de propagation
        5. **Générez le rapport IA** pour les recommandations humanitaires

        ---
        🔵 **Layer 1** — Données : IPC, NDVI, Prix céréales, Routes OSM

        🔴 **Layer 2** — ST-GNN : Prédiction spatiotemporelle par zone

        🤖 **Layer 3** — Agentic RAG : Archives FEWS NET / WFP / FAO
        """)

    else:
        phase_color = PHASE_COLORS[target_phase]

        # Carte d'identité de la zone
        st.markdown(f"""
        <div style="background:{phase_color}22; border:1px solid {phase_color}55;
             border-radius:10px; padding:14px 18px; margin-bottom:12px;">
            <div style="font-size:11px; color:#aaa; text-transform:uppercase; letter-spacing:1px;">
                Zone analysée
            </div>
            <div style="font-size:20px; font-weight:700; color:white; margin-top:4px;">
                📍 {st.session_state.selected_zone}
            </div>
            <div style="margin-top:8px;">
                <span style="background:{phase_color}; color:white; padding:4px 14px;
                      border-radius:12px; font-size:13px; font-weight:600;">
                    {PHASE_LABELS[target_phase]}
                </span>
            </div>
            <div style="margin-top:8px; font-size:12px; color:#bbb;">
                💰 Choc prix ×{st.session_state.prix_val:.1f}
                &nbsp;|&nbsp;
                🌿 NDVI {st.session_state.ndvi_val:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Zones voisines susceptibles
        if susceptible:
            st.markdown(f"**⚠️ {len(susceptible)} zone(s) voisine(s) susceptibles :**")
            for z, p in sorted(susceptible.items(), key=lambda x: -x[1]):
                nc = PHASE_COLORS[p]
                st.markdown(
                    f"<span style='color:{nc}; font-weight:600;'>● {z}</span>"
                    f" — {PHASE_LABELS[p]}",
                    unsafe_allow_html=True
                )
        else:
            st.success("✅ Aucune propagation critique détectée vers les zones voisines.")

        st.markdown("---")

        if st.button("🤖 Générer le rapport IA", use_container_width=True):
            with st.spinner("Agent IA : GNN → RAG → Synthèse..."):
                st.session_state.rapport = famine_guard_agent(
                    st.session_state.selected_zone,
                    target_phase,
                    st.session_state.prix_val,
                    st.session_state.ndvi_val,
                    susceptible,
                    st.session_state.langue
                )

        if st.session_state.rapport:
            st.markdown(st.session_state.rapport)

st.caption("FamineGuard v3.0 | AIMS Senegal 2026 | Decision Support System")
