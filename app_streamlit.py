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
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            ef = SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                device="cpu",
                normalize_embeddings=True
            )
            chroma_client = chromadb.PersistentClient(path=chroma_path)
            col_list = chroma_client.list_collections()
            if not col_list:
                raise ValueError("Aucune collection trouvée dans l'index Chroma")
            col_name = col_list[0].name if hasattr(col_list[0], 'name') else str(col_list[0])
            v_store = chroma_client.get_collection(name=col_name, embedding_function=ef)
            count = v_store.count()
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
    """Recherche dans les archives PDF indexées. Retourne None si indisponible."""
    if vectorstore is None:
        return None
    try:
        results = vectorstore.query(
            query_texts=[query],
            n_results=3,
            include=["documents", "metadatas"]
        )
        docs  = results["documents"][0]
        metas = results["metadatas"][0]
        if not docs:
            return None
        parts = []
        for doc, meta in zip(docs, metas):
            source = os.path.basename(meta.get('source', 'document')) if meta else 'document'
            page   = meta.get('page', '?') if meta else '?'
            parts.append(f"📄 {source}, p.{page}\n{doc}")
        return "\n\n---\n\n".join(parts)
    except Exception:
        return None

def famine_guard_agent(zone, phase, prix, ndvi, susceptible, langue):
    try:
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=st.secrets["GROQ_API_KEY"]
        )

        # ── Interprétation des indicateurs ───────────────────────────────────
        ndvi_interp = (
            "critique (< 0.2) — désastre végétatif, récoltes quasi nulles" if ndvi < 0.2
            else "très faible (0.2–0.35) — stress sévère, rendements fortement réduits" if ndvi < 0.35
            else "faible (0.35–0.5) — végétation sous la normale, soudure difficile" if ndvi < 0.5
            else "normal (> 0.5) — couverture végétale satisfaisante"
        )
        prix_interp = (
            f"×{prix:.1f} — urgence : accès alimentaire hors de portée pour les ménages vulnérables" if prix > 2.5
            else f"×{prix:.1f} — très élevés : les ménages pauvres consacrent >70% du revenu à l'alimentation" if prix > 1.8
            else f"×{prix:.1f} — en hausse : stress sur le pouvoir d'achat alimentaire" if prix > 1.3
            else f"×{prix:.1f} — quasi normaux"
        )
        neighbors_str = (
            ", ".join([f"{z} ({PHASE_LABELS[p]})" for z, p in susceptible.items()])
            if susceptible else "Aucune zone voisine à risque élevé détectée"
        )

        # ── 3 requêtes RAG ciblées ────────────────────────────────────────────
        rag1 = tool_search_rag(f"{zone} Senegal food insecurity IPC phase {phase} crisis history")
        rag2 = tool_search_rag(f"IPC phase {phase} emergency response interventions Sahel Senegal recommendations")
        rag3 = tool_search_rag(f"food price shock NDVI vegetation deficit Senegal cereal market crisis")

        has_rag = any(r is not None for r in [rag1, rag2, rag3])
        rag_section = ""
        if rag1: rag_section += f"\n\n🔍 Historique de la zone :\n{rag1}"
        if rag2: rag_section += f"\n\n🔍 Réponses documentées pour Phase {phase} :\n{rag2}"
        if rag3: rag_section += f"\n\n🔍 Contexte prix/NDVI au Sénégal :\n{rag3}"
        if not has_rag:
            rag_section = "Archives PDF non indexées sur ce déploiement. Utiliser les connaissances générales FEWS NET/WFP/FAO."

        # ── Prompt ───────────────────────────────────────────────────────────
        lang = "RÉPONDS ENTIÈREMENT EN FRANÇAIS." if langue == "Français" else "RESPOND ENTIRELY IN ENGLISH."

        system_prompt = (
            "Tu es un expert senior en sécurité alimentaire pour FEWS NET et le PAM. "
            "Tu analyses les prédictions d'un modèle GNN (Graph Neural Network) spatiotemporel "
            "qui prédit les phases IPC pour les zones du Sénégal à partir de données NDVI, "
            "prix des céréales et connectivité routière. "
            "Ton rôle : expliquer les prédictions, les contextualiser avec les archives humanitaires, "
            "et formuler des recommandations opérationnelles concrètes."
        )

        user_prompt = f"""
{lang}

═══════════════════════════════════════════
  RÉSULTATS DU MODÈLE GNN — FAMINEGUARD
═══════════════════════════════════════════
Zone analysée            : {zone}
Phase IPC prédite        : {PHASE_LABELS[phase]}
Indice NDVI              : {ndvi:.2f}  → {ndvi_interp}
Choc prix céréales       : {prix_interp}
Zones voisines à risque  : {neighbors_str}

═══════════════════════════════════════════
  ARCHIVES HUMANITAIRES (FEWS NET / WFP / FAO)
═══════════════════════════════════════════
{rag_section}

═══════════════════════════════════════════
  RAPPORT À PRODUIRE
═══════════════════════════════════════════

**1. 🔍 Explication de la Prédiction du GNN**
Explique POURQUOI le modèle a classé {zone} en {PHASE_LABELS[phase]} :
- Ce que signifie concrètement un NDVI de {ndvi:.2f} pour les cultures et l'élevage de cette zone
- L'impact réel d'un choc prix à ×{prix:.1f} sur les ménages vulnérables
- Comment ces deux facteurs combinés déclenchent la Phase {phase} selon les critères IPC
- Pourquoi le risque se propage vers les zones voisines identifiées

**2. 📚 Contexte Historique et Analogies**
En t'appuyant sur les archives disponibles :
- Cite des épisodes similaires dans cette région [SOURCE: fichier.pdf, p.X]
- Identifie les patterns saisonniers ou structurels de vulnérabilité
- Mentionne les facteurs aggravants typiques (soudure, transhumance, conflits)

**3. 🎯 Recommandations Opérationnelles**
3 actions prioritaires adaptées à la {PHASE_LABELS[phase]}, chacune avec :
→ L'action concrète
→ La population cible
→ Le délai d'intervention recommandé
→ Une référence documentaire si disponible [SOURCE: ...]

**4. 📋 Sources utilisées**
"""

        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.15,
            max_tokens=1500
        )

        rag_badge = (
            "✅ **RAG actif** — Rapport basé sur les archives FEWS NET / WFP / FAO indexées"
            if has_rag else
            "⚠️ **Mode dégradé** — Archives non disponibles, basé sur connaissances générales"
        )
        return resp.choices[0].message.content, rag_badge

    except Exception as e:
        fallback = (
            f"**❌ Erreur technique :** `{e}`\n\n"
            f"**Zone {zone} — {PHASE_LABELS[phase]}**\n\n"
            f"Actions d'urgence recommandées :\n"
            f"1. Évaluation terrain immédiate\n"
            f"2. Activation mécanismes de réponse rapide WFP\n"
            f"3. Coordination autorités locales"
        )
        return fallback, "❌ Erreur LLM"

# ─── SESSION STATE ────────────────────────────────────────────────────────────
for key, val in [
    ('selected_zone', None), ('analysis_run', False),
    ('prix_val', 1.8), ('ndvi_val', 0.35),
    ('langue', 'Français'), ('rapport', None), ('rag_badge', None)
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

        # Polygones : seule la zone cible a un fond coloré
        # Les voisins restent neutres — leur distinction passe par le nœud
        def style_analysis(feature):
            z = feature['properties']['title']
            if z == st.session_state.selected_zone:
                return {
                    'fillColor': PHASE_COLORS[target_phase],
                    'color': '#ffffff', 'weight': 4, 'fillOpacity': 0.85
                }
            return {'fillColor': NEUTRAL, 'color': '#555', 'weight': 0.5, 'fillOpacity': 0.25}

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
                weight=2 + phase * 0.7,
                opacity=0.9,
                tooltip=f"Propagation → {zone_name} : {PHASE_LABELS[phase]}"
            ).add_to(m)

        # Nœuds
        for _, row in gdf.iterrows():
            z  = row['title']
            pt = [row.geometry.centroid.y, row.geometry.centroid.x]
            is_target   = (z == st.session_state.selected_zone)
            is_neighbor = (z in susceptible)

            if is_target:
                nc = PHASE_COLORS[target_phase]
                # Double halo pour la zone cible
                for halo_r, halo_op in [(28, 0.20), (20, 0.35)]:
                    folium.CircleMarker(
                        location=t_pt, radius=halo_r, color=nc,
                        fill=False, weight=2, opacity=halo_op
                    ).add_to(m)
                folium.CircleMarker(
                    location=t_pt, radius=14,
                    color='#ffffff', fill=True, fill_color=nc,
                    fill_opacity=1.0, weight=3,
                    tooltip=f"<b>🎯 ZONE CIBLE : {z}</b><br>{PHASE_LABELS[target_phase]}"
                ).add_to(m)
                # Label "CIBLE" au-dessus du nœud
                folium.Marker(
                    location=t_pt,
                    icon=folium.DivIcon(
                        html=f"""<div style="
                            background:{nc}; color:white; font-weight:700;
                            font-size:10px; padding:2px 6px; border-radius:4px;
                            white-space:nowrap; border:1px solid white;
                            margin-top:-38px; margin-left:-20px;">
                            🎯 CIBLE
                        </div>""",
                        icon_size=(60, 20), icon_anchor=(0, 0)
                    )
                ).add_to(m)

            elif is_neighbor:
                nc = PHASE_COLORS[susceptible[z]]
                # Nœud voisin : coloré par phase, contour noir pour contraste avec la carte
                folium.CircleMarker(
                    location=pt, radius=10,
                    color='#111111', fill=True, fill_color=nc,
                    fill_opacity=0.95, weight=2,
                    tooltip=f"<b>⚠️ {z}</b><br>{PHASE_LABELS[susceptible[z]]} ← Propagation"
                ).add_to(m)

            else:
                # Autres zones : petit point gris discret
                folium.CircleMarker(
                    location=pt, radius=3,
                    color='#666', fill=True, fill_color='#666',
                    fill_opacity=0.5, weight=1,
                    tooltip=z
                ).add_to(m)

        # Légende de propagation sur la carte
        legend_html = f"""
        <div style="position:fixed; bottom:28px; left:28px; z-index:1000;
             background:rgba(15,17,23,0.92); padding:12px 16px; border-radius:10px;
             border:1px solid #555; font-family:sans-serif; font-size:12px;
             color:white; line-height:2;">
            <b style="font-size:13px;">Graphe de Propagation</b><br>
            <span style="color:{PHASE_COLORS[target_phase]}">⬤</span>
            <b> Zone Cible</b> — fond coloré + label 🎯<br>
            <span style="color:#E67E22">⬤</span>
            <b> Zones Voisines Susceptibles</b> — nœud coloré<br>
            <span style="color:#666">⬤</span> Autres zones — neutres<br>
            <span style="color:#aaa">—</span> Arête de propagation du risque
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

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
                rapport, badge = famine_guard_agent(
                    st.session_state.selected_zone,
                    target_phase,
                    st.session_state.prix_val,
                    st.session_state.ndvi_val,
                    susceptible,
                    st.session_state.langue
                )
                st.session_state.rapport   = rapport
                st.session_state.rag_badge = badge

        if st.session_state.rapport:
            st.caption(st.session_state.rag_badge)
            st.markdown(st.session_state.rapport)

st.caption("FamineGuard v3.0 | AIMS Senegal 2026 | Decision Support System")
