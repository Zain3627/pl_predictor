import streamlit as st
import pandas as pd
import psycopg2
import os
from pathlib import Path
import base64

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PL Predictor",
    page_icon="⚽",
    layout="centered",
)

# ── Team crest mapping ──────────────────────────────────────────────────────
TEAM_ICONS = {
    "Arsenal": "arsenal.png",
    "Aston Villa": "aston_villa.png",
    "Bournemouth": "bournemouth.png",
    "Brentford": "brentford.png",
    "Brighton": "brighton.png",
    "Burnley": "burnley.png",
    "Chelsea": "chelsea.png",
    "Crystal Palace": "crystal_palace.png",
    "Everton": "everton.png",
    "Fulham": "fulham.png",
    "Leeds": "leeds.png",
    "Liverpool": "liverpool.png",
    "Man City": "man_city.png",
    "Man United": "man_united.png",
    "Newcastle": "newcastle.png",
    "Nott'm Forest": "nottm_forest.png",
    "Sunderland": "sunderland.png",
    "Tottenham": "tottenham.png",
    "West Ham": "west_ham.png",
    "Wolves": "wolves.png",
}

ICONS_DIR = Path(__file__).parent / "icons"


def img_to_base64(path: Path) -> str:
    """Return a base64-encoded data URI for an image file."""
    if path.exists():
        encoded = base64.b64encode(path.read_bytes()).decode()
        suffix = path.suffix.lstrip(".")
        return f"data:image/{suffix};base64,{encoded}"
    return ""


# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""<style>
.block-container { max-width: 720px; padding-top: 2rem; }
.header { text-align: center; margin-bottom: 2rem; }
.header img { width: 80px; margin-bottom: 0.5rem; }
.header h1 {
    font-size: 2rem; font-weight: 700; margin: 0;
    background: linear-gradient(135deg, #3d195b, #00ff87);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.header p { color: #888; font-size: 0.95rem; margin-top: 0.25rem; }
.table-container {
    background: #ffffff08;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #ffffff15;
    margin-bottom: 2rem;
}
.table-row {
    display: flex; align-items: center;
    padding: 12px 20px;
    border-bottom: 1px solid #ffffff0a;
    transition: background 0.15s;
}
.table-row:hover { background: #ffffff0a; }
.table-row:last-child { border-bottom: none; }
.table-header {
    display: flex; align-items: center;
    padding: 14px 20px;
    background: linear-gradient(135deg, #3d195b, #1a0a2e);
    font-weight: 600; font-size: 0.85rem;
    color: #00ff87; text-transform: uppercase; letter-spacing: 0.5px;
}
.col-pos  { width: 40px; text-align: center; font-weight: 700; color: #888; font-size: 0.9rem; }
.col-crest { width: 36px; margin-right: 12px; display: flex; align-items: center; justify-content: center; }
.col-crest img { width: 28px; height: 28px; object-fit: contain; }
.col-team { flex: 1; font-weight: 500; font-size: 1rem; }
.col-pts  { width: 60px; text-align: center; font-weight: 700; font-size: 1.05rem; color: #00ff87; }
.pos-1 .col-pos, .pos-2 .col-pos, .pos-3 .col-pos, .pos-4 .col-pos {
    color: #00ff87;
}
.pos-18 .col-pos, .pos-19 .col-pos, .pos-20 .col-pos {
    color: #ff4654;
}
.zone-ucl { border-left: 3px solid #00ff87; }
.zone-rel { border-left: 3px solid #ff4654; }
.footer {
    text-align: center; padding: 2rem 0 1rem;
    border-top: 1px solid #ffffff15; margin-top: 1rem;
}
.footer p { color: #888; font-size: 0.85rem; margin: 0.25rem 0; }
.footer a {
    color: #00ff87; text-decoration: none; margin: 0 8px;
    font-weight: 500; transition: opacity 0.2s;
}
.footer a:hover { opacity: 0.75; }
.footer-links { margin-top: 0.5rem; }
</style>""", unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_predicted_table() -> pd.DataFrame:
    """Fetch the predicted league table from Supabase (PostgreSQL)."""
    connection = psycopg2.connect(
        user=st.secrets["db"]["user"],
        password=st.secrets["db"]["password"],
        host=st.secrets["db"]["host"],
        port=st.secrets["db"]["port"],
        dbname=st.secrets["db"]["dbname"],
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT * FROM "predicted_league_table" ORDER BY row_id'
            )
            df = pd.DataFrame(
                cursor.fetchall(),
                columns=[desc[0] for desc in cursor.description],
            )
            df = df.drop(columns=["row_id"])
    finally:
        connection.close()
    return df


# ── Header ───────────────────────────────────────────────────────────────────
logo_path = ICONS_DIR / "pl_logo.png"
logo_html = (
    f'<img src="{img_to_base64(logo_path)}" alt="PL Logo">'
    if logo_path.exists()
    else ""
)

st.markdown(f"""
<div class="header">
    {logo_html}
    <h1>Premier League Predictor</h1>
    <p>Predicted final standings for the 2025 / 26 season</p>
</div>
""", unsafe_allow_html=True)

# ── Load & display table ────────────────────────────────────────────────────
try:
    df = load_predicted_table()
except Exception as e:
    st.error(f"Failed to load data from database: {e}")
    st.stop()

# Table header row
st.markdown("""
<div class="table-container">
<div class="table-header">
    <div class="col-pos">#</div>
    <div class="col-crest"></div>
    <div class="col-team">Club</div>
    <div class="col-pts">Pts</div>
</div>
""", unsafe_allow_html=True)

# Data rows
rows_html = ""
for idx, row in df.iterrows():
    pos = idx + 1
    team = row["team"]
    pts = int(row["total_points"])

    # Zone CSS class
    if pos <= 5:
        zone = "zone-ucl"
    elif pos >= 18:
        zone = "zone-rel"
    else:
        zone = ""

    # Crest image
    icon_file = TEAM_ICONS.get(team, "")
    icon_path = ICONS_DIR / icon_file if icon_file else None
    if icon_path and icon_path.exists():
        crest_html = f'<img src="{img_to_base64(icon_path)}" alt="{team}">'
    else:
        crest_html = ""

    rows_html += f"""
<div class="table-row pos-{pos} {zone}">
<div class="col-pos">{pos}</div>
<div class="col-crest">{crest_html}</div>
<div class="col-team">{team}</div>
<div class="col-pts">{pts}</div>
</div>
"""

st.markdown(rows_html + "</div>", unsafe_allow_html=True)

# ── Legend ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; gap:24px; justify-content:center; margin:1rem 0 0.5rem; font-size:0.8rem; color:#888;">
<span><span style="color:#00ff87;">&#9632;</span> UEFA Champions League</span>
<span><span style="color:#ff4654;">&#9632;</span> Relegation</span>
</div>
""", unsafe_allow_html=True)

# ── Footer / Contact ────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
<p>Built by <strong>Zain Tamer</strong></p>
<div class="footer-links">
    <a href="https://www.linkedin.com/in/zaintamer/" target="_blank">LinkedIn</a>
    <a href="https://github.com/Zain3627/" target="_blank">GitHub</a>
    <a href="https://github.com/Zain3627/pl_predictor" target="_blank">Source Code</a>
</div>
</div>
""", unsafe_allow_html=True)
