"""
<APP_TITLE>
<APP_TAGLINE>
"""
import streamlit as st

# `set_page_config` must be the first Streamlit call in the script.
st.set_page_config(
    page_title="<APP_TITLE>",
    page_icon="favicon.png",
    layout="wide",
)

# ── Home-link logo ────────────────────────────────────────────────────────────
# Pins a 32x32 Griffith PSE blackletter G to the viewport top-left corner.
# Clicking it navigates back to https://griffith-pse.com (same tab — the
# user is leaving the demo). Image is loaded from the Quarto site so a
# single CDN-served copy is the source of truth across all apps.
#
# `position: fixed` means the logo sits at the same screen location whether
# this app uses a sidebar or not — pick the matching markdown call below.
st.markdown("""
<style>
.home-logo-corner {
    position: fixed;
    top: 0.5rem;
    left: 0.75rem;
    z-index: 999999;
}
.home-logo-corner img {
    width: 32px;
    height: 32px;
    border-radius: 4px;
    display: block;
}
</style>
""", unsafe_allow_html=True)

_HOME_LOGO_HTML = (
    '<a class="home-logo-corner" href="https://griffith-pse.com" target="_self">'
    '<img src="https://griffith-pse.com/images/favicon.png" '
        'alt="Griffith PSE — home" />'
    '</a>'
)

# Sidebarless apps (default — Knapsack, Diet pattern):
st.markdown(_HOME_LOGO_HTML, unsafe_allow_html=True)

# Sidebar apps (quad-tank pattern) — comment out the line above and use:
# st.sidebar.markdown(_HOME_LOGO_HTML, unsafe_allow_html=True)

# ── Title block ───────────────────────────────────────────────────────────────
st.title("<APP_TITLE>")
st.caption("<APP_TAGLINE>")

# ── Sidebar inputs ────────────────────────────────────────────────────────────
# Sliders, file uploaders, model parameters, dropdowns, etc. Use a sidebar
# when the workflow is set-then-solve (configure inputs, hit a button, view
# results). Skip the sidebar when interaction is continuous and inputs are
# few — put controls inline in the main area instead.
#
# Example:
#   st.sidebar.header("Inputs")
#   x = st.sidebar.slider("x", 0.0, 10.0, 5.0)

# ── Main computation ──────────────────────────────────────────────────────────
# Build your model, call your library, run the analysis.
# Cache expensive work with @st.cache_data (for serializable returns) or
# @st.cache_resource (for solver objects, ML models, etc.).
#
# Example:
#   @st.cache_data
#   def fit_model(data):
#       return some_model(data).fit()

# ── Display ───────────────────────────────────────────────────────────────────
# Plotly / Altair charts, data tables, text output, math via st.latex, etc.

st.write("Hello, world. Replace this with your app.")
