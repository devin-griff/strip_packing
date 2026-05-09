# Streamlit app on Python 3.12 slim, deployed to Fly.io.
# This Dockerfile is the template default. Pure-pip apps (scikit-learn,
# scipy, plotly, pyomo, pyomo-ripopt, etc.) need NO changes here — pip
# installs everything from requirements.txt.
#
# If your app needs a system-level library (a solver binary, GraphViz,
# FFmpeg, etc.) uncomment the matching block below.
FROM python:3.12-slim

# ── Optional system dependencies ─────────────────────────────────────────────
# Uncomment whichever your app needs. Default is nothing — most apps don't
# need any system packages.
#
# # GLPK (LP/MIP solver via Pyomo: SolverFactory('glpk'))
# RUN apt-get update \
#     && apt-get install -y --no-install-recommends glpk-utils \
#     && rm -rf /var/lib/apt/lists/*
#
# # GraphViz (for network/graph diagrams)
# RUN apt-get update \
#     && apt-get install -y --no-install-recommends graphviz \
#     && rm -rf /var/lib/apt/lists/*
#
# # FFmpeg (video / audio processing)
# RUN apt-get update \
#     && apt-get install -y --no-install-recommends ffmpeg \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies first (better Docker layer caching).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App source + favicon (referenced by st.set_page_config(page_icon=...)).
COPY app.py favicon.png ./

# Overwrite Streamlit's default static index.html title and favicon so the
# initial render — before the React app boots and applies set_page_config —
# already shows our app name and the blackletter-G favicon, instead of the
# default "Streamlit" title flashing for ~1s before being replaced.
RUN STATIC=$(python -c "import streamlit, os; print(os.path.join(os.path.dirname(streamlit.__file__), 'static'))") \
    && sed -i 's|<title>Streamlit</title>|<title><APP_TITLE></title>|' "$STATIC/index.html" \
    && cp /app/favicon.png "$STATIC/favicon.png"

EXPOSE 8080
CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
