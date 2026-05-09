# <APP_TITLE>

<APP_TAGLINE>

**Live demo:** https://<APP_SLUG>.griffith-pse.com  
**Home:** https://griffith-pse.com

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

## Deployment

Auto-deploys to Fly.io on every push to `main` via
`.github/workflows/deploy.yml`. The `Dockerfile` builds a Python 3.12 image
and installs everything from `requirements.txt`; `fly.toml` configures
auto-stop machines (idle = $0/mo). Custom domain wired through Cloudflare DNS.

## Files

- `app.py` — Streamlit UI and computation
- `requirements.txt` — Python deps
- `favicon.png` — Griffith PSE blackletter G favicon
- `Dockerfile`, `fly.toml`, `.dockerignore` — Fly.io production image config
- `.github/workflows/deploy.yml` — auto-deploy pipeline
