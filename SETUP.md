# Setup — bootstrapping a new app from this template

This is the one-time setup you do after using the template to create a new
repo. After this is done, every `git push origin main` auto-deploys to Fly.

## Prerequisites

- `gh` CLI installed and authenticated: `gh auth status`
- `flyctl` installed and authenticated: `~/.fly/bin/flyctl auth whoami`
- Cloudflare account with `griffith-pse.com` DNS zone

## 1. Create the repo from this template

Either use the GitHub UI's "Use this template" button on this repo's page,
or via CLI:

```bash
APP_SLUG=pinch-analysis           # short, lowercase, hyphenated
APP_TITLE="Pinch Analysis"        # human-readable display name
APP_TAGLINE="Heat-integration via the pinch design method"

gh repo create devin-griff/$APP_SLUG \
    --template devin-griff/griffith-pse-app-template \
    --private \
    --clone

cd $APP_SLUG
```

## 2. Substitute placeholders

Replace `<APP_SLUG>`, `<APP_TITLE>`, `<APP_NAME>`, and `<APP_TAGLINE>` in
every text file (including the Dockerfile and fly.toml):

```bash
find . -type f \( -name '*.py' -o -name '*.md' -o -name '*.toml' -o -name 'Dockerfile' \) \
    -exec sed -i \
    "s|<APP_SLUG>|$APP_SLUG|g; \
     s|<APP_TITLE>|$APP_TITLE|g; \
     s|<APP_NAME>|griffith-pse-$APP_SLUG|g; \
     s|<APP_TAGLINE>|$APP_TAGLINE|g" {} +
```

Sanity check — no placeholders left:
```bash
grep -rn '<APP_' . --include='*.py' --include='*.md' --include='*.toml' --include='Dockerfile' || echo "all substituted"
```

## 3. Add Python dependencies

Edit `requirements.txt`. Pure-pip libraries — `pyomo`, `pyomo-ripopt`,
`scikit-learn`, `scipy`, `plotly`, `altair`, `networkx`, `cvxpy`, `openai`,
`anthropic`, etc. — just go on a line each.

If you need a system library (GLPK solver binary, GraphViz, FFmpeg, etc.),
uncomment the matching block in the `Dockerfile`.

### Sidebar vs. no sidebar

`app.py` ships with the home-link logo wired up via `st.markdown` (the
sidebarless pattern used by Knapsack and Diet). If your app uses a sidebar
for set-then-solve workflows (the quad-tank pattern), swap the call for
`st.sidebar.markdown` — see the comment block above the call. The CSS uses
`position: fixed`, so the logo sits at the same viewport corner either way.

## 4. Create the Fly app

```bash
~/.fly/bin/flyctl apps create griffith-pse-$APP_SLUG
```

## 5. Issue a deploy token + add as GitHub secret (one pipe)

The token must NEVER pass through chat or shell history. Use this pipe so
it goes straight from `flyctl` stdout into `gh` stdin:

```bash
~/.fly/bin/flyctl tokens create deploy -a griffith-pse-$APP_SLUG --name github-actions \
    | gh secret set FLY_API_TOKEN --repo devin-griff/$APP_SLUG
```

Verify the secret was set:
```bash
gh secret list --repo devin-griff/$APP_SLUG
```

## 6. Add Cloudflare DNS records for the subdomain

In the Cloudflare dashboard for `griffith-pse.com`:

- Type **A**, name `<APP_SLUG>`, value `66.241.124.X` (Fly's edge — get the
  exact IP from `flyctl certs add` below; often the same IP used by other
  apps in your org)
- Type **AAAA**, name `<APP_SLUG>`, value `2a09:8280:1::112:XXXX:0`
- **Both records must be DNS-only (gray cloud)**, not Proxied. Streamlit's
  WebSocket connections won't survive Cloudflare's proxy on Fly origins.

## 7. Issue the SSL cert via Fly

```bash
~/.fly/bin/flyctl certs add $APP_SLUG.griffith-pse.com -a griffith-pse-$APP_SLUG
```

Fly responds with the recommended A/AAAA values — paste those into Cloudflare
if you didn't already in step 6. Cert validation takes 30–60 s once DNS resolves.

## 8. First deploy + commit the substituted template

```bash
git add -A
git commit -m "Bootstrap from template"
git push origin main
```

This triggers GitHub Actions which runs `flyctl deploy --remote-only`. About
2–3 minutes from push to live at `https://$APP_SLUG.griffith-pse.com`.

## 9. (Optional) Add a card to the Quarto site

On the `griffith-pse-site` repo's `wip` branch (and later `main` when ready),
add a new entry to `index.qmd` under "Featured demos":

```markdown
::: {.g-col-12 .g-col-md-4}
### <APP_TITLE>

<short description of what the app does>

[Launch demo →](https://<APP_SLUG>.griffith-pse.com){.btn .btn-primary target="_blank"}
:::
```

Push the site repo → Cloudflare Pages rebuilds in ~30 s.

---

## Future-app extension hints

### AI / LLM apps (OpenAI, Anthropic, etc.)

Set the API key as a Fly secret — it's mounted as an env var at runtime,
never committed to the repo:

```bash
~/.fly/bin/flyctl secrets set OPENAI_API_KEY=sk-... -a griffith-pse-$APP_SLUG
```

Read in the app via `os.environ["OPENAI_API_KEY"]` (or `st.secrets` if you
prefer Streamlit's wrapper). For local development, use a `.env` file (and
add `.env` to `.gitignore`).

### Heavier compute

Bump the machine size in `fly.toml` `[[vm]]` block. See the comments there
for options. Cost scales linearly with size; auto-stop still keeps idle
cost at $0.

### GPU workloads

Fly supports GPU machines (`a10`, `a100`). They require a CUDA-enabled base
image and a different Dockerfile entirely. This template targets CPU; you'd
fork it for GPU work.

### Persistent state (DB, file uploads, user history)

Add a `[mounts]` block to `fly.toml` and create a Fly volume:
```bash
flyctl volumes create data --size 1 -a griffith-pse-$APP_SLUG
```
Then in `fly.toml`:
```toml
[[mounts]]
  source = "data"
  destination = "/data"
```
SQLite at `/data/app.db` is the simplest pattern. For Postgres, use a
separate Fly Postgres app + connection string.

### Commercial solvers (Gurobi, CPLEX)

License files mount via `fly secrets`. Dockerfile fetches at startup. Out
of scope for this template, but the pattern is documented in Fly's docs.
