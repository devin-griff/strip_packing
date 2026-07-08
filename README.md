# Strip Packing GDP Optimizer

A Streamlit app for the classic strip-packing problem as a generalized
disjunctive program (Pyomo + HiGHS/Gurobi): pack N rectangles into a
fixed-width strip to minimize the used length. Edit the rectangle list
inline, pick a GDP transformation (Big-M, Hull) and a MIP
solver (open-source HiGHS or commercial Gurobi); the app reformulates the
disjunctive non-overlap constraints into a MILP, solves it, and visualizes
the optimal packing. The in-app **📐 Formulation** tab walks through the
disjunctive math, the three reformulations, and the symmetry-breaking
constraints: see [References](#references) below.

**Live demo:** https://strip-packing.griffith-pse.com  
**Home:** https://griffith-pse.com

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

HiGHS ships as a pip wheel (`highspy`), so `pip install` covers everything -
no separate solver install needed. The Gurobi mode additionally needs a
Gurobi license (set `GRB_LICENSE_FILE` to point at it); without one,
gurobipy falls back to its bundled size-limited trial license, which only
handles small instances. HiGHS mode always works.

## Deployment

Auto-deploys to Fly.io on every push to `main` via
`.github/workflows/deploy.yml`. The `Dockerfile` builds a Python 3.12 image
and installs everything from `requirements.txt`; `fly.toml` configures
auto-stop machines. Custom domain wired through Cloudflare DNS.

- **Machine**: `shared-cpu-1x` · 1 GB RAM · single region (`ord`) · `min_machines_running=0` (auto-stops on idle).
- **Cost ceiling**: ~$3.89/mo if traffic kept the VM awake 24/7. Realistic on idle-heavy demo traffic: well under $1/mo per app. Bandwidth is effectively free under Fly's 100 GB/mo egress allowance.

## Files

- `app.py`: Streamlit UI, Pyomo model, HiGHS/Gurobi solve paths
- `Strip packing.ipynb`: formulation in a notebook
- `requirements.txt`: Python deps
- `Dockerfile`, `fly.toml`, `.dockerignore`: Fly.io production image config
- `.github/workflows/deploy.yml`: auto-deploy pipeline

## References

[1] Q. Chen, E. S. Johnson, D. E. Bernal, R. Valentin, S. Kale, J. Bates,
J. D. Siirola, and I. E. Grossmann, "Pyomo.GDP: an ecosystem for logic based
modeling and optimization development," *Optimization and Engineering*,
vol. 23, no. 1, pp. 607–642, 2022.
[Springer](https://link.springer.com/article/10.1007/s11081-021-09601-7)

[2] R. Raman and I. E. Grossmann, "Modelling and computational techniques for
logic based integer programming," *Computers & Chemical Engineering*,
vol. 18, no. 7, pp. 563–578, 1994.
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/0098135493E00107)

[3] P. M. Castro and I. E. Grossmann, "Generalized Disjunctive Programming as
a Systematic Modeling Framework to Derive Scheduling Formulations,"
*Industrial & Engineering Chemistry Research*, vol. 51, no. 16, pp. 5781–5792,
2012.
[ACS](https://pubs.acs.org/doi/10.1021/ie2030486)

[4] N. W. Sawaya and I. E. Grossmann, "A cutting plane method for solving
linear generalized disjunctive programming problems," *Computers & Chemical
Engineering*, vol. 29, no. 9, pp. 1891–1913, 2005.
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0098135405000992)

[5] F. Trespalacios and I. E. Grossmann, "Symmetry breaking for generalized
disjunctive programming formulation of the strip packing problem," *Annals
of Operations Research*, vol. 258, pp. 747–759, 2017.
[Springer](https://link.springer.com/article/10.1007/s10479-016-2112-9)

[6] Q. Huangfu and J. A. J. Hall, "Parallelizing the dual revised simplex
method," *Mathematical Programming Computation*, vol. 10, no. 1, pp. 119–142,
2018.
[Springer](https://link.springer.com/article/10.1007/s12532-017-0130-5)

[7] Gurobi Optimization, LLC, *Gurobi Optimizer Reference Manual*, 2026.
[gurobi.com](https://www.gurobi.com)

[8] M. L. Bynum, G. A. Hackebeil, W. E. Hart, C. D. Laird, B. L. Nicholson,
J. D. Siirola, J.-P. Watson, and D. L. Woodruff, *Pyomo: Optimization
Modeling in Python*, 3rd ed. Cham: Springer, 2021.
[Springer](https://link.springer.com/book/10.1007/978-3-030-68928-5)
