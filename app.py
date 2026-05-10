# =============================================================================
# Strip Packing GDP Optimizer — a Streamlit tutorial app.
#
# This file builds an interactive web app around the classic strip-packing
# problem: given N rectangles with widths w_i and heights h_i, place them
# inside a vertical strip of fixed width W so that no two rectangles overlap
# and the strip's used height H is minimized.
#
# It is a Generalized Disjunctive Program (GDP) — for each pair of rectangles
# (i, j), at least one of four geometric relationships must hold:
#     i is left of j, i is right of j, i is below j, or i is above j.
# Pyomo's `gdp` module expresses these `Disjunction` blocks natively. A
# `TransformationFactory` step (Big-M or Hull) reformulates the GDP into a
# standard MILP that HiGHS can solve.
#
# Library roadmap:
#   - streamlit  — the UI framework. Each interaction reruns this script
#                  top-to-bottom; persistent values live in `st.session_state`.
#   - pyomo      — algebraic modeling: sets, params, vars, objective,
#                  constraints. The `pyomo.gdp` submodule adds Disjunction.
#   - HiGHS      — the MILP solver, called via Pyomo's appsi_highs interface.
#                  Ships as a pip wheel (`highspy`).
#   - pandas     — DataFrame shape for Streamlit's data editor and Altair.
#   - altair     — Rectangle plot with `mark_rect` + `mark_text` labels.
#
# File roadmap:
#   1. Solver       — model definition, GDP transformation, HiGHS log capture.
#   2. State        — session_state init / reset.
#   3. Utilities    — DataFrame <-> internal-dict conversion, geometry helpers.
#   4. LaTeX        — render the general formulation and instance summary.
#   5. Tabs         — render_optimizer / render_data / render_formulation /
#                     render_logs.
#   6. Main         — page config, sidebar, tab assembly.
# =============================================================================

import base64
import copy
from pathlib import Path

import altair as alt
import pandas as pd
import pyomo.environ as pyo
import streamlit as st
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.gdp import Disjunction
from pyomo.opt import TerminationCondition


# Hard cap on rectangle count. Big-M MILP gets slow beyond ~15 rectangles
# because the number of disjunctions grows as N(N-1)/2.
MAX_RECTS = 15

# Default instance shown on first load and after the "Reset to defaults"
# button. A representative 8-rectangle problem with mixed shapes — some tall,
# some wide, some square — so the optimal layout is visually interesting.
DEFAULT_DATA = {
    "rects": [1, 2, 3, 4, 5, 6, 7, 8],
    "w": {1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 2.0, 6: 3.0, 7: 6.0, 8: 1.0},
    "h": {1: 6.0, 2: 4.0, 3: 2.0, 4: 3.0, 5: 5.0, 6: 3.0, 7: 2.0, 8: 7.0},
    "W": 10.0,
}


# ---------- Solver ----------
#
# Standard Pyomo with the `gdp` submodule. Disjunctions are written natively;
# a `TransformationFactory` step rewrites them into ordinary MILP constraints
# that HiGHS can solve. The only twist is `_solve_capturing`, which redirects
# HiGHS's solver output at the OS file-descriptor level so we can show it in
# the Logs tab.

def build_model(data):
    # ConcreteModel: components bound to data at construction time.
    m = pyo.ConcreteModel()

    rects = list(data["rects"])
    W = float(data["W"])

    # Sums used for variable bounds. The strip height H is bounded above by
    # the sum of all rectangle heights (worst case: stacking all rectangles
    # vertically). Each y_i is bounded by the same sum.
    H_max = float(sum(data["h"][i] for i in rects)) if rects else 1.0
    H_max = max(H_max, 1.0)

    # Index set over rectangles.
    m.RECTS = pyo.Set(initialize=rects, ordered=True)

    # Parameters: known data the solver does not change.
    m.w = pyo.Param(m.RECTS, initialize={i: float(data["w"][i]) for i in rects})
    m.h = pyo.Param(m.RECTS, initialize={i: float(data["h"][i]) for i in rects})
    m.W = pyo.Param(initialize=W)

    # Decision variables. Bounded explicitly so Pyomo's gdp.bigm transformation
    # can derive sensible Big-M values automatically. x_i sits in [0, W],
    # y_i in [0, H_max], H itself in [0, H_max].
    m.x = pyo.Var(m.RECTS, domain=pyo.NonNegativeReals, bounds=(0.0, W))
    m.y = pyo.Var(m.RECTS, domain=pyo.NonNegativeReals, bounds=(0.0, H_max))
    m.H = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.0, H_max))

    # Objective: minimize the strip height H.
    m.height = pyo.Objective(expr=m.H, sense=pyo.minimize)

    # Containment: each rectangle must fit inside the strip. Horizontal fit
    # is enforced by the upper bound on x plus this constraint; vertical fit
    # ties each rectangle's top edge to the strip-height variable H.
    def fit_x_def(m, i):
        return m.x[i] + m.w[i] <= m.W
    m.fit_x = pyo.Constraint(m.RECTS, rule=fit_x_def)

    def fit_y_def(m, i):
        return m.y[i] + m.h[i] <= m.H
    m.fit_y = pyo.Constraint(m.RECTS, rule=fit_y_def)

    # Non-overlap disjunctions: for every unordered pair (i, j) with i < j,
    # at least one of the four geometric separations must hold. `Disjunction`
    # accepts a list of disjuncts, each being a list of constraint expressions.
    pairs = [(i, j) for idx_i, i in enumerate(rects) for j in rects[idx_i + 1:]]
    if pairs:
        m.PAIRS = pyo.Set(initialize=pairs, dimen=2)

        def disj_rule(m, i, j):
            return [
                [m.x[i] + m.w[i] <= m.x[j]],   # i is left of j
                [m.x[j] + m.w[j] <= m.x[i]],   # i is right of j
                [m.y[i] + m.h[i] <= m.y[j]],   # i is below j
                [m.y[j] + m.h[j] <= m.y[i]],   # i is above j
            ]
        m.no_overlap = Disjunction(m.PAIRS, rule=disj_rule)

    return m


def _solve_capturing(m, transform):
    """Apply the GDP transformation, run HiGHS, return (results, log_text).
    Captures HiGHS's stdout via Pyomo's capture_output (FD-level redirect on
    newer Pyomo, plain stdout capture on older). HiGHS via the appsi_highs
    LegacySolver doesn't support a logfile= kwarg, so the FD capture is the
    only path."""
    # Reformulate the GDP into a standard MILP. Big-M produces fewer
    # variables but weaker LP relaxations; Hull adds disaggregated copies of
    # the variables but tends to give tighter relaxations and better solve
    # times on harder instances.
    pyo.TransformationFactory(transform).apply_to(m)

    log_text = ""
    try:
        with capture_output(capture_fd=True) as buf:
            solver = pyo.SolverFactory("appsi_highs")
            results = solver.solve(m, tee=True)
        log_text = buf.getvalue()
    except TypeError:
        # Older Pyomo without capture_fd — fall back to plain stdout capture.
        with capture_output() as buf:
            solver = pyo.SolverFactory("appsi_highs")
            results = solver.solve(m, tee=True)
        log_text = buf.getvalue()
    return results, log_text


def solve(data, transform="gdp.bigm"):
    # Top-level entrypoint used by the UI. Always returns a plain dict so the
    # caller can stash the result in session_state without holding on to a
    # live Pyomo model.

    if not data["rects"]:
        return {"status": "no_rects", "x": {}, "y": {}, "H": None,
                "log": "", "transform": transform}

    # Sanity check: any rectangle wider than the strip is infeasible by
    # inspection. Surface the error before the solver does so we can show a
    # nicer message.
    bad = [i for i in data["rects"] if data["w"][i] > data["W"] + 1e-9]
    if bad:
        return {
            "status": "infeasible_data",
            "message": (
                f"Rectangle(s) {bad} are wider than the strip width "
                f"W = {data['W']:g}. Reduce their width or increase W."
            ),
            "x": {}, "y": {}, "H": None, "log": "", "transform": transform,
        }

    m = build_model(data)

    try:
        results, log = _solve_capturing(m, transform)
    except ApplicationError as e:
        return {
            "status": "solver_missing",
            "message": (
                "HiGHS solver not available. Run `pip install highspy` "
                f"in your environment. ({e})"
            ),
            "x": {}, "y": {}, "H": None, "log": "", "transform": transform,
        }

    # Translate Pyomo's TerminationCondition enum into a small set of stable
    # status strings the UI knows how to render.
    tc = results.solver.termination_condition
    if tc == TerminationCondition.optimal:
        x = {i: float(pyo.value(m.x[i])) for i in data["rects"]}
        y = {i: float(pyo.value(m.y[i])) for i in data["rects"]}
        H = float(pyo.value(m.H))
        return {
            "status": "optimal",
            "x": x, "y": y, "H": H,
            "log": log, "transform": transform,
        }
    if tc in (
        TerminationCondition.infeasible,
        TerminationCondition.infeasibleOrUnbounded,
    ):
        return {"status": "infeasible", "x": {}, "y": {}, "H": None,
                "log": log, "transform": transform}
    if tc == TerminationCondition.unbounded:
        return {"status": "unbounded", "x": {}, "y": {}, "H": None,
                "log": log, "transform": transform}
    return {"status": str(tc), "x": {}, "y": {}, "H": None,
            "log": log, "transform": transform}


# ---------- State ----------
#
# Streamlit re-executes the whole script on every interaction. Anything that
# must persist between runs lives in `st.session_state`. The keys we use:
#   - data:                the current problem instance (rects, w, h, W)
#   - optimal:             the most recent solver result, or None
#   - _pending_reset:      one-shot flag to reset on the next run
#   - W_input:             the value backing the sidebar number_input
#   - transform_radio:     the value backing the sidebar radio
#   - data_editor:         backing key for the data editor widget

def init_state():
    # Idempotent initialization: only seed defaults the first time, otherwise
    # the user's edits would be wiped on every rerun.
    if "data" not in st.session_state:
        st.session_state.data = copy.deepcopy(DEFAULT_DATA)
    if "optimal" not in st.session_state:
        st.session_state.optimal = None
    # The reset button can't directly mutate widget-backed keys without
    # raising a Streamlit error, so it sets a flag and reruns. We then
    # apply the reset *before* widgets are instantiated this run.
    if st.session_state.pop("_pending_reset", False):
        apply_reset()


def apply_reset():
    # Restore the default instance and clear any user-driven state.
    st.session_state.data = copy.deepcopy(DEFAULT_DATA)
    st.session_state.optimal = None
    st.session_state["W_input"] = float(DEFAULT_DATA["W"])


# ---------- Utilities ----------
#
# Adapters between two data shapes:
#   - The "internal" dict shape used by the solver and most of the app.
#   - The DataFrame shape used by Streamlit's data editor widget.
# Plus small helpers for areas, lower bounds, and labelled rows.

def data_to_df(data):
    # Internal -> DataFrame. Used to seed the data editor each render.
    return pd.DataFrame([
        {"Index": int(i), "Width": float(data["w"][i]), "Height": float(data["h"][i])}
        for i in data["rects"]
    ])


def df_to_data(df, W):
    # DataFrame -> internal. Normalizes whatever the user typed: drop blank
    # rows, coerce numerics, clamp to non-negative. Indices are always
    # renumbered 1..N so the user can delete rows freely without leaving gaps.
    df = df.copy()
    for col in ["Width", "Height"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Width", "Height"])
    df["Width"] = df["Width"].clip(lower=0.0)
    df["Height"] = df["Height"].clip(lower=0.0)
    df = df[(df["Width"] > 0) & (df["Height"] > 0)]
    rects = list(range(1, len(df) + 1))
    w = {i: float(row.Width) for i, row in zip(rects, df.itertuples())}
    h = {i: float(row.Height) for i, row in zip(rects, df.itertuples())}
    return {"rects": rects, "w": w, "h": h, "W": float(W)}


def total_area(data):
    return sum(float(data["w"][i]) * float(data["h"][i]) for i in data["rects"])


def lower_bound_H(data):
    # A trivial lower bound on the strip height H: the taller of (a) the
    # tallest rectangle and (b) total area divided by strip width.
    if not data["rects"]:
        return 0.0
    max_h = max(float(data["h"][i]) for i in data["rects"])
    area_lb = total_area(data) / float(data["W"]) if data["W"] > 0 else 0.0
    return max(max_h, area_lb)


# ---------- Tabs ----------
#
# One render_* function per tab. Optimizer is the main view; Data lets the
# user edit rectangles; Formulation shows the math; Logs shows HiGHS output.

# A 12-color categorical palette repeated as needed. Tableau-style; reads
# well at small rectangle sizes and keeps adjacent indices distinguishable.
_PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B",
    "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#1F77B4", "#9467BD",
]


def _packing_chart(data, layout, H, H_max):
    # Build the rectangle plot. `layout` is the dict {i: (x, y)} from the
    # solver. The strip outline is drawn separately so it stays visible even
    # when no rectangles overlap it.
    rects = data["rects"]
    rows = []
    for i in rects:
        x = float(layout["x"][i])
        y = float(layout["y"][i])
        w = float(data["w"][i])
        h = float(data["h"][i])
        rows.append({
            "index": str(int(i)),
            "x": x, "y": y,
            "x_end": x + w, "y_end": y + h,
            "cx": x + w / 2.0, "cy": y + h / 2.0,
            "w": w, "h": h,
        })
    df = pd.DataFrame(rows)

    W = float(data["W"])
    # Force consistent x and y scales so rectangles look square-ish; the
    # chart's pixel size is set via .properties(width, height).
    x_scale = alt.Scale(domain=[0.0, W])
    y_top = max(H_max, H if H else 0.0, 1.0)
    y_scale = alt.Scale(domain=[0.0, y_top])

    domain = [str(int(i)) for i in rects]
    palette = [_PALETTE[(int(i) - 1) % len(_PALETTE)] for i in rects]

    rects_chart = (
        alt.Chart(df)
        .mark_rect(stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X("x:Q", scale=x_scale, title="x"),
            x2="x_end:Q",
            y=alt.Y("y:Q", scale=y_scale, title="y"),
            y2="y_end:Q",
            color=alt.Color(
                "index:N",
                scale=alt.Scale(domain=domain, range=palette),
                legend=alt.Legend(title="Rectangle"),
            ),
            tooltip=[
                alt.Tooltip("index:N", title="i"),
                alt.Tooltip("x:Q", format=".2f"),
                alt.Tooltip("y:Q", format=".2f"),
                alt.Tooltip("w:Q", format=".2f", title="width"),
                alt.Tooltip("h:Q", format=".2f", title="height"),
            ],
        )
    )

    labels = (
        alt.Chart(df)
        .mark_text(color="white", fontSize=14, fontWeight="bold")
        .encode(
            x=alt.X("cx:Q", scale=x_scale),
            y=alt.Y("cy:Q", scale=y_scale),
            text="index:N",
        )
    )

    # Strip-outline rectangle: x in [0, W], y in [0, H]. Drawn as a filled
    # mark_rect with no fill so we get a clean rule on all four sides.
    if H is not None and H > 0:
        outline_df = pd.DataFrame([{"x": 0.0, "x_end": W, "y": 0.0, "y_end": H}])
    else:
        outline_df = pd.DataFrame([{"x": 0.0, "x_end": W, "y": 0.0, "y_end": y_top}])
    outline = (
        alt.Chart(outline_df)
        .mark_rect(fill=None, stroke="#dc2626", strokeWidth=2, strokeDash=[6, 4])
        .encode(
            x=alt.X("x:Q", scale=x_scale),
            x2="x_end:Q",
            y=alt.Y("y:Q", scale=y_scale),
            y2="y_end:Q",
        )
    )

    # Pick a chart pixel size that approximately matches the data aspect
    # ratio so rectangles look like rectangles, not ellipses.
    aspect = y_top / W if W > 0 else 1.0
    base_w = 520
    base_h = max(260, min(720, int(base_w * aspect)))
    return (rects_chart + labels + outline).properties(
        width=base_w, height=base_h
    )


def render_optimizer_tab():
    data = st.session_state.data
    optimal = st.session_state.optimal

    if not data["rects"]:
        st.info("Add at least one rectangle on the Data tab.")
        return

    # Instructions sit just below the title block.
    st.markdown(
        "Pack rectangles into a vertical strip of fixed width $W$ to minimize "
        "the strip height $H$. Use the **sidebar** to set $W$, choose the GDP "
        "transformation (Big-M or Hull), and click **Solve** — the layout "
        "below updates with the optimal placement."
    )

    # Headline metrics: strip height H, total rectangle area, packing
    # efficiency (% of strip area covered by rectangles).
    area = total_area(data)
    if optimal and optimal["status"] == "optimal":
        H = float(optimal["H"])
        eff = (area / (data["W"] * H) * 100.0) if (data["W"] > 0 and H > 0) else 0.0
        h_text = f"{H:.3f}"
        eff_text = f"{eff:.1f}%"
    else:
        H = None
        h_text = "—"
        eff_text = "—"

    m1, m2, m3 = st.columns(3)
    m1.metric("Strip height H", h_text)
    m2.metric("Total area", f"{area:.2f}")
    m3.metric("Packing efficiency", eff_text)

    # Solver-status messages (only on non-optimal outcomes)
    if optimal:
        if optimal["status"] == "solver_missing":
            st.error(optimal.get("message", "Solver missing"))
        elif optimal["status"] == "infeasible_data":
            st.error(optimal.get("message", "Infeasible data"))
        elif optimal["status"] == "infeasible":
            st.error("Infeasible — no packing fits these rectangles in the strip.")
        elif optimal["status"] == "unbounded":
            st.error("Unbounded problem.")
        elif optimal["status"] not in ("optimal", "no_rects"):
            st.error(f"Solver returned: {optimal['status']}")

    # Plot the layout. If we have no solution yet, show a placeholder strip
    # outline so the layout area doesn't snap to zero height.
    H_max = float(sum(data["h"][i] for i in data["rects"]))
    if optimal and optimal["status"] == "optimal":
        layout = {"x": optimal["x"], "y": optimal["y"]}
        chart = _packing_chart(data, layout, H, H_max)
        st.altair_chart(chart, use_container_width=False)
        st.caption(
            f"Solved with GDP → MILP via "
            f"**{'Big-M' if optimal['transform'] == 'gdp.bigm' else 'Hull'}** "
            f"transformation. Lower bound on H: "
            f"{lower_bound_H(data):.3f}."
        )
    else:
        st.info(
            "Click **Solve** in the sidebar to compute and visualize the "
            "optimal packing."
        )


def render_data_tab():
    # Editable rectangles table. `num_rows="dynamic"` lets the user add and
    # delete rows freely; we cap the result at MAX_RECTS.
    data = st.session_state.data
    st.subheader(f"Rectangles (max {MAX_RECTS})")

    df = data_to_df(data)
    table_col, _ = st.columns([2, 3])
    with table_col:
        edited = st.data_editor(
            df,
            num_rows="dynamic",
            width="stretch",
            height=(len(df) + 2) * 35 + 3,
            column_config={
                "Index": st.column_config.NumberColumn(
                    "i", disabled=True, format="%d",
                    help="Rectangle index (renumbered automatically).",
                ),
                "Width": st.column_config.NumberColumn(
                    "Width (w_i)", min_value=0.0, format="%.2f",
                ),
                "Height": st.column_config.NumberColumn(
                    "Height (h_i)", min_value=0.0, format="%.2f",
                ),
            },
            key="data_editor",
        )

    # Validate / report on the edited table.
    warnings = []
    if len(edited) > MAX_RECTS:
        warnings.append(
            f"Capped at {MAX_RECTS} rectangles; extra rows ignored "
            "(Big-M MILP gets slow beyond this)."
        )
        edited = edited.head(MAX_RECTS)

    new_data = df_to_data(edited, data["W"])

    # Per-rectangle width and positivity warnings, surfaced from the cleaned
    # data (so they don't fire on rows the user is still editing).
    too_wide = [i for i in new_data["rects"] if new_data["w"][i] > new_data["W"] + 1e-9]
    if too_wide:
        warnings.append(
            f"Rectangle(s) {too_wide} are wider than the strip "
            f"(W = {new_data['W']:g}); the problem will be infeasible."
        )

    raw_invalid = (
        (pd.to_numeric(edited["Width"], errors="coerce") <= 0)
        | (pd.to_numeric(edited["Height"], errors="coerce") <= 0)
    ).any()
    if raw_invalid:
        warnings.append(
            "Rows with non-positive width or height were dropped."
        )

    # If the cleaned data differs from what we had, commit it to state and
    # rerun so other tabs see the change. Invalidate any prior solver result.
    if new_data != st.session_state.data:
        st.session_state.data = new_data
        st.session_state.optimal = None
        st.rerun()

    for w in warnings:
        st.warning(w)

    # Reset uses the deferred-flag pattern documented in `init_state`.
    if st.button("Reset to defaults"):
        st.session_state["_pending_reset"] = True
        st.rerun()


def render_formulation_tab():
    # Two sub-tabs: a static reference formulation (general math) and an
    # instance-specific summary (numbers from the current data).
    sub_general, sub_instance = st.tabs(["General", "Instance"])

    with sub_general:
        st.subheader("General Formulation")
        left, right, _ = st.columns([1, 1, 1])
        with left:
            st.markdown(
                "**Sets**  \n"
                r"$\mathcal{I} = \{1, \ldots, N\}$ rectangles"
            )
            st.markdown(
                "**Parameters**  \n"
                r"$w_i$ width of rectangle $i \in \mathcal{I}$" "  \n"
                r"$h_i$ height of rectangle $i \in \mathcal{I}$" "  \n"
                r"$W$ strip width (fixed)"
            )
            st.markdown(
                "**Variables**  \n"
                r"$x_i, y_i \ge 0$ bottom-left corner of rectangle $i$" "  \n"
                r"$H \ge 0$ strip height (objective)"
            )
        with right:
            # Title + display math in one centered block.
            st.markdown(
                r"""<div style="text-align: center;">

**Objective and Constraints**

$$
\begin{gathered}
\min_{x, y, H} \; H \\
\text{s.t.} \quad x_i + w_i \le W \quad \forall i \in \mathcal{I} \\
y_i + h_i \le H \quad \forall i \in \mathcal{I} \\
x_i, y_i, H \ge 0
\end{gathered}
$$

</div>""",
                unsafe_allow_html=True,
            )

        st.markdown("**Disjunction (no overlap)**")
        st.markdown(
            "For every pair $(i, j)$ with $i < j$, at least one of four "
            "geometric separations must hold — rectangle $i$ must lie to "
            "the left of, right of, below, or above rectangle $j$:"
        )
        st.latex(
            r"""
            \begin{bmatrix} x_i + w_i \le x_j \end{bmatrix}
            \;\vee\;
            \begin{bmatrix} x_j + w_j \le x_i \end{bmatrix}
            \;\vee\;
            \begin{bmatrix} y_i + h_i \le y_j \end{bmatrix}
            \;\vee\;
            \begin{bmatrix} y_j + h_j \le y_i \end{bmatrix}
            \quad \forall i < j
            """
        )

        st.markdown("**GDP → MILP reformulation**")
        st.markdown(
            "- **Big-M**: each disjunct $a^\\top x \\le b$ is replaced with "
            "$a^\\top x \\le b + M(1 - z_k)$, where $z_k \\in \\{0, 1\\}$ is "
            "the disjunct indicator and $M$ is a constant large enough that "
            "the constraint becomes vacuous when $z_k = 0$. Few extra "
            "variables; the LP relaxation is often loose."
        )
        st.markdown(
            "- **Hull (convex hull / disaggregated)**: introduces "
            "disaggregated copies of each variable, one per disjunct, with "
            "scaled bounds. More variables and constraints, but the LP "
            "relaxation is the convex hull of the feasible region — typically "
            "tighter and faster on harder instances."
        )

    with sub_instance:
        st.subheader("Instance Summary")
        data = st.session_state.data
        if not data["rects"]:
            st.info("Add at least one rectangle on the Data tab.")
            return

        N = len(data["rects"])
        area = total_area(data)
        H_lb = lower_bound_H(data)
        n_disj = N * (N - 1) // 2

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"**N (rectangles)** &nbsp; {N}  \n"
                f"**W (strip width)** &nbsp; {data['W']:g}  \n"
                f"**Total area $\\sum_i w_i h_i$** &nbsp; {area:.3f}  \n"
                f"**Lower bound on H** &nbsp; "
                f"$\\max(\\max_i h_i,\\ \\sum_i w_i h_i / W) = {H_lb:.3f}$  \n"
                f"**Disjunctions $N(N-1)/2$** &nbsp; {n_disj}"
            )
        with c2:
            # Render the per-rectangle data as a small LaTeX-style table for
            # consistency with the other apps' formulation views.
            st.markdown("**Per-rectangle data**")
            rows = ["| $i$ | $w_i$ | $h_i$ |", "|---|---|---|"]
            for i in data["rects"]:
                rows.append(f"| {int(i)} | {data['w'][i]:g} | {data['h'][i]:g} |")
            st.markdown("\n".join(rows))


def render_logs_tab():
    optimal = st.session_state.optimal
    if not optimal:
        st.info("Click **Solve** in the sidebar to see solver logs.")
        return

    transform_label = (
        "Big-M (gdp.bigm)" if optimal.get("transform") == "gdp.bigm"
        else "Hull (gdp.hull)"
    )
    st.markdown(f"**GDP transformation used:** {transform_label}")

    log = optimal.get("log", "") or ""
    if not log.strip():
        st.info("No solver output captured for the last run.")
        return
    st.code(log, language="text")


# ---------- Main ----------
#
# Module-level code runs on every Streamlit rerun, so this section needs to
# be cheap and idempotent: configure the page, ensure session_state is set
# up, draw the sidebar, then assemble the four tabs.

st.set_page_config(
    page_title="Strip Packing GDP Optimizer",
    page_icon="favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session_state defaults and apply any pending reset.
init_state()

# Tighten the top of the main block so the title sits closer to the page top
# and the tabs are visible without scrolling. Same value used by the other
# apps; smaller numbers hide the title under Streamlit's sticky header.
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        user-select: none;
        -webkit-user-select: none;
    }
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
    /* Top padding shared across the template family — clears the sticky
       header without clipping the title. See griffith-pse-app-template. */
    .block-container,
    [data-testid="stMainBlockContainer"] {
        padding-top: 2.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Home link: clicking the Griffith PSE logo navigates back to the portfolio
# site. Same-tab navigation since the user is leaving the demo. Lives at the
# top of the sidebar (the upper-left of the page when expanded), matching the
# quad-tank pattern. Image is embedded from the local favicon.png as a base64
# data URL — the link still navigates to griffith-pse.com when clicked, but
# loading the page itself doesn't make any third-party request.
_FAVICON_DATA_URL = "data:image/png;base64," + base64.b64encode(
    (Path(__file__).parent / "favicon.png").read_bytes()
).decode()
st.sidebar.markdown(
    f'<a class="home-logo-corner" href="https://griffith-pse.com" target="_self">'
    f'<img src="{_FAVICON_DATA_URL}" '
        f'alt="Griffith PSE — home" />'
    f'</a>',
    unsafe_allow_html=True,
)

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("Inputs")

# Strip width W. The number_input is bound to `W_input` in session_state so
# `apply_reset` can seed it after a Reset.
W_value = st.sidebar.number_input(
    "Strip width W",
    min_value=0.1,
    value=float(st.session_state.data["W"]),
    step=0.5,
    format="%.2f",
    key="W_input",
)

# GDP transformation choice. Both are one-line code changes inside `solve()`;
# expose the choice as a radio so users can compare them.
transform_label = st.sidebar.radio(
    "GDP transformation",
    options=["Big-M", "Hull"],
    index=0,
    key="transform_radio",
    help=(
        "Big-M: fewer variables, looser LP relaxation. "
        "Hull: more variables but tighter relaxation — often faster on "
        "harder instances."
    ),
)
transform_key = "gdp.bigm" if transform_label == "Big-M" else "gdp.hull"

# Commit any change to W back to the data dict so downstream renders see it.
if abs(float(W_value) - float(st.session_state.data["W"])) > 1e-12:
    new_data = dict(st.session_state.data)
    new_data["W"] = float(W_value)
    st.session_state.data = new_data
    st.session_state.optimal = None

# Solve button. MILP can be slow for many rectangles, so it's explicit rather
# than auto-running on every state change.
if st.sidebar.button("Solve", type="primary", use_container_width=True):
    with st.spinner("Solving GDP-transformed MILP via HiGHS..."):
        st.session_state.optimal = solve(st.session_state.data, transform_key)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='margin: 0 0 0.25rem 0; padding: 0; font-size: 1.5rem; font-weight: 700;'>"
    "Strip Packing GDP Optimizer "
    "<span style='font-size: 1.15rem; font-weight: 400; color: #6b7280;'>"
    "powered by "
    "<a href='https://github.com/ERGO-Code/HiGHS' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>HiGHS</a>"
    "</span>"
    "</h2>",
    unsafe_allow_html=True,
)
_caption_col, _ = st.columns([1, 1])
with _caption_col:
    st.markdown(
        "Pack $N$ rectangles into a vertical strip of fixed width $W$ to "
        "minimize the strip height $H$. The non-overlap conditions are written "
        "as **disjunctions** (`pyomo.gdp`) and reformulated to a MILP with "
        "either the **Big-M** or **Hull** transformation — pick one in the "
        "sidebar and click **Solve**. Edit rectangles in the **Data** tab; "
        "the **Formulation** and **Logs** tabs show the underlying GDP and "
        "solver output."
    )

# Four tabs for the four views of the problem.
optimizer_tab, data_tab, formulation_tab, logs_tab = st.tabs(
    ["📦 Optimizer", "📋 Data", "📐 Formulation", "📜 Logs"]
)

with optimizer_tab:
    render_optimizer_tab()
with data_tab:
    render_data_tab()
with formulation_tab:
    render_formulation_tab()
with logs_tab:
    render_logs_tab()
