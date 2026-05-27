# =============================================================================
# Strip Packing GDP Optimizer — a Streamlit tutorial app.
#
# This file builds an interactive web app around the classic strip-packing
# problem: given N rectangles with widths w_i and lengths ℓ_i, place them
# inside a strip of fixed width W so that no two rectangles overlap and the
# strip's used length L is minimized. The visualization rotates the strip
# 90° clockwise so W is vertical (fixed) and L grows to the right.
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
#
# File roadmap:
#   1. Solver       — model definition, GDP transformation, HiGHS log capture.
#   2. State        — session_state init / reset.
#   3. Utilities    — rectangle add/delete + geometry helpers.
#   4. LaTeX        — render the general formulation and instance summary.
#   5. Tabs         — render_optimizer / render_formulation / render_logs.
#   6. Main         — page config, corner-logo CSS, header/caption, 3 tabs.
# =============================================================================

import base64
import contextlib
import copy
import io
import math
import time
from pathlib import Path

import pyomo.environ as pyo
import streamlit as st
from pyomo.common.errors import ApplicationError
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
    "length": {1: 6.0, 2: 4.0, 3: 2.0, 4: 3.0, 5: 5.0, 6: 3.0, 7: 2.0, 8: 7.0},
    "W": 10.0,
}

# GDP → MILP transformations offered via the radio above the strip on the
# Optimizer tab. Big-M and Hull are the classical pair; mbigm uses a per-
# constraint tight Big-M, midway between the two. All three are
# TransformationFactory entries in pyomo.gdp.
_GDP_TRANSFORMS = {
    "Big-M": "gdp.bigm",
    "Multiple Big-M": "gdp.mbigm",
    "Hull": "gdp.hull",
}
_GDP_LABEL = {v: k for k, v in _GDP_TRANSFORMS.items()}


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

    # Sums used for variable bounds. The strip length L is bounded above by
    # the sum of all rectangle lengths (worst case: stacking all rectangles
    # end-to-end along the strip). Each y_i is bounded by the same sum.
    L_max = float(sum(data["length"][i] for i in rects)) if rects else 1.0
    L_max = max(L_max, 1.0)

    # Index set over rectangles.
    m.RECTS = pyo.Set(initialize=rects, ordered=True)

    # Parameters: known data the solver does not change.
    m.w = pyo.Param(m.RECTS, initialize={i: float(data["w"][i]) for i in rects})
    m.length = pyo.Param(
        m.RECTS, initialize={i: float(data["length"][i]) for i in rects}
    )
    m.W = pyo.Param(initialize=W)

    # Decision variables. Bounded explicitly so Pyomo's gdp.bigm transformation
    # can derive sensible Big-M values automatically. x_i sits in [0, W],
    # y_i in [0, L_max], L itself in [0, L_max].
    m.x = pyo.Var(m.RECTS, domain=pyo.NonNegativeReals, bounds=(0.0, W))
    m.y = pyo.Var(m.RECTS, domain=pyo.NonNegativeReals, bounds=(0.0, L_max))
    m.L = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.0, L_max))

    # Objective: minimize the strip length L.
    m.total_length = pyo.Objective(expr=m.L, sense=pyo.minimize)

    # Containment: each rectangle must fit inside the strip. Width-direction
    # fit is enforced by the upper bound on x plus this constraint; length-
    # direction fit ties each rectangle's far edge to the strip-length L.
    def fit_x_def(m, i):
        return m.x[i] + m.w[i] <= m.W
    m.fit_x = pyo.Constraint(m.RECTS, rule=fit_x_def)

    def fit_y_def(m, i):
        return m.y[i] + m.length[i] <= m.L
    m.fit_y = pyo.Constraint(m.RECTS, rule=fit_y_def)

    # Non-overlap disjunctions: for every unordered pair (i, j) with i < j,
    # at least one of the four geometric separations must hold. `Disjunction`
    # accepts a list of disjuncts, each being a list of constraint expressions.
    pairs = [(i, j) for idx_i, i in enumerate(rects) for j in rects[idx_i + 1:]]
    if pairs:
        m.PAIRS = pyo.Set(initialize=pairs, dimen=2)

        def disj_rule(m, i, j):
            return [
                [m.x[i] + m.w[i] <= m.x[j]],            # i is left of j
                [m.x[j] + m.w[j] <= m.x[i]],            # i is right of j
                [m.y[i] + m.length[i] <= m.y[j]],       # i is below j
                [m.y[j] + m.length[j] <= m.y[i]],       # i is above j
            ]
        m.no_overlap = Disjunction(m.PAIRS, rule=disj_rule)

    return m


# Hard cap on the HiGHS master solve. Large instances (especially with
# Big-M on N close to MAX_RECTS) can take much longer than this in the
# worst case; cutting off at 10 s keeps the UI responsive and surfaces
# the optimality gap when the solver doesn't converge.
SOLVE_TIME_LIMIT_S = 10.0

# Below this relative gap, treat the solve as effectively optimal (the
# default HiGHS MIP gap tolerance is 0.01% = 1e-4; we use a tiny bit
# higher to absorb floating-point noise).
GAP_OPTIMAL_THRESHOLD_PCT = 0.05


def _ensure_pyomo_thread_locals():
    """Workaround for a Pyomo bug. pyomo/gdp/plugins/multiple_bigm.py's
    `_apply_to` opens with `if _thread_local.in_progress: raise ...`
    without first lazy-initializing the attribute. The flag is only ever
    *set* inside the function's try/finally, so on the very first call
    on a given thread the read raises AttributeError. Streamlit rotates
    requests through a worker-thread pool, so fresh workers hit this on
    their first Multiple Big-M solve. Pre-initialize the flag here. The
    try/except keeps us safe if Pyomo's internal module layout changes
    (or upstream fixes the bug and we no longer need the workaround).
    """
    try:
        from pyomo.gdp.plugins import multiple_bigm as _mbigm
        if not hasattr(_mbigm._thread_local, "in_progress"):
            _mbigm._thread_local.in_progress = False
    except Exception:
        pass


def _solve_capturing(m, transform):
    """Apply the GDP transformation, run HiGHS, return
    (results, log_text, elapsed). Captures HiGHS's stdout via
    contextlib.redirect_stdout/stderr — same pattern as knapsack /
    diet / circle-packing. `appsi_highs` routes HiGHS's output through
    Python, so the simpler Python-level redirect catches it without
    going through Pyomo's capture_output. `elapsed` is the wall-clock
    time of transformation + solve, in seconds — shown as a metric on
    the Optimizer tab so users can compare the three GDP
    reformulations head-to-head."""
    # Reformulate the GDP into a standard MILP. Big-M / Multiple Big-M use a
    # linearization with a large constant; Hull adds disaggregated copies of
    # the variables but tends to give tighter relaxations. Multiple Big-M
    # additionally solves LP subproblems to tighten each per-constraint M;
    # we point those subproblems at HiGHS (default is Gurobi, which we don't
    # ship in the image).
    _ensure_pyomo_thread_locals()
    t0 = time.perf_counter()
    apply_kwargs = {"solver": "appsi_highs"} if transform == "gdp.mbigm" else {}
    pyo.TransformationFactory(transform).apply_to(m, **apply_kwargs)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        solver = pyo.SolverFactory("appsi_highs")
        solver.options["time_limit"] = SOLVE_TIME_LIMIT_S
        # When HiGHS hits the time cap without finding any feasible
        # solution (e.g. Hull on N=15), the default solve(load_solutions=
        # True) path raises RuntimeError because there's nothing to load.
        # Disable the auto-load at the solver-call level (the kwarg
        # overrides the solver default for this one call), then manually
        # call m.solutions.load_from(results) below only if the primal
        # bound is finite — i.e. a feasible incumbent actually exists.
        results = solver.solve(m, tee=True, load_solutions=False)
        _load_solution_if_present(m, results)
    log_text = buf.getvalue()
    elapsed = time.perf_counter() - t0
    return results, log_text, elapsed


def _load_solution_if_present(m, results):
    """Best-effort copy of the solver's solution onto the model. Tries
    two paths because `m.solutions.load_from(results)` has shown
    intermittent behavior across Pyomo versions when paired with
    appsi_highs's Results object (works on 6.8.x in local testing,
    appears to no-op on 6.10.x in production).

    Path 1: the legacy `m.solutions.load_from(results)` — works when
    Pyomo's solution adapter understands the appsi_highs Results
    format. Path 2: walk `results.solution[0]['Variable']` directly
    and assign values via `m.find_component(name).value = v`. Both are
    wrapped in try/except so a failure on either path leaves the
    model variables unset rather than raising; the caller's
    `_extract_layout()` then surfaces the `no_incumbent` status."""
    # Path 1: legacy load_from
    try:
        m.solutions.load_from(results)
    except Exception:
        pass
    # If path 1 worked, m.L.value is now populated. Otherwise fall
    # through to the explicit walk.
    if m.L.value is not None:
        return
    # Path 2: manual walk
    try:
        var_dict = results.solution[0]["Variable"]
    except Exception:
        return
    for var_name, val_dict in var_dict.items():
        try:
            var = m.find_component(var_name)
            if var is not None and "Value" in val_dict:
                var.value = val_dict["Value"]
        except Exception:
            continue


def _extract_gap_pct(results):
    """Pull the relative optimality gap (in percent) out of a legacy
    Pyomo Results object. Returns `None` if the bounds aren't both
    available — e.g. if HiGHS hit the time limit without a feasible
    incumbent, or for solver backends that don't populate the
    problem-level bound fields. (The values returned by `problem[k]`
    are plain floats despite the docs sometimes describing them as
    ScalarData wrappers.)"""
    try:
        problem = results.problem[0]
        primal = float(problem["Upper bound"])
        dual = float(problem["Lower bound"])
    except Exception:
        return None
    # Guard against unbounded / degenerate values (Pyomo emits ±inf for
    # missing bounds on some paths).
    if not (math.isfinite(primal) and math.isfinite(dual)):
        return None
    if primal <= 1e-10:
        return None
    return max(0.0, (primal - dual) / primal * 100.0)


def solve(data, transform="gdp.bigm"):
    # Top-level entrypoint used by the UI. Always returns a plain dict so the
    # caller can stash the result in session_state without holding on to a
    # live Pyomo model.

    if not data["rects"]:
        return {"status": "no_rects", "x": {}, "y": {}, "L": None,
                "log": "", "transform": transform, "elapsed": None}

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
            "x": {}, "y": {}, "L": None, "log": "", "transform": transform,
            "elapsed": None,
        }

    m = build_model(data)

    try:
        results, log, elapsed = _solve_capturing(m, transform)
    except ApplicationError as e:
        return {
            "status": "solver_missing",
            "message": (
                "HiGHS solver not available. Run `pip install highspy` "
                f"in your environment. ({e})"
            ),
            "x": {}, "y": {}, "L": None, "log": "", "transform": transform,
            "elapsed": None,
        }

    # Translate Pyomo's TerminationCondition enum into a small set of stable
    # status strings the UI knows how to render. With a time limit set, the
    # solver may return `maxTimeLimit` carrying a best-known feasible
    # incumbent — treated as a "feasible" status so the UI still draws the
    # packing, with the gap surfaced separately.
    tc = results.solver.termination_condition
    gap_pct = _extract_gap_pct(results)

    def _extract_layout():
        """Pull x, y, L off the model. May raise if the solver returned
        no feasible solution (variables still hold their initial values
        or are stale)."""
        x = {i: float(pyo.value(m.x[i])) for i in data["rects"]}
        y = {i: float(pyo.value(m.y[i])) for i in data["rects"]}
        L = float(pyo.value(m.L))
        return x, y, L

    if tc == TerminationCondition.optimal:
        # Even on optimal termination, _extract_layout can raise if
        # m.solutions.load_from(results) silently failed upstream (the
        # appsi_highs <-> legacy Results adapter has shipped buggy
        # versions where the load step is a no-op). Catch that here so
        # the page surfaces a "no incumbent" warning instead of
        # crashing with ValueError.
        try:
            x, y, L = _extract_layout()
        except Exception:
            return {"status": "no_incumbent", "x": {}, "y": {}, "L": None,
                    "gap_pct": gap_pct, "log": log, "transform": transform,
                    "elapsed": elapsed}
        return {
            "status": "optimal",
            "x": x, "y": y, "L": L, "gap_pct": gap_pct,
            "log": log, "transform": transform, "elapsed": elapsed,
        }
    if tc == TerminationCondition.maxTimeLimit:
        # Feasible incumbent expected (the LP root is always feasible for
        # this problem). If the bounds tell us a finite primal exists,
        # extract the layout and surface the gap.
        try:
            x, y, L = _extract_layout()
        except Exception:
            return {"status": "no_incumbent", "x": {}, "y": {}, "L": None,
                    "gap_pct": gap_pct, "log": log, "transform": transform,
                    "elapsed": elapsed}
        return {
            "status": "time_limit",
            "x": x, "y": y, "L": L, "gap_pct": gap_pct,
            "log": log, "transform": transform, "elapsed": elapsed,
        }
    if tc in (
        TerminationCondition.infeasible,
        TerminationCondition.infeasibleOrUnbounded,
    ):
        return {"status": "infeasible", "x": {}, "y": {}, "L": None,
                "gap_pct": None,
                "log": log, "transform": transform, "elapsed": elapsed}
    if tc == TerminationCondition.unbounded:
        return {"status": "unbounded", "x": {}, "y": {}, "L": None,
                "gap_pct": None,
                "log": log, "transform": transform, "elapsed": elapsed}
    return {"status": str(tc), "x": {}, "y": {}, "L": None,
            "gap_pct": gap_pct,
            "log": log, "transform": transform, "elapsed": elapsed}


# ---------- State ----------
#
# Streamlit re-executes the whole script on every interaction. Anything that
# must persist between runs lives in `st.session_state`. The keys we use:
#   - data:                the current problem instance (rects, w, length, W)
#   - optimal:             the most recent solver result, or None
#   - _pending_reset:      one-shot flag to reset on the next run
#   - W_input:             value backing the inline strip-width number_input
#   - transform_radio:     value backing the inline GDP-transformation radio
#   - _rect_editor_ver:    counter bumped on Reset so per-rectangle stepper
#                          widget keys re-init instead of holding stale state
#   - w_{rid}_{ver} / l_{rid}_{ver} / del_{rid}_{ver}:
#                          per-row stepper / delete-button widget keys

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
    # Bump the rect-editor widget version so all per-rectangle steppers
    # re-init from data instead of holding onto sticky pre-reset values.
    st.session_state["_rect_editor_ver"] = (
        st.session_state.get("_rect_editor_ver", 0) + 1
    )


# ---------- Utilities ----------
#
# Rectangles are tracked by stable opaque integer ids — `rects` is a list
# of ids, `w` and `length` map id → value. Ids don't renumber on delete:
# this keeps widget state in the editor below from getting reassigned to a
# different rectangle whenever one is removed.

def add_rect(data, w=1.0, length=1.0):
    """Append a rectangle with a fresh id. Mutates `data` and returns it."""
    new_id = (max(data["rects"]) + 1) if data["rects"] else 1
    data["rects"] = list(data["rects"]) + [new_id]
    data["w"] = dict(data["w"]); data["w"][new_id] = float(w)
    data["length"] = dict(data["length"]); data["length"][new_id] = float(length)
    return data


def remove_rect(data, rid):
    """Drop the rectangle with id `rid`. Mutates `data` and returns it."""
    data["rects"] = [i for i in data["rects"] if i != rid]
    data["w"] = {i: v for i, v in data["w"].items() if i != rid}
    data["length"] = {i: v for i, v in data["length"].items() if i != rid}
    return data


def total_area(data):
    return sum(
        float(data["w"][i]) * float(data["length"][i]) for i in data["rects"]
    )


def lower_bound_L(data):
    # A trivial lower bound on the strip length L: the longer of (a) the
    # longest rectangle and (b) total area divided by strip width.
    if not data["rects"]:
        return 0.0
    max_length = max(float(data["length"][i]) for i in data["rects"])
    area_lb = total_area(data) / float(data["W"]) if data["W"] > 0 else 0.0
    return max(max_length, area_lb)


def naive_layout(data):
    """Cascade layout — all rectangles at x=0, stacked end-to-end along
    the L direction. This is the worst-case feasible packing and serves
    as a no-solver default visualization."""
    layout = {"x": {}, "y": {}}
    cumulative = 0.0
    for i in data["rects"]:
        layout["x"][i] = 0.0
        layout["y"][i] = cumulative
        cumulative += float(data["length"][i])
    return layout


# ---------- Tabs ----------
#
# One render_* function per tab. Optimizer is the main view (rectangle
# editor on the left, strip + controls + metrics on the right); Formulation
# shows the math; Logs shows HiGHS output.

# A 12-color categorical palette repeated as needed. Tableau-style; reads
# well at small rectangle sizes and keeps adjacent indices distinguishable.
_PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B",
    "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#1F77B4", "#9467BD",
]


def _render_top_metric(slot, label, value, suffix_html=""):
    """Render a metric-shaped block via raw HTML. Mirrors the
    `colored_metric` helper in the diet app: a small gray label on
    top, large value below, with an optional HTML suffix appended
    inside the value div (used to drop a red ⚠ glyph next to "Best
    length" when the solver didn't prove optimality).

    All five top-row metrics use this helper so they're styled
    identically; mixing st.metric with custom HTML produced visible
    alignment / font-size mismatches in earlier attempts. Wrapper
    margin matches diet's colored_metric (top 0.25rem / bottom 1rem)
    so the value has enough breathing room below — without it, the
    strip's negative-margin layout crowds up against the value text."""
    slot.markdown(
        "<div style='margin:0.25rem 0 1rem 0; line-height:1.2;'>"
        "<div style='font-size:0.875rem; color:rgba(49,51,63,0.6); "
        "margin-bottom:0.25rem;'>"
        f"{label}"
        "</div>"
        "<div style='font-size:2.25rem; font-weight:400; line-height:1.2;'>"
        f"{value}{suffix_html}"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_optimizer_strip(data, layout, L, x_top):
    """Render the optimizer's layout as absolutely-positioned HTML divs
    inside a responsive container. The container's width fills its parent
    column (100%) and its aspect ratio is locked to x_top:W via the
    `aspect-ratio` CSS property, so percentage-based child positions stay
    geometrically accurate as the column resizes."""
    rects = data["rects"]
    W = float(data["W"])
    if x_top <= 0 or W <= 0:
        st.markdown(
            '<div style="width:100%;aspect-ratio:4/1;background:#f4f6fa;">'
            "</div>",
            unsafe_allow_html=True,
        )
        return

    rect_divs = []
    has_layout = layout is not None and layout.get("x")
    for i in rects:
        if not has_layout:
            continue
        x = float(layout["x"][i])
        y = float(layout["y"][i])
        w = float(data["w"][i])
        length = float(data["length"][i])
        # 90° CW mapping: container x = orig y (along L), container y =
        # orig x (along W). Positions and sizes are expressed as
        # percentages of x_top (horizontal) and W (vertical) so the
        # container can resize freely.
        left_pct = (y / x_top) * 100.0
        top_pct = (x / W) * 100.0
        width_pct = (length / x_top) * 100.0
        height_pct = (w / W) * 100.0
        color = _PALETTE[(int(i) - 1) % len(_PALETTE)]
        rect_divs.append(
            f'<div style="position:absolute;'
            f'left:{left_pct:.4f}%;top:{top_pct:.4f}%;'
            f'width:{width_pct:.4f}%;height:{height_pct:.4f}%;'
            f'background:{color};'
            f'border:2px solid #ffffff;box-sizing:border-box;'
            f'color:#ffffff;font-weight:700;font-size:14px;'
            f'display:flex;align-items:center;justify-content:center;">'
            f"{int(i)}</div>"
        )
    # Dashed red strip outline runs from x=0 to x=L if solved, otherwise
    # to x_top so the area doesn't collapse on a fresh page.
    outline_w_units = L if (L is not None and L > 0) else x_top
    outline_w_pct = (outline_w_units / x_top) * 100.0
    outline_div = (
        f'<div style="position:absolute;left:0;top:0;'
        f'width:{outline_w_pct:.4f}%;height:100%;'
        f'border:2px dashed #dc2626;box-sizing:border-box;'
        f'pointer-events:none;"></div>'
    )
    container = (
        f'<div style="position:relative;width:100%;'
        f'aspect-ratio:{x_top} / {W};background:#f4f6fa;">'
        f'{outline_div}{"".join(rect_divs)}</div>'
    )
    st.markdown(container, unsafe_allow_html=True)


def render_optimizer_tab():
    # Page-wide CSS for the editor steppers (tight spacing, right-aligned
    # numbers next to the +/- buttons). Applied globally — the only place
    # with many stacked horizontal blocks is the editor; other sections are
    # minimally affected.
    st.markdown(
        """
        <style>
        [data-testid="stMainBlockContainer"]
            [data-testid="stHorizontalBlock"] {
            margin-bottom: -0.75rem;
        }
        [data-testid="stNumberInputContainer"] input {
            padding-top: 0.25rem; padding-bottom: 0.25rem;
            text-align: right; padding-right: 0.4rem;
        }
        /* Red ⚠ glyph next to "Best length" when the solver didn't
           prove optimality within the time cap. Hovering the glyph
           reveals a black tooltip bubble with the time-cap
           explanation. Same pattern diet / knapsack use for their
           constraint-violation marks. */
        .strip-violation-icon {
            position: relative;
            display: inline-block;
        }
        .strip-violation-icon:hover::after {
            content: attr(data-violation-tooltip);
            position: absolute;
            top: 100%;
            left: 0;
            margin-top: 0.25rem;
            background: #000;
            color: #fff;
            padding: 0.5rem 0.75rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-family: inherit;
            font-weight: 400;
            line-height: 1.2;
            max-width: 22rem;
            white-space: normal;
            z-index: 1000;
            pointer-events: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    data = st.session_state.data
    optimal = st.session_state.optimal

    # Feasibility floor on W: must be ≥ widest rectangle. Computed before
    # the columns are laid out so we can clamp data["W"] if a rectangle
    # edit bumped the floor above the current strip width.
    rects = data["rects"]
    min_W = (
        max(float(data["w"][i]) for i in rects)
        if rects else 1.0
    )
    current_W = float(data["W"])
    if current_W < min_W:
        current_W = min_W
        data = dict(data); data["W"] = min_W
        st.session_state.data = data
        st.session_state["W_input"] = min_W

    # ── Two-column layout: editor (left) | strip column fills remainder ──
    # The metric column is gone — metrics now live in a sub-row above the
    # strip, so the strip itself can stretch to the right edge of the page.
    editor_col, strip_col = st.columns([3, 9])

    # Render the editor in the left column (its edits commit + rerun on
    # change, so we don't need a placeholder for it).
    with editor_col:
        _render_rect_editor(data)

    # Right column: controls + metrics in one row above the strip. Solve
    # sits on the far left, then W, then the transformation radio, then
    # the four metric slots. The strip below renders through a placeholder
    # so the controls in the top row can update session_state before we
    # paint — avoids a stale-render lag.
    with strip_col:
        # Six metric slots after the controls: title room shrinks one
        # column-unit to make room for Gap. Order matches the user's
        # mental flow: bounds → result → quality → optimality → time.
        top_row = st.columns(
            [1, 1, 3, 1, 1, 1, 1, 1],
            vertical_alignment="bottom",
        )
        with top_row[0]:
            solve_clicked = st.button(
                "Solve", type="primary", use_container_width=True,
            )
        with top_row[1]:
            W_value = st.number_input(
                "Strip width W",
                min_value=min_W,
                max_value=30.0,
                value=current_W,
                step=1.0,
                format="%g",
                key="W_input",
                help=(
                    "Strip width is bounded below by the widest rectangle "
                    f"(currently {min_W:g}) — narrower would be infeasible."
                ),
            )
        with top_row[2]:
            transform_label = st.radio(
                "GDP transformation",
                options=list(_GDP_TRANSFORMS.keys()),
                index=0,
                horizontal=True,
                key="transform_radio",
            )
        ub_slot = top_row[3].empty()
        opt_slot = top_row[4].empty()
        eff_slot = top_row[5].empty()
        gap_slot = top_row[6].empty()
        time_slot = top_row[7].empty()
        strip_slot = st.empty()

    transform_key = _GDP_TRANSFORMS[transform_label]

    # Commit any change to W back into data before painting the strip.
    if abs(float(W_value) - float(data["W"])) > 1e-12:
        data = dict(data); data["W"] = float(W_value)
        st.session_state.data = data
        st.session_state.optimal = None
        optimal = None

    # Run Solve if requested. solve() returns a result dict; storing it
    # flips session_state.optimal so the strip below renders the optimum.
    # The spinner is rendered inside strip_slot so it shows in the strip
    # area (which is empty while we solve), not below the editor.
    if solve_clicked:
        with strip_slot.container():
            with st.spinner("Solving GDP-transformed MILP via HiGHS..."):
                result = solve(data, transform_key)
        st.session_state.optimal = result
        optimal = result

    # ── Fill the top-row slots with the now-current state ──────────────────
    if not data["rects"]:
        strip_slot.info("Add at least one rectangle on the left to compute a packing.")
        return

    L_max = float(sum(data["length"][i] for i in data["rects"])) or 1.0
    # A result counts as "renderable" if HiGHS returned an incumbent —
    # either a proven optimum, or a best-feasible found when the time
    # limit fired. Both populate x / y / L on the result dict.
    has_incumbent = bool(
        optimal and optimal["status"] in ("optimal", "time_limit")
    )
    opt_L = float(optimal["L"]) if has_incumbent else None
    gap_pct = optimal.get("gap_pct") if optimal else None
    # "Proved optimal" drives the metric label: HiGHS marked it optimal,
    # OR the bound gap is below the noise floor (e.g. solver terminated
    # at time-limit but already had a near-zero gap).
    proved_optimal = bool(
        optimal
        and (
            optimal["status"] == "optimal"
            or (gap_pct is not None and gap_pct < GAP_OPTIMAL_THRESHOLD_PCT)
        )
    )
    x_top = max(opt_L or 0.0, L_max, 12.0)
    area = total_area(data)
    W = float(data["W"])

    # Strip visualization tracks the latest packing: naive cascade until
    # the user solves, then the incumbent (proven optimum or
    # best-feasible from a time-limited run). Metric labels use stable
    # text — "Upper bound" / "Optimal length" or "Best length" /
    # "Efficiency" / "Gap" / "Solve time" — and show "—" for
    # solve-dependent values until then.
    display_layout = (
        {"x": optimal["x"], "y": optimal["y"]} if has_incumbent
        else naive_layout(data)
    )
    display_L = opt_L if has_incumbent else L_max
    eff_pct = (
        (area / (W * opt_L) * 100.0)
        if (has_incumbent and W > 0 and opt_L and opt_L > 0)
        else None
    )

    with strip_slot.container():
        _render_optimizer_strip(data, display_layout, display_L, x_top)
        # Solver-status messages (only on terminal-failure outcomes).
        # We DON'T render a status caption for "time_limit" — the Gap
        # metric tells that story without nagging under the strip.
        if optimal:
            if optimal["status"] == "solver_missing":
                st.error(optimal.get("message", "Solver missing"))
            elif optimal["status"] == "infeasible_data":
                st.error(optimal.get("message", "Infeasible data"))
            elif optimal["status"] == "infeasible":
                st.error(
                    "Infeasible — no packing fits these rectangles in the strip."
                )
            elif optimal["status"] == "unbounded":
                st.error("Unbounded problem.")
            elif optimal["status"] == "no_incumbent":
                st.warning(
                    f"Hit the {SOLVE_TIME_LIMIT_S:g} s solve cap before "
                    "finding any feasible packing. Try a smaller instance "
                    "or a different transformation."
                )
            elif optimal["status"] not in ("optimal", "time_limit", "no_rects"):
                st.error(f"Solver returned: {optimal['status']}")
    elapsed = optimal.get("elapsed") if optimal else None

    # All 5 metrics in the top row are rendered via the same custom
    # HTML helper so they're visually consistent with each other AND
    # so the "Best length" case can drop a red ⚠ glyph in cleanly as
    # `suffix_html`. This is the same pattern the diet / knapsack apps
    # use; mixing st.metric with custom HTML for one slot is what
    # caused the earlier alignment / styling issues.
    _render_top_metric(ub_slot, "Upper bound", f"{L_max:.0f}")
    if proved_optimal:
        _render_top_metric(opt_slot, "Optimal length", f"{opt_L:.0f}")
    elif has_incumbent:
        tooltip = (
            f"Solver hit the {SOLVE_TIME_LIMIT_S:g} s time cap before "
            "proving optimality. This is the best feasible packing "
            "found so far; see Gap for how far it could still tighten."
        )
        violation_icon = (
            '<span class="strip-violation-icon" '
            f'data-violation-tooltip="{tooltip}" '
            'style="color:#dc2626; cursor:default; font-weight:700; '
            'margin-left:0.4em; vertical-align:baseline;">⚠</span>'
        )
        _render_top_metric(
            opt_slot, "Best length", f"{opt_L:.0f}",
            suffix_html=violation_icon,
        )
    else:
        _render_top_metric(opt_slot, "Optimal length", "—")
    _render_top_metric(
        eff_slot, "Efficiency",
        f"{eff_pct:.1f}%" if eff_pct is not None else "—",
    )
    if gap_pct is None:
        _render_top_metric(gap_slot, "Gap", "—")
    elif gap_pct < GAP_OPTIMAL_THRESHOLD_PCT:
        _render_top_metric(gap_slot, "Gap", "0.0%")
    else:
        _render_top_metric(gap_slot, "Gap", f"{gap_pct:.1f}%")
    _render_top_metric(
        time_slot, "Solve time",
        f"{elapsed:.2f} s" if isinstance(elapsed, (int, float)) else "—",
    )


def _render_rect_editor(data):
    """The rectangles editor — one row per rectangle with stepper inputs
    and a delete button. Used in the left column of the Optimizer tab."""
    st.markdown(f"#### Rectangles (max {MAX_RECTS})")

    ver = st.session_state.get("_rect_editor_ver", 0)
    _W = float(data["W"])
    _editor_cols = [0.6, 1.6, 1.6, 0.6]

    header = st.columns(_editor_cols)
    header[0].markdown("")
    header[1].markdown("**Width**")
    header[2].markdown("**Length**")
    header[3].markdown("")

    new_data = None
    for idx, rid in enumerate(data["rects"], start=1):
        cols = st.columns(_editor_cols, vertical_alignment="center")
        color = _PALETTE[(int(idx) - 1) % len(_PALETTE)]
        cols[0].markdown(
            f'<div style="display:inline-flex;align-items:center;'
            f'justify-content:center;width:1.6rem;height:1.6rem;'
            f'border-radius:0.3rem;background:{color};color:#fff;'
            f'font-weight:700;font-size:0.85rem;">{idx}</div>',
            unsafe_allow_html=True,
        )
        new_w = cols[1].number_input(
            "Width", min_value=1.0, max_value=_W, step=1.0, format="%g",
            value=float(data["w"][rid]),
            key=f"w_{rid}_{ver}",
            label_visibility="collapsed",
        )
        new_l = cols[2].number_input(
            "Length", min_value=1.0, max_value=30.0, step=1.0, format="%g",
            value=float(data["length"][rid]),
            key=f"l_{rid}_{ver}",
            label_visibility="collapsed",
        )
        delete_clicked = cols[3].button(
            "🗑", key=f"del_{rid}_{ver}",
        )
        if delete_clicked:
            st.session_state.data = remove_rect(dict(data), rid)
            st.session_state.optimal = None
            st.rerun()
        if new_w != data["w"][rid] or new_l != data["length"][rid]:
            new_data = dict(data)
            new_data["w"] = dict(new_data["w"]); new_data["w"][rid] = new_w
            new_data["length"] = dict(new_data["length"]); new_data["length"][rid] = new_l

    if new_data is not None:
        st.session_state.data = new_data
        st.session_state.optimal = None
        st.rerun()

    can_add = len(data["rects"]) < MAX_RECTS
    # Buttons share the editor's column structure so Add rectangle aligns
    # with the Width column and Reset to defaults aligns with Length.
    btn_cols = st.columns(_editor_cols)
    with btn_cols[1]:
        if st.button(
            "➕ Add rectangle",
            key="rects_add",
            disabled=not can_add,
            help=(
                None
                if can_add
                else f"Max {MAX_RECTS} rectangles (Big-M MILP gets slow beyond this)."
            ),
        ):
            st.session_state.data = add_rect(dict(data))
            st.session_state.optimal = None
            st.rerun()
    with btn_cols[2]:
        if st.button(
            "Reset to defaults",
            key="rects_reset",
            help="Restore the default instance.",
        ):
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
                r"$\ell_i$ length of rectangle $i \in \mathcal{I}$" "  \n"
                r"$W$ strip width (fixed)"
            )
            st.markdown(
                "**Variables**  \n"
                r"$x_i, y_i \ge 0$ near corner of rectangle $i$" "  \n"
                r"$L \ge 0$ strip length (objective)"
            )
        with right:
            # Title + display math in one centered block.
            st.markdown(
                r"""<div style="text-align: center;">

**Objective and Constraints**

$$
\begin{gathered}
\min_{x, y, L} \; L \\
\text{s.t.} \quad x_i + w_i \le W \quad \forall i \in \mathcal{I} \\
y_i + \ell_i \le L \quad \forall i \in \mathcal{I} \\
x_i, y_i, L \ge 0
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
            \begin{bmatrix} y_i + \ell_i \le y_j \end{bmatrix}
            \;\vee\;
            \begin{bmatrix} y_j + \ell_j \le y_i \end{bmatrix}
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
            "- **Multiple Big-M**: the same shape as Big-M, but each "
            "constraint gets its own per-constraint $M$ rather than sharing "
            "one large global constant. Each $M$ is computed tight to that "
            "constraint from the variable bounds, so the LP relaxation is "
            "tighter than single-$M$ without growing the variable count the "
            "way Hull does — usually a midpoint between the two."
        )
        st.markdown(
            "- **Hull (convex hull / disaggregated)**: introduces "
            "disaggregated copies of each variable, one per disjunct, with "
            "scaled bounds. More variables and constraints, but the LP "
            "relaxation is the convex hull of the feasible region — typically "
            "tighter and faster on harder instances."
        )

        st.markdown("**Solution method**")
        st.markdown(
            "Once the GDP is reformulated to a MILP, HiGHS solves it via "
            "branch-and-bound: relax the binary disjunct indicators $z_k$ to "
            "$[0, 1]$, solve the resulting LP, and either accept the solution "
            "if all indicators are integer or branch on the most fractional "
            "one. The solver is capped at 10 s — if it can't prove optimality "
            "in that time, the Optimizer tab labels the result **Best length** "
            "instead of *Optimal length* and surfaces the remaining "
            "optimality **Gap**. HiGHS is a modern open-source LP/MILP solver "
            "from Edinburgh's ERGO group, distributed as a pip wheel via "
            "`highspy`."
        )
        st.markdown(
            "See the [companion Jupyter notebook]"
            "(https://github.com/devin-griff/strip-packing/blob/main/Strip%20packing.ipynb) "
            "for the Pyomo implementation."
        )

        st.markdown("**References**")
        st.markdown(
            "[1] Q. Chen, E. S. Johnson, D. E. Bernal, R. Valentin, S. Kale, "
            "J. Bates, J. D. Siirola, and I. E. Grossmann, "
            '"Pyomo.GDP: an ecosystem for logic based modeling and '
            'optimization development," *Optimization and Engineering*, '
            "vol. 23, no. 1, pp. 607–642, 2022. "
            "[Springer](https://link.springer.com/article/10.1007/s11081-021-09601-7)"
        )
        st.markdown(
            "[2] R. Raman and I. E. Grossmann, "
            '"Modelling and computational techniques for logic based integer '
            'programming," *Computers & Chemical Engineering*, vol. 18, '
            "no. 7, pp. 563–578, 1994. "
            "[ScienceDirect](https://www.sciencedirect.com/science/article/pii/0098135493E00107)"
        )
        st.markdown(
            "[3] P. M. Castro and I. E. Grossmann, "
            '"Generalized Disjunctive Programming as a Systematic Modeling '
            'Framework to Derive Scheduling Formulations," *Industrial & '
            "Engineering Chemistry Research*, vol. 51, no. 16, pp. 5781–5792, "
            "2012. [ACS](https://pubs.acs.org/doi/10.1021/ie2030486)"
        )
        st.markdown(
            "[4] Q. Huangfu and J. A. J. Hall, "
            '"Parallelizing the dual revised simplex method," *Mathematical '
            "Programming Computation*, vol. 10, no. 1, pp. 119–142, 2018. "
            "[Springer](https://link.springer.com/article/10.1007/s12532-017-0130-5)"
        )
        st.markdown(
            "[5] M. L. Bynum, G. A. Hackebeil, W. E. Hart, C. D. Laird, "
            "B. L. Nicholson, J. D. Siirola, J.-P. Watson, and D. L. Woodruff, "
            "*Pyomo — Optimization Modeling in Python*, 3rd ed. "
            "Cham: Springer, 2021. "
            "[Springer](https://link.springer.com/book/10.1007/978-3-030-68928-5)"
        )

    with sub_instance:
        st.subheader("Instance Summary")
        data = st.session_state.data
        if not data["rects"]:
            st.info("Add at least one rectangle on the Optimizer tab.")
            return

        N = len(data["rects"])
        area = total_area(data)
        L_lb = lower_bound_L(data)
        n_disj = N * (N - 1) // 2

        st.markdown(
            f"**N (rectangles)** &nbsp; {N}  \n"
            f"**W (strip width)** &nbsp; {data['W']:g}  \n"
            f"**Total area $\\sum_i w_i \\ell_i$** &nbsp; {area:.3f}  \n"
            f"**Lower bound on L** &nbsp; "
            f"$\\max(\\max_i \\ell_i,\\ \\sum_i w_i \\ell_i / W) = {L_lb:.3f}$  \n"
            f"**Disjunctions $N(N-1)/2$** &nbsp; {n_disj}"
        )

        if N >= 2:
            # Worked disjunction for the first pair (i, j) with this
            # instance's actual widths/lengths substituted in. The General
            # sub-tab carries the abstract template; here we ground it in
            # the user's data so the four geometric separations are
            # concrete numbers rather than symbols.
            rects = data["rects"]
            i_, j_ = int(rects[0]), int(rects[1])
            wi = float(data["w"][rects[0]])
            li = float(data["length"][rects[0]])
            wj = float(data["w"][rects[1]])
            lj = float(data["length"][rects[1]])
            st.markdown("---")
            st.markdown(
                rf"**Disjunction (instantiated)** &nbsp; for the pair "
                rf"$({i_},\,{j_})$ with "
                rf"$w_{i_}={wi:g},\ \ell_{i_}={li:g},\ "
                rf"w_{j_}={wj:g},\ \ell_{j_}={lj:g}$, "
                "at least one of these four separations must hold:"
            )
            st.latex(
                r"\begin{array}{rl}"
                rf"& \underbrace{{x_{i_} + {wi:g} \le x_{j_}}}_{{{i_}\text{{ left of }}{j_}}} \\"
                rf"\vee & \underbrace{{x_{j_} + {wj:g} \le x_{i_}}}_{{{j_}\text{{ left of }}{i_}}} \\"
                rf"\vee & \underbrace{{y_{i_} + {li:g} \le y_{j_}}}_{{{i_}\text{{ below }}{j_}}} \\"
                rf"\vee & \underbrace{{y_{j_} + {lj:g} \le y_{i_}}}_{{{j_}\text{{ below }}{i_}}}"
                r"\end{array}"
            )
            st.caption(
                f"This is one of the {n_disj} pairwise disjunctions in the "
                f"model. The GDP transformation rewrites each one into a "
                f"set of standard MILP constraints (Big-M / mbigm / Hull)."
            )


def render_logs_tab():
    optimal = st.session_state.optimal
    if not optimal:
        st.info("Click **Solve** on the Optimizer tab to see solver logs.")
        return

    tx = optimal.get("transform")
    transform_label = (
        f"{_GDP_LABEL[tx]} ({tx})" if tx in _GDP_LABEL else str(tx)
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
# up, inject the fixed-corner home-logo CSS, render the header/caption,
# then assemble the three tabs.

st.set_page_config(
    page_title="Strip Packing GDP Optimizer",
    page_icon="favicon.png",
    layout="wide",
)

# Initialize session_state defaults and apply any pending reset.
init_state()

# Fixed-corner home logo (no sidebar in this app — all controls are inline
# on the Optimizer tab). Same pattern as the diet and knapsack apps.
_FAVICON_DATA_URL = "data:image/png;base64," + base64.b64encode(
    (Path(__file__).parent / "favicon.png").read_bytes()
).decode()
st.markdown(
    """
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
    /* Top padding shared across the template family — clears the sticky
       header without clipping the title. See griffith-pse-app-template. */
    .block-container,
    [data-testid="stMainBlockContainer"] {
        padding-top: 2.5rem !important;
    }
    </style>
    """
    f'<a href="https://griffith-pse.com" target="_self" '
    f'class="home-logo-corner">'
    f'<img src="{_FAVICON_DATA_URL}" alt="Griffith PSE — home" />'
    f"</a>",
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='margin: 0 0 0.25rem 0; padding: 0; font-size: 1.5rem; font-weight: 700;'>"
    "Strip Packing GDP Optimizer "
    "<span style='font-size: 1.15rem; font-weight: 400; color: #6b7280;'>"
    "powered by "
    "<a href='https://github.com/Pyomo/pyomo' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>Pyomo</a>"
    " + "
    "<a href='https://github.com/ERGO-Code/HiGHS' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>HiGHS</a>"
    "</span>"
    "</h2>",
    unsafe_allow_html=True,
)
_caption_col, _ = st.columns([5, 3])
with _caption_col:
    st.markdown(
        "Pack $N$ rectangles into a strip of fixed width $W$ to minimize "
        "the strip length $L$. Edit the rectangle list directly on the "
        "Optimizer tab, pick a GDP transformation (Big-M, Multiple Big-M, "
        "Hull) below, and click **Solve**. "
        "Non-overlap is written as **disjunctions** (`pyomo.gdp`) and "
        "reformulated to a MILP for HiGHS, capped at **10 s** of solve "
        "time — if the solver doesn't prove optimality in that window, "
        "the best feasible packing is shown alongside the remaining "
        "**Gap**. The **Formulation** and **Logs** tabs show the "
        "underlying GDP and solver output."
    )

# Three tabs for the three views of the problem.
optimizer_tab, formulation_tab, logs_tab = st.tabs(
    ["📦 Optimizer", "📐 Formulation", "📜 Logs"]
)

with optimizer_tab:
    render_optimizer_tab()
with formulation_tab:
    render_formulation_tab()
with logs_tab:
    render_logs_tab()
