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
import time
from pathlib import Path

import altair as alt
import pandas as pd
import pyomo.environ as pyo
import streamlit as st
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.gdp import Disjunction
from pyomo.opt import TerminationCondition
from streamlit_drawable_canvas import st_canvas


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

# GDP → MILP transformations exposed in the sidebar. Big-M and Hull are the
# classical pair; mbigm uses a per-constraint tight Big-M and Cutting Plane
# iteratively tightens a Big-M base with violated facets of the hull. All
# four are TransformationFactory entries in pyomo.gdp.
_GDP_TRANSFORMS = {
    "Big-M": "gdp.bigm",
    "Hull": "gdp.hull",
    "Multiple Big-M": "gdp.mbigm",
    "Cutting Plane": "gdp.cuttingplane",
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


def _solve_capturing(m, transform):
    """Apply the GDP transformation, run HiGHS, return
    (results, log_text, elapsed). Captures HiGHS's stdout via Pyomo's
    capture_output (FD-level redirect on newer Pyomo, plain stdout capture
    on older). HiGHS via the appsi_highs LegacySolver doesn't support a
    logfile= kwarg, so the FD capture is the only path. `elapsed` is the
    wall-clock time of transformation + solve, in seconds — shown as a
    metric on the Optimizer tab so users can compare the four GDP
    reformulations head-to-head."""
    # Reformulate the GDP into a standard MILP. Big-M / Multiple Big-M use a
    # linearization with a large constant; Hull adds disaggregated copies of
    # the variables but tends to give tighter relaxations; Cutting Plane
    # iteratively adds violated facets of the hull to a Big-M base.
    t0 = time.perf_counter()
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
    elapsed = time.perf_counter() - t0
    return results, log_text, elapsed


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
    # status strings the UI knows how to render.
    tc = results.solver.termination_condition
    if tc == TerminationCondition.optimal:
        x = {i: float(pyo.value(m.x[i])) for i in data["rects"]}
        y = {i: float(pyo.value(m.y[i])) for i in data["rects"]}
        L = float(pyo.value(m.L))
        return {
            "status": "optimal",
            "x": x, "y": y, "L": L,
            "log": log, "transform": transform, "elapsed": elapsed,
        }
    if tc in (
        TerminationCondition.infeasible,
        TerminationCondition.infeasibleOrUnbounded,
    ):
        return {"status": "infeasible", "x": {}, "y": {}, "L": None,
                "log": log, "transform": transform, "elapsed": elapsed}
    if tc == TerminationCondition.unbounded:
        return {"status": "unbounded", "x": {}, "y": {}, "L": None,
                "log": log, "transform": transform, "elapsed": elapsed}
    return {"status": str(tc), "x": {}, "y": {}, "L": None,
            "log": log, "transform": transform, "elapsed": elapsed}


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
        {"Index": int(i), "Width": float(data["w"][i]),
         "Length": float(data["length"][i])}
        for i in data["rects"]
    ])


def df_to_data(df, W):
    # DataFrame -> internal. Normalizes whatever the user typed: drop blank
    # rows, coerce numerics, clamp to non-negative. Indices are always
    # renumbered 1..N so the user can delete rows freely without leaving gaps.
    df = df.copy()
    for col in ["Width", "Length"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Width", "Length"])
    df["Width"] = df["Width"].clip(lower=0.0)
    df["Length"] = df["Length"].clip(lower=0.0)
    df = df[(df["Width"] > 0) & (df["Length"] > 0)]
    rects = list(range(1, len(df) + 1))
    w = {i: float(row.Width) for i, row in zip(rects, df.itertuples())}
    length = {i: float(row.Length) for i, row in zip(rects, df.itertuples())}
    return {"rects": rects, "w": w, "length": length, "W": float(W)}


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


# Pixel budget for both strips (user's drawable canvas and the optimizer's
# Altair chart). They share the same x_top (extent along the H direction) and
# the same scale, so they stack and align visually.
_STRIP_MAX_W_PX = 720
_STRIP_MAX_H_PX = 320


def _strip_pixel_size(x_top, W):
    """Pixel dimensions for both strips at a shared scale factor. Returns
    (width_px, height_px, scale_px). The scale is the larger one that keeps
    both dimensions inside the budget."""
    if x_top <= 0 or W <= 0:
        return _STRIP_MAX_W_PX, _STRIP_MAX_H_PX, 30.0
    scale_px = min(_STRIP_MAX_W_PX / x_top, _STRIP_MAX_H_PX / W)
    return int(scale_px * x_top), int(scale_px * W), scale_px


def _render_compact_metrics(items):
    """Render a stack of label / value pairs in less vertical space than
    `st.metric`. Used in the narrow right column beside each strip so all
    three metrics fit within the strip's pixel height instead of spilling
    below it."""
    blocks = []
    for label, value in items:
        blocks.append(
            '<div style="margin-bottom:0.6rem;">'
            f'<div style="font-size:0.8rem;color:rgba(49,51,63,0.65);'
            f'line-height:1.2;">{label}</div>'
            f'<div style="font-size:1.55rem;font-weight:400;color:#0e1117;'
            f'line-height:1.2;">{value}</div>'
            "</div>"
        )
    st.markdown("".join(blocks), unsafe_allow_html=True)


def _render_optimizer_strip(data, layout, L, x_top):
    """Render the optimizer's solved layout as absolutely-positioned HTML
    divs at exactly the same pixel scale as the user's drawable-canvas strip
    above it. Going via HTML (rather than Altair) lets us share the scale
    factor directly — Altair's compound charts shave a few pixels off the
    plot area for padding, which made the chart's rectangles slightly
    smaller than the canvas's. Tooltips and a legend are sacrificed; rect
    indices are drawn on top of each shape, same as on the user's strip."""
    rects = data["rects"]
    W = float(data["W"])
    canvas_w, canvas_h, scale_px = _strip_pixel_size(x_top, W)
    rect_divs = []
    has_layout = layout is not None and layout.get("x")
    for i in rects:
        if not has_layout:
            continue
        x = float(layout["x"][i])
        y = float(layout["y"][i])
        w = float(data["w"][i])
        length = float(data["length"][i])
        # 90° CW mapping: canvas_x = orig_y, canvas_y = orig_x; the
        # rectangle's canvas extent is (length, w) since w runs along the
        # W direction (vertical here) and length along the L direction
        # (horizontal).
        cx = y * scale_px
        cy = x * scale_px
        cw = length * scale_px
        ch = w * scale_px
        color = _PALETTE[(int(i) - 1) % len(_PALETTE)]
        rect_divs.append(
            f'<div style="position:absolute;left:{cx:.2f}px;top:{cy:.2f}px;'
            f'width:{cw:.2f}px;height:{ch:.2f}px;background:{color};'
            f'border:2px solid #ffffff;box-sizing:border-box;color:#ffffff;'
            f'font-weight:700;font-size:14px;display:flex;align-items:center;'
            f'justify-content:center;">{int(i)}</div>'
        )
    # Dashed red strip outline runs from x=0 to x=L if solved, otherwise to
    # the shared x_top so the area doesn't collapse on a fresh page.
    outline_w_units = L if (L is not None and L > 0) else x_top
    outline_w_px = outline_w_units * scale_px
    outline_div = (
        f'<div style="position:absolute;left:0;top:0;'
        f'width:{outline_w_px:.2f}px;height:{canvas_h}px;'
        f'border:2px dashed #dc2626;box-sizing:border-box;'
        f'pointer-events:none;"></div>'
    )
    container = (
        f'<div style="position:relative;width:{canvas_w}px;'
        f'height:{canvas_h}px;background:#f4f6fa;">'
        f'{outline_div}{"".join(rect_divs)}</div>'
    )
    st.markdown(container, unsafe_allow_html=True)


# ---------- User strip (drag-and-drop) ----------
#
# The user's interactive strip is a Fabric.js canvas (via streamlit-drawable-
# canvas) sized to exactly match the optimizer's HTML strip below it. Each
# rectangle from the current data set is pre-placed in the strip, cascaded
# from the top-left so they're individually clickable. The user drags them
# into a packing of their choosing; on every interaction the canvas state
# comes back as JSON and we convert it back to original-frame (x, y) for
# overlap / out-of-strip validation and "used length" computation.

def _initial_canvas_drawing(data, scale_px, x_top, W):
    """Fabric.js JSON for the initial canvas state: one locked rectangle per
    item in the current data set. Rectangles are pre-placed in a feasible
    "worst-case" layout — all in a single row at the near edge of the strip
    (x=0), stretched end-to-end along the L direction (so `Your length`
    starts at L_max and the user improves from there)."""
    rects = data["rects"]
    objects = []
    cumulative = 0.0  # running sum along the L direction (original y)
    for idx, i in enumerate(rects):
        w = float(data["w"][i])           # W-direction width → vertical on canvas
        length = float(data["length"][i]) # L-direction length → horizontal on canvas
        color = _PALETTE[(int(i) - 1) % len(_PALETTE)]
        # End-to-end packing: orig_x = 0 (touching the strip's near edge),
        # orig_y = cumulative length of all earlier rectangles.
        orig_x = 0.0
        orig_y = cumulative
        cumulative += length
        # 90° CW mapping for the canvas (y grows down): canvas_x = orig_y,
        # canvas_y = orig_x. Each rectangle is a Fabric.js Group of (rect,
        # text) so the index label moves with the shape during drags.
        cw = length * scale_px
        ch = w * scale_px
        objects.append({
            "type": "group",
            "version": "4.6.0",
            "originX": "left",
            "originY": "top",
            "left": orig_y * scale_px,
            "top": orig_x * scale_px,
            "width": cw,
            "height": ch,
            "lockScalingX": True,
            "lockScalingY": True,
            "lockRotation": True,
            "hasControls": False,
            # Override Fabric.js's default four-arrows "move" cursor — a grab
            # hand reads better for "pick up and drag" UX.
            "hoverCursor": "grab",
            "moveCursor": "grabbing",
            "subTargetCheck": False,
            "objects": [
                {
                    "type": "rect",
                    "version": "4.6.0",
                    "originX": "center",
                    "originY": "center",
                    "left": 0,
                    "top": 0,
                    "width": cw,
                    "height": ch,
                    "fill": color,
                    "stroke": "#ffffff",
                    "strokeWidth": 2,
                },
                {
                    "type": "text",
                    "version": "4.6.0",
                    "originX": "center",
                    "originY": "center",
                    "left": 0,
                    "top": 0,
                    "text": str(int(i)),
                    "fontSize": 14,
                    "fontFamily": "Helvetica",
                    "fontWeight": "700",
                    "fill": "#ffffff",
                    "textAlign": "center",
                },
            ],
        })
    return {"version": "4.6.0", "objects": objects}


def _parse_canvas_layout(canvas_result, scale_px, rects):
    """Read rectangle positions from the canvas JSON state and return an
    original-frame layout dict {x: {...}, y: {...}}. Objects come back in
    the order they were drawn, which matches `rects`. Returns None when
    the canvas hasn't reported yet or hasn't fully loaded its rectangles
    — otherwise an interim report would zero-fill missing positions and
    falsely flag every rectangle as overlapping at the origin."""
    if canvas_result is None or canvas_result.json_data is None:
        return None
    objs = [o for o in canvas_result.json_data.get("objects", [])
            if o.get("type") == "group"]
    if len(objs) < len(rects):
        return None
    layout = {"x": {}, "y": {}}
    for idx, i in enumerate(rects):
        o = objs[idx]
        # The group is serialized with originX="left", originY="top", so
        # `left`/`top` is the top-left corner. canvas_x = orig_y,
        # canvas_y = orig_x → invert when mapping back.
        layout["x"][i] = float(o.get("top", 0.0)) / scale_px
        layout["y"][i] = float(o.get("left", 0.0)) / scale_px
    return layout


def _validate_user_layout(data, layout):
    """Compute used length, overlapping pairs, and out-of-strip rectangles.
    The canvas already constrains rectangles inside its area, so out-of-
    strip flags should only fire on data edge cases. Tolerance of 0.01 lets
    rectangles touch edges without registering an overlap."""
    if layout is None:
        return None
    rects = data["rects"]
    W = float(data["W"])
    tol = 0.01
    overlaps = []
    oob = set()
    used_length = 0.0
    for i in rects:
        x = float(layout["x"].get(i, 0.0))
        y = float(layout["y"].get(i, 0.0))
        w = float(data["w"][i])
        length = float(data["length"][i])
        if x < -tol or x + w > W + tol or y < -tol:
            oob.add(int(i))
        used_length = max(used_length, y + length)
    for idx_i, i in enumerate(rects):
        xi = float(layout["x"].get(i, 0.0))
        yi = float(layout["y"].get(i, 0.0))
        wi = float(data["w"][i])
        li = float(data["length"][i])
        for j in rects[idx_i + 1:]:
            xj = float(layout["x"].get(j, 0.0))
            yj = float(layout["y"].get(j, 0.0))
            wj = float(data["w"][j])
            lj = float(data["length"][j])
            x_sep = (xi + wi <= xj + tol) or (xj + wj <= xi + tol)
            y_sep = (yi + li <= yj + tol) or (yj + lj <= yi + tol)
            if not (x_sep or y_sep):
                overlaps.append((int(i), int(j)))
    return {
        "used_length": used_length,
        "overlaps": overlaps,
        "oob": sorted(oob),
        "feasible": (not overlaps) and (not oob),
    }


def render_user_strip(data, x_top):
    """Draw the user's drag-and-drop canvas. Returns the parsed layout in
    the original (un-rotated) frame, or None if the canvas isn't ready
    yet. The canvas key is a hash of the rectangle set + W + a reset
    counter, so editing data or hitting Reset re-initializes; everything
    else keeps the iframe mounted."""
    W = float(data["W"])
    canvas_w, canvas_h, scale_px = _strip_pixel_size(x_top, W)
    reset_v = st.session_state.get("user_canvas_reset_v", 0)
    sig = (
        tuple(int(i) for i in data["rects"]),
        tuple(round(float(data["w"][i]), 3) for i in data["rects"]),
        tuple(round(float(data["length"][i]), 3) for i in data["rects"]),
        round(W, 3),
        reset_v,
    )
    key = f"user_canvas_{abs(hash(sig))}"
    initial = _initial_canvas_drawing(data, scale_px, x_top, W)

    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=2,
        stroke_color="#ffffff",
        background_color="#f4f6fa",
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode="transform",
        initial_drawing=initial,
        display_toolbar=False,
        key=key,
    )

    # Suppress streamlit-drawable-canvas's built-in "remove active object
    # on double-click" — it would let the user accidentally delete a
    # rectangle. A capture-phase dblclick listener on the upper-canvas
    # stops propagation before fabric's remove handler runs. The
    # drawable-canvas iframe mounts asynchronously; we retry briefly with
    # backoff. Streamlit reruns re-inject this block, so the listener
    # re-attaches if the canvas iframe ever remounts.
    st.components.v1.html(
        """
        <script>
        (function() {
            function attach() {
                const ifr = window.parent.document.querySelector(
                    'iframe[src*="drawable_canvas"]'
                );
                if (!ifr || !ifr.contentDocument) return false;
                const upper = ifr.contentDocument.querySelector(
                    'canvas.upper-canvas'
                );
                if (!upper) return false;
                if (!upper.__qtDblclickBlocked) {
                    upper.__qtDblclickBlocked = true;
                    upper.addEventListener('dblclick', function(e) {
                        e.stopImmediatePropagation();
                        e.preventDefault();
                    }, true);
                }
                return true;
            }
            if (attach()) return;
            let attempts = 0;
            const poll = () => {
                attempts++;
                if (attach() || attempts > 40) return;
                setTimeout(poll, 200);
            };
            setTimeout(poll, 100);
        })();
        </script>
        """,
        height=0,
    )
    return _parse_canvas_layout(canvas_result, scale_px, data["rects"])


def render_optimizer_tab():
    data = st.session_state.data
    optimal = st.session_state.optimal

    if not data["rects"]:
        st.info("Add at least one rectangle on the Data tab.")
        return

    # Shared length extent for both strips so they line up visually. The
    # MILP's worst-case (stack everything end-to-end) is L_max = sum of
    # length_i; that's the most space the user could conceivably need,
    # with a small floor so a single-rect instance still has room to drag.
    L_max = float(sum(data["length"][i] for i in data["rects"])) or 1.0
    opt_L = float(optimal["L"]) if (optimal and optimal["status"] == "optimal") else None
    x_top = max(opt_L or 0.0, L_max, 12.0)

    area = total_area(data)
    W = float(data["W"])

    # ── Your packing (drag-and-drop) ─────────────────────────────────────────
    # Strip on the left, metric stack tight against its right edge, and a
    # trailing spacer column so the metric column doesn't drift to the far
    # side of the page (the strip is a fixed 720 px max). Title row is a
    # flexbox so any OOB / overlap notice sits flush against the heading
    # instead of inheriting a column's left-edge offset. The slot is
    # filled at the bottom of this block once we know the user layout.
    title_slot = st.empty()

    user_strip_col, user_metric_col, _ = st.columns([4, 1, 3])
    with user_strip_col:
        user_layout = render_user_strip(data, x_top)
    user_result = _validate_user_layout(data, user_layout) if user_layout else None

    with user_metric_col:
        if user_result is not None:
            used = user_result["used_length"]
            if user_result["feasible"] and used > 0 and W > 0:
                user_eff_text = f"{area / (W * used) * 100.0:.1f}%"
            else:
                user_eff_text = "—"
            _render_compact_metrics([
                ("Your length", f"{used:.3f}"),
                ("Your efficiency", user_eff_text),
            ])
            if st.button("Reset your packing"):
                st.session_state["user_canvas_reset_v"] = (
                    st.session_state.get("user_canvas_reset_v", 0) + 1
                )
                st.rerun()
        else:
            _render_compact_metrics([
                ("Your length", "—"),
                ("Your efficiency", "—"),
            ])

    notice_parts = []
    if user_result is not None:
        if user_result["oob"]:
            notice_parts.append(
                "Out of strip: rectangle(s) "
                + ", ".join(str(i) for i in user_result["oob"])
                + f" extend past the strip width W={data['W']:.2f}."
            )
        if user_result["overlaps"]:
            pairs = ", ".join(f"({a},{b})" for a, b in user_result["overlaps"])
            notice_parts.append(f"Overlapping pairs: {pairs}.")
    notice_html = ""
    if notice_parts:
        # Inline styling mirrors Streamlit's st.error look but without the
        # full-width alert box.
        notice_html = (
            '<div style="background:#fff0f0;color:#7d1d1d;'
            'padding:0.4rem 0.9rem;border-radius:0.4rem;'
            'border:1px solid #ffcccc;font-size:0.9rem;">'
            + " ".join(notice_parts)
            + "</div>"
        )
    title_slot.markdown(
        '<div style="display:flex;align-items:center;gap:0.75rem;'
        'margin-bottom:0.5rem;">'
        '<h4 style="margin:0;">Your packing</h4>'
        + notice_html
        + "</div>",
        unsafe_allow_html=True,
    )

    # ── Optimizer's packing ─────────────────────────────────────────────────
    # Title slot mirrors the user-packing layout: heading on the left, a
    # "Click Solve" info notice flush against it while no solution exists.
    # Notice state depends only on whether `optimal` is set, so we can
    # render the final title HTML up-front (no two-pass replace needed,
    # which keeps the title stable across canvas reruns).
    if optimal is None:
        opt_notice_html = (
            '<div style="background:#e7f3fe;color:#084298;'
            'padding:0.4rem 0.9rem;border-radius:0.4rem;'
            'border:1px solid #b6d4fe;font-size:0.9rem;">'
            "Click <b>Solve</b> in the sidebar to compute and visualize "
            "the optimal packing.</div>"
        )
    else:
        opt_notice_html = ""
    st.markdown(
        '<div style="display:flex;align-items:center;gap:0.75rem;'
        'margin:0;">'
        "<h4 style=\"margin:0;\">Optimizer's packing</h4>"
        + opt_notice_html
        + "</div>",
        unsafe_allow_html=True,
    )

    if optimal and optimal["status"] == "optimal":
        opt_eff = (area / (W * opt_L) * 100.0) if (W > 0 and opt_L > 0) else 0.0
        l_text = f"{opt_L:.3f}"
        opt_eff_text = f"{opt_eff:.1f}%"
    else:
        l_text = "—"
        opt_eff_text = "—"
    elapsed = optimal.get("elapsed") if optimal else None
    time_text = f"{elapsed:.2f} s" if isinstance(elapsed, (int, float)) else "—"

    opt_strip_col, opt_metric_col, _ = st.columns([4, 1, 3])
    with opt_strip_col:
        if optimal and optimal["status"] == "optimal":
            layout = {"x": optimal["x"], "y": optimal["y"]}
            _render_optimizer_strip(data, layout, opt_L, x_top)
        else:
            _render_optimizer_strip(data, None, None, x_top)
    with opt_metric_col:
        _render_compact_metrics([
            ("Optimal length", l_text),
            ("Optimal efficiency", opt_eff_text),
            ("Solve time", time_text),
        ])

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
                "Length": st.column_config.NumberColumn(
                    "Length (ℓ_i)", min_value=0.0, format="%.2f",
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
        | (pd.to_numeric(edited["Length"], errors="coerce") <= 0)
    ).any()
    if raw_invalid:
        warnings.append(
            "Rows with non-positive width or length were dropped."
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
                f"set of standard MILP constraints (Big-M / Hull / mbigm / "
                f"Cutting Plane)."
            )


def render_logs_tab():
    optimal = st.session_state.optimal
    if not optimal:
        st.info("Click **Solve** in the sidebar to see solver logs.")
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

# GDP transformation choice. The four reformulations route through pyomo.gdp
# and produce different MILPs; the Optimizer tab reports the solve time so
# the user can compare them head-to-head.
transform_label = st.sidebar.radio(
    "GDP transformation",
    options=list(_GDP_TRANSFORMS.keys()),
    index=0,
    key="transform_radio",
    help=(
        "Big-M: classical linearization with a single large constant per "
        "disjunct — fewest variables, loosest LP relaxation. "
        "Hull: disaggregated copies of the variables; tighter relaxation. "
        "Multiple Big-M: a per-constraint tight Big-M, usually between the "
        "two. "
        "Cutting Plane: starts from Big-M and iteratively adds violated "
        "hull facets — can be slow but gives the tightest formulation."
    ),
)
transform_key = _GDP_TRANSFORMS[transform_label]

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
    "<a href='https://github.com/Pyomo/pyomo' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>Pyomo</a>"
    " + "
    "<a href='https://github.com/ERGO-Code/HiGHS' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>HiGHS</a>"
    "</span>"
    "</h2>",
    unsafe_allow_html=True,
)
_caption_col, _ = st.columns([1, 1])
with _caption_col:
    st.markdown(
        "Pack $N$ rectangles into a strip of fixed width $W$ to minimize "
        "the strip length $L$. Drag rectangles in **Your packing** to try "
        "your own layout. Non-overlap is written as **disjunctions** "
        "(`pyomo.gdp`) and reformulated to a MILP — pick a transformation "
        "(Big-M, Hull, Multiple Big-M, Cutting Plane) in the sidebar and "
        "click **Solve** to see the optimum. Edit instances in the "
        "**Data** tab; **Formulation** and **Logs** show the underlying "
        "GDP and solver output."
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
