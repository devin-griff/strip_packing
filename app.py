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
import os
import time
from pathlib import Path

import pyomo.environ as pyo
import streamlit as st
from pyomo.common.errors import ApplicationError
from pyomo.gdp import Disjunction
from pyomo.opt import TerminationCondition


def _materialize_gurobi_license():
    """Production license shim. Fly secrets surface as environment
    variables, but gurobipy wants a license FILE — so if the three WLS
    values are present and no license file is configured, write one to
    the home directory and point GRB_LICENSE_FILE at it. Local dev is
    untouched: there GRB_LICENSE_FILE already points at a file on disk
    (or gurobipy finds one in its default locations), and the values
    never appear in the repo or the image — only in Fly's secret store
    and the container's private filesystem."""
    if os.environ.get("GRB_LICENSE_FILE"):
        return
    access = os.environ.get("GRB_WLSACCESSID")
    secret = os.environ.get("GRB_WLSSECRET")
    license_id = os.environ.get("GRB_LICENSEID")
    if not (access and secret and license_id):
        return
    lic_path = Path.home() / "gurobi.lic"
    if not lic_path.exists():
        lic_path.write_text(
            f"WLSACCESSID={access}\n"
            f"WLSSECRET={secret}\n"
            f"LICENSEID={license_id}\n",
            encoding="utf-8",
        )
    os.environ["GRB_LICENSE_FILE"] = str(lic_path)


_materialize_gurobi_license()


# Hard cap on rectangle count. Big-M MILP gets slow beyond ~15 rectangles
# because the number of disjunctions grows as N(N-1)/2.
MAX_RECTS = 15

# Default instance shown on first load and after the "Reset to defaults"
# button. Fifteen rectangles produced by a random guillotine partition of
# a 10 x 12 rectangle (search seed 21), so a perfect packing exists:
# total area 120, optimum L = 12 at 100% efficiency. Several identical
# groups (including five 4x2 pieces) keep the identical-rectangle
# ordering constraints active out of the box. Gurobi + Big-M proves
# optimality in ~15 s on a workstation — comfortably inside the 60 s
# cap — while HiGHS at the default 10 s leaves a visible gap, so the
# solver comparison has a story on first Solve.
DEFAULT_DATA = {
    "rects": list(range(1, 16)),
    "w": {1: 6.0, 2: 6.0, 3: 4.0, 4: 4.0, 5: 4.0, 6: 4.0, 7: 4.0,
          8: 4.0, 9: 4.0, 10: 4.0, 11: 4.0, 12: 4.0, 13: 2.0,
          14: 1.0, 15: 1.0},
    "length": {1: 2.0, 2: 1.0, 3: 3.0, 4: 3.0, 5: 3.0, 6: 2.0, 7: 2.0,
               8: 2.0, 9: 2.0, 10: 2.0, 11: 1.0, 12: 1.0, 13: 3.0,
               14: 6.0, 15: 6.0},
    "W": 10.0,
}

# GDP → MILP transformations offered via the radio above the strip on
# the Optimizer tab — the classical Big-M / Hull pair, both
# TransformationFactory entries in pyomo.gdp.
_GDP_TRANSFORMS = {
    "Big-M": "gdp.bigm",
    "Hull": "gdp.hull",
}
_GDP_LABEL = {v: k for k, v in _GDP_TRANSFORMS.items()}

# MIP solver choices for the Optimizer tab radio. Both consume the same
# GDP-reformulated MILP under the same selectable time cap; HiGHS is the
# open-source default (pip wheel, no license), Gurobi the commercial
# comparison — licensed via Gurobi's Web License Service, with a seat
# checked out per solve and released immediately after.
_MIP_SOLVERS = {"HiGHS": "appsi_highs", "Gurobi": "appsi_gurobi"}


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
    # end-to-end along the strip). Each x_i (position along the length) is
    # bounded by the same sum.
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

    # Decision variables (near corner of each rectangle). x runs ALONG the
    # strip length — the horizontal, minimized direction — bounded by
    # L_max; y runs ACROSS the fixed width, bounded by W. This matches the
    # on-screen picture (x horizontal, y vertical) and the Sawaya &
    # Grossmann notation. Explicit bounds let gdp.bigm derive sensible
    # Big-M values automatically.
    m.x = pyo.Var(m.RECTS, domain=pyo.NonNegativeReals, bounds=(0.0, L_max))
    m.y = pyo.Var(m.RECTS, domain=pyo.NonNegativeReals, bounds=(0.0, W))
    m.L = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.0, L_max))

    # Objective: minimize the strip length L.
    m.total_length = pyo.Objective(expr=m.L, sense=pyo.minimize)

    # Containment: each rectangle fits inside the strip. fit_x ties each
    # rectangle's far end (along the length) to the strip length L; fit_y
    # keeps it within the fixed width W.
    def fit_x_def(m, i):
        return m.x[i] + m.length[i] <= m.L
    m.fit_x = pyo.Constraint(m.RECTS, rule=fit_x_def)

    def fit_y_def(m, i):
        return m.y[i] + m.w[i] <= m.W
    m.fit_y = pyo.Constraint(m.RECTS, rule=fit_y_def)

    # Symmetry breaking (Sawaya & Grossmann): pin the largest rectangle's
    # CENTER into the lower-left quadrant. Every packing either satisfies
    # these or its mirror image does (reflect across the strip's
    # centerline / along its length), so an optimal solution always
    # survives — but branch-and-bound no longer proves the same bound
    # separately for all four reflections of each layout. The y pin is
    # legal despite L being a variable: the vertical mirror of a feasible
    # packing has the same L, and the constraint stays linear. Largest
    # rectangle as reference = strongest pruning (least placement
    # freedom). Plain global constraints, untouched by the GDP
    # transformations.
    if rects:
        ref = max(rects, key=lambda i: float(data["w"][i]) * float(data["length"][i]))
        m.sym_x = pyo.Constraint(expr=m.x[ref] + m.length[ref] / 2.0 <= m.L / 2.0)
        m.sym_y = pyo.Constraint(expr=m.y[ref] + m.w[ref] / 2.0 <= m.W / 2.0)

        # Permutation symmetry: rectangles with identical (w, length) are
        # interchangeable, so order each identical group along the length
        # (x) to keep one representative per permutation. The reference
        # rectangle's group is exempt — ordering it could fight the
        # quadrant pin above and jointly cut off every optimum.
        groups = {}
        for i in rects:
            key = (float(data["w"][i]), float(data["length"][i]))
            groups.setdefault(key, []).append(i)
        m.lex = pyo.ConstraintList()
        for members in groups.values():
            if len(members) < 2 or ref in members:
                continue
            for a, b in zip(members, members[1:]):
                m.lex.add(m.x[a] <= m.x[b])

    # Non-overlap disjunctions: for every unordered pair (i, j) with i < j,
    # at least one of the four geometric separations must hold. `Disjunction`
    # accepts a list of disjuncts, each being a list of constraint expressions.
    #
    # The above/below disjuncts carry two extra inequalities — the S2
    # degeneracy-breaking form of Trespalacios & Grossmann. In the classic
    # four-disjunct model the regions OVERLAP: a pair separated both
    # across the width and along the length can be encoded by two
    # different disjunct selections, and branch-and-bound explores both
    # encodings of the same packing. Requiring the above/below disjuncts
    # to also overlap lengthwise by >= 1 routes every diagonal/flush
    # arrangement uniquely through left/right. The "+1" (rather than
    # >= 0) closes the edge-flush tie and is valid because all dimensions
    # are integer — the editor enforces integer inputs.
    pairs = [(i, j) for idx_i, i in enumerate(rects) for j in rects[idx_i + 1:]]
    if pairs:
        m.PAIRS = pyo.Set(initialize=pairs, dimen=2)

        def disj_rule(m, i, j):
            return [
                [m.x[i] + m.length[i] <= m.x[j]],       # i left of j
                [m.x[j] + m.length[j] <= m.x[i]],       # i right of j
                [m.y[i] + m.w[i] <= m.y[j],             # i below j ...
                 m.x[i] + m.length[i] >= m.x[j] + 1,    # ... and lengthwise
                 m.x[j] + m.length[j] >= m.x[i] + 1],   #     overlap >= 1
                [m.y[j] + m.w[j] <= m.y[i],             # i above j ...
                 m.x[i] + m.length[i] >= m.x[j] + 1,
                 m.x[j] + m.length[j] >= m.x[i] + 1],
            ]
        m.no_overlap = Disjunction(m.PAIRS, rule=disj_rule)

    return m


# Default cap on the MIP master solve (HiGHS or Gurobi). Large instances
# (especially with Big-M on N close to MAX_RECTS) can take much longer
# than this in the worst case; cutting off keeps the UI responsive and
# surfaces the optimality gap when the solver doesn't converge. The user
# can raise the cap via the Time limit select on the Optimizer tab —
# bounded at 60 s so a public visitor can't pin the page (or a Gurobi
# WLS seat) for minutes.
SOLVE_TIME_LIMIT_S = 10.0
_TIME_LIMITS = {"10": 10.0, "30": 30.0, "60": 60.0}

# Below this relative gap, treat the solve as effectively optimal (the
# default HiGHS MIP gap tolerance is 0.01% = 1e-4; we use a tiny bit
# higher to absorb floating-point noise).
GAP_OPTIMAL_THRESHOLD_PCT = 0.05


class _LicenseBusyError(RuntimeError):
    """Raised when Gurobi's WLS checkout fails even after a retry —
    typically because the license's concurrent-session seats are all
    taken. solve() maps this onto the `license_busy` status."""


def _solve_capturing(m, transform, solver_name="appsi_highs",
                     time_limit_s=SOLVE_TIME_LIMIT_S):
    """Apply the GDP transformation, run the chosen MIP solver, and
    return (termination_condition, gap_pct, log_text, elapsed) with the
    solution (if any) loaded onto `m`. Captures the solver's stdout via
    contextlib.redirect_stdout/stderr — same pattern as knapsack /
    diet / circle-packing. `elapsed` is the wall-clock time of
    transformation + solve, in seconds — shown as a metric on the
    Optimizer tab so users can compare reformulations and solvers
    head-to-head.

    The two solvers ride different Pyomo interfaces on purpose. HiGHS
    uses the legacy SolverFactory("appsi_highs") wrapper — proven in
    production, warts documented in _load_solution_if_present. Gurobi
    uses the NATIVE appsi interface: the legacy wrapper's symbol-map
    bookkeeping crashes on GDP-transformed models when paired with
    appsi_gurobi + load_solutions=False ('DisjunctData' has no
    attribute 'solutions'), and the native interface both avoids that
    and hands us the primal/dual bounds directly."""
    # Reformulate the GDP into a standard MILP. Big-M replaces each
    # disjunct constraint with a linearization that goes vacuous unless
    # the disjunct's indicator is selected; Hull adds disaggregated
    # variable copies for a tighter (convex-hull) relaxation. Neither
    # touches a solver, so the transformation is fast and license-free.
    t0 = time.perf_counter()
    pyo.TransformationFactory(transform).apply_to(m)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        if solver_name == "appsi_gurobi":
            tc, gap_pct = _run_gurobi(m, time_limit_s)
        else:
            tc, gap_pct = _run_highs(m, time_limit_s)
    # Scrub license-identifying lines from the captured log before it
    # reaches the public Logs tab — Gurobi's WLS banner prints the
    # license ID and registrant ("WLS license NNNNNNN - registered to
    # ..."). Substring match keeps this robust to wording shifts across
    # Gurobi versions; HiGHS logs never match.
    log_text = "\n".join(
        ln for ln in buf.getvalue().splitlines()
        if not any(
            marker in ln.lower()
            for marker in ("wls", "registered to", "academic license")
        )
    )
    elapsed = time.perf_counter() - t0
    return tc, gap_pct, log_text, elapsed


def _run_highs(m, time_limit_s=SOLVE_TIME_LIMIT_S):
    """Master solve via the legacy appsi_highs wrapper. Returns
    (termination_condition, gap_pct); solution loaded onto m when an
    incumbent exists."""
    solver = pyo.SolverFactory("appsi_highs")
    solver.options["time_limit"] = time_limit_s
    # When HiGHS hits the time cap without finding any feasible
    # solution (e.g. Hull on N=15), the default solve(load_solutions=
    # True) path raises RuntimeError because there's nothing to load.
    # Disable the auto-load at the solver-call level, then manually
    # copy the solution below only if an incumbent actually exists.
    results = solver.solve(m, tee=True, load_solutions=False)
    _load_solution_if_present(m, results)
    return results.solver.termination_condition, _extract_gap_pct(results)


def _run_gurobi(m, time_limit_s=SOLVE_TIME_LIMIT_S):
    """Master solve via the NATIVE appsi Gurobi interface. Returns
    (termination_condition mapped onto the legacy enum, gap_pct);
    solution loaded onto m when an incumbent exists.

    Gurobi checks out a WLS license seat when its environment starts.
    The license allows a small number of concurrent sessions and seats
    churn in seconds, so one checkout collision gets a quiet retry
    before surfacing as license_busy — and the seat is ALWAYS released
    afterward, since holding it would pin license capacity to this
    machine between solves. (The seat is held for the duration of the
    solve, which is why the user-selectable time limit is bounded.)"""
    from pyomo.contrib.appsi.solvers import Gurobi as AppsiGurobi

    opt = AppsiGurobi()
    opt.config.time_limit = time_limit_s
    opt.config.load_solution = False
    opt.config.stream_solver = True  # log into the redirected stdout
    try:
        for attempt in (1, 2):
            try:
                res = opt.solve(m)
                break
            except Exception as e:
                lowered = str(e).lower()
                if "license" in lowered or "wls" in lowered:
                    if attempt == 1:
                        time.sleep(2.0)
                        continue
                    raise _LicenseBusyError(str(e)) from e
                raise
        if res.best_feasible_objective is not None:
            res.solution_loader.load_vars()
    finally:
        try:
            opt.release_license()
        except Exception:
            pass

    # Map the appsi TerminationCondition onto the legacy enum solve()
    # branches on — the member names match for everything we handle
    # (optimal, maxTimeLimit, infeasible, infeasibleOrUnbounded,
    # unbounded); anything exotic falls back to `unknown`.
    tc = getattr(
        TerminationCondition, res.termination_condition.name,
        TerminationCondition.unknown,
    )

    primal = res.best_feasible_objective
    dual = res.best_objective_bound
    gap_pct = None
    if (primal is not None and dual is not None
            and math.isfinite(primal) and math.isfinite(dual)
            and primal > 1e-10):
        gap_pct = max(0.0, (primal - dual) / primal * 100.0)
    return tc, gap_pct


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


def solve(data, transform="gdp.bigm", solver_name="appsi_highs",
          time_limit_s=SOLVE_TIME_LIMIT_S):
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
        tc, gap_pct, log, elapsed = _solve_capturing(
            m, transform, solver_name, time_limit_s)
    except _LicenseBusyError:
        return {
            "status": "license_busy",
            "message": (
                "The Gurobi license is busy (it allows a limited number "
                "of concurrent solves). Wait a few seconds and click "
                "Solve again."
            ),
            "x": {}, "y": {}, "L": None, "log": "", "transform": transform,
            "elapsed": None,
        }
    except ApplicationError as e:
        pkg = "gurobipy" if solver_name == "appsi_gurobi" else "highspy"
        return {
            "status": "solver_missing",
            "message": (
                f"MIP solver not available. Run `pip install {pkg}` "
                f"in your environment. ({e})"
            ),
            "x": {}, "y": {}, "L": None, "log": "", "transform": transform,
            "elapsed": None,
        }

    # Translate Pyomo's TerminationCondition enum (returned by
    # _solve_capturing alongside the gap) into a small set of stable
    # status strings the UI knows how to render. With a time limit set,
    # the solver may return `maxTimeLimit` carrying a best-known feasible
    # incumbent — treated as a "feasible" status so the UI still draws
    # the packing, with the gap surfaced separately.

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
                    "elapsed": elapsed, "time_limit_s": time_limit_s}
        return {
            "status": "optimal",
            "x": x, "y": y, "L": L, "gap_pct": gap_pct,
            "log": log, "transform": transform, "elapsed": elapsed,
            "time_limit_s": time_limit_s,
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
                    "elapsed": elapsed, "time_limit_s": time_limit_s}
        return {
            "status": "time_limit",
            "x": x, "y": y, "L": L, "gap_pct": gap_pct,
            "log": log, "transform": transform, "elapsed": elapsed,
            "time_limit_s": time_limit_s,
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
#   - solver_radio:        value backing the inline MIP-solver radio
#   - time_limit_radio:    value backing the inline time-limit radio
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
    # Drop (don't assign) the widget-backed key: the number_input then
    # re-initializes from its value= argument on the next render.
    # Assigning here triggers Streamlit's "created with a default value
    # but also had its value set via the Session State API" warning,
    # which flashes as a yellow box in the strip column on Reset.
    st.session_state.pop("W_input", None)
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


def _delete_rect(rid):
    """on_click callback for the editor's per-row delete button. Runs
    before the rerun renders anything, so the shortened list paints in
    one clean pass instead of aborting the render mid-loop. Clears the
    stored solve since the instance changed."""
    st.session_state.data = remove_rect(dict(st.session_state.data), rid)
    st.session_state.optimal = None


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
    """Cascade layout — all rectangles at y=0 (against one width edge),
    stacked end-to-end along the length (x). This is the worst-case
    feasible packing and serves as a no-solver default visualization."""
    layout = {"x": {}, "y": {}}
    cumulative = 0.0
    for i in data["rects"]:
        layout["y"][i] = 0.0
        layout["x"][i] = cumulative
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
    strip's negative-margin layout crowds up against the value text.
    `white-space: nowrap` on both label and value keeps each metric
    on a single line per row (otherwise "100.0%" or "Efficiency"
    wrap in narrow columns)."""
    # Value font runs a notch under diet's 2.25rem original — the top
    # row hosts three radios plus five metrics, and 1.8rem buys the
    # difference without reading "small".
    slot.markdown(
        "<div style='margin:0.25rem 0 1.3rem 0; line-height:1.2;'>"
        "<div style='font-size:0.875rem; "
        "margin-bottom:0.6rem; white-space:nowrap;'>"
        f"{label}"
        "</div>"
        "<div style='font-size:1.8rem; font-weight:400; line-height:1.1; "
        "white-space:nowrap;'>"
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
        # Direct mapping: container-horizontal = x (along the length L),
        # container-vertical = y (across the width W). Positions and sizes
        # are percentages of x_top (horizontal) and W (vertical) so the
        # container resizes freely.
        left_pct = (x / x_top) * 100.0
        top_pct = (y / W) * 100.0
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
            /* Streamlit's default 1rem column gap is the difference
               between the ten-item control row fitting on one line and
               the radios wrapping — nine gutters at half a rem buys
               ~70px back. */
            gap: 0.5rem !important;
        }
        /* Keep the gap between a widget's label and its content equal
           to the metric blocks' label gap so the rows align. */
        [data-testid="stMainBlockContainer"] [data-testid="stWidgetLabel"] {
            margin-bottom: 0.25rem !important;
        }
        /* st.number_input stretches to fill its column, so the W
           column could never show a gap before "MIP solver" no matter
           the column weights — every other group's gap comes from its
           natural content width. Cap the W input so it behaves like
           the radios and the inter-group spacing evens out. (st-key-*
           classes come from the widget's key.) */
        .st-key-W_input {
            max-width: 8.5rem;
        }
        /* Same story inside the radio groups: tighten the spacing
           between options so three-option groups (Time limit) hold one
           line in their column. */
        div[role="radiogroup"] {
            gap: 0.4rem !important;
        }
        div[role="radiogroup"] label {
            margin-right: 0 !important;
        }
        [data-testid="stNumberInputContainer"] input {
            padding-top: 0.25rem; padding-bottom: 0.25rem;
            text-align: right; padding-right: 0.4rem;
            /* Click-only entry: values change exclusively through the
               +/- steppers, so they stay integers by construction (the
               S2 degeneracy-breaking disjuncts require integer data).
               pointer-events: none blocks click-to-focus typing while
               leaving the stepper buttons (separate elements) live. */
            pointer-events: none;
            user-select: none;
            caret-color: transparent;
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
            line-height: 1.4;
            /* `width: max-content` makes the bubble expand to its
               natural content width (no wrapping); `max-width: 24rem`
               then caps it so very long messages still wrap. Without
               max-content the absolute-positioned ::after would
               shrink-to-fit aggressively and wrap after every word or
               two, which is what showed up in the field. */
            width: max-content;
            max-width: 24rem;
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
        # Same pop-don't-assign rationale as apply_reset: the widget
        # re-inits from value=current_W (already clamped) next render.
        st.session_state.pop("W_input", None)

    # ── Two-column layout: editor (left) | strip column fills remainder ──
    # The metric column is gone — metrics now live in a sub-row above the
    # strip, so the strip itself can stretch to the right edge of the page.
    editor_col, strip_col = st.columns([2.7, 9.3])

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
        # Column weights: Solve / W / Transformation / 5 metric slots.
        # W gets 1.6 (vs 1 for the metric slots) so the +/- stepper
        # buttons have room to sit inline beside the input — at
        # weight 1 the column was too narrow and the steppers wrapped
        # to a second line as a thin teal bar.
        # Column order: Solve / W / GDP transformation / MIP solver /
        # Time limit / 5 metric slots. The radios get the widths their
        # two-option rows actually need, the time-limit selectbox stays
        # compact, and metrics render at full size (diet's
        # colored_metric proportions), tolerating the same mild label
        # overflow the original single-radio layout had.
        top_row = st.columns(
            [0.7, 1.6, 1.7, 1.7, 1.7, 0.97, 0.97, 0.97, 0.97, 0.92],
            vertical_alignment="bottom",
        )
        with top_row[0]:
            solve_clicked = st.button(
                "Solve", type="primary", use_container_width=True,
            )
        with top_row[1]:
            # Integer-typed for the same reason as the rectangle editor:
            # the S2 disjuncts assume integer data throughout.
            W_value = st.number_input(
                "Strip width W",
                min_value=int(math.ceil(min_W - 1e-9)),
                max_value=30,
                value=int(current_W),
                step=1,
                key="W_input",
            )
        with top_row[2]:
            transform_label = st.radio(
                "GDP transformation",
                options=list(_GDP_TRANSFORMS.keys()),
                index=0,
                horizontal=True,
                key="transform_radio",
            )
        with top_row[3]:
            solver_label = st.radio(
                "MIP solver",
                options=list(_MIP_SOLVERS.keys()),
                index=0,
                horizontal=True,
                key="solver_radio",
            )
        with top_row[4]:
            time_label = st.radio(
                "Solve time limit (s)",
                options=list(_TIME_LIMITS.keys()),
                index=0,
                horizontal=True,
                key="time_limit_radio",
            )
        ub_slot = top_row[5].empty()
        opt_slot = top_row[6].empty()
        eff_slot = top_row[7].empty()
        gap_slot = top_row[8].empty()
        time_slot = top_row[9].empty()
        strip_slot = st.empty()
        # Dedicated slot for the "Solving..." spinner so it can appear
        # just below the strip without replacing the strip itself.
        # st.empty() has zero height when empty, so it doesn't affect
        # layout when no solve is running.
        spinner_slot = st.empty()

    transform_key = _GDP_TRANSFORMS[transform_label]
    solver_key = _MIP_SOLVERS[solver_label]
    time_limit_s = _TIME_LIMITS[time_label]

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
        # Render the spinner in its own slot below the strip so the
        # existing strip (if any) stays visible during the solve
        # instead of being replaced. After the solve returns, clear
        # the spinner_slot so it reverts to zero height.
        with spinner_slot.container():
            with st.spinner(f"Solving GDP-transformed MILP via {solver_label}..."):
                result = solve(data, transform_key, solver_key, time_limit_s)
        spinner_slot.empty()
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
    x_top = max(opt_L or 0.0, L_max, 12.0)
    area = total_area(data)
    W = float(data["W"])
    eff_pct = (
        (area / (W * opt_L) * 100.0)
        if (has_incumbent and W > 0 and opt_L and opt_L > 0)
        else None
    )
    # 100% efficiency means the rectangles fill the strip with zero
    # waste — that's the theoretical lower bound on L (L >= area / W),
    # so this incumbent IS provably optimal regardless of what the
    # solver's LP-derived dual bound says. The Big-M LP relaxation can
    # be very loose so the solver may report a large gap even when the
    # geometry rules out anything better.
    geometrically_optimal = (
        eff_pct is not None and eff_pct >= 100.0 - GAP_OPTIMAL_THRESHOLD_PCT
    )
    if geometrically_optimal:
        gap_pct = 0.0
    # "Proved optimal" drives the metric label: HiGHS marked it optimal,
    # OR the bound gap is below the noise floor (solver terminated at
    # time-limit but already had a near-zero gap), OR efficiency hit
    # 100% (geometric proof of optimality).
    proved_optimal = bool(
        optimal
        and (
            optimal["status"] == "optimal"
            or (gap_pct is not None and gap_pct < GAP_OPTIMAL_THRESHOLD_PCT)
            or geometrically_optimal
        )
    )

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
                _cap = optimal.get("time_limit_s", SOLVE_TIME_LIMIT_S)
                st.warning(
                    f"Hit the {_cap:g} s solve cap before finding any "
                    "feasible packing. Try a smaller instance, a longer "
                    "time limit, or a different transformation."
                )
            elif optimal["status"] == "license_busy":
                st.error(
                    optimal.get(
                        "message",
                        "Gurobi license busy — try again in a moment.",
                    )
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
        _cap = optimal.get("time_limit_s", SOLVE_TIME_LIMIT_S)
        tooltip = (
            f"Solver hit the {_cap:g} s time cap before "
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
    # "Total time" not "Solve time": this is wall-clock time for the
    # GDP transformation + master solve combined. The selected time cap
    # applies only to the solver; transformation time (small for Big-M /
    # Hull) rides on top.
    _render_top_metric(
        time_slot, "Total time",
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
    # Fixed slot count: every rerun renders exactly MAX_RECTS row slots,
    # with slots beyond the current rectangle count as invisible
    # placeholders. Streamlit replaces elements positionally as deltas
    # stream in but only sweeps TRAILING leftovers at end-of-run, so a
    # rerun that shrinks the element count leaves the old last row
    # visible for the whole round-trip — the ghost-row flash on delete
    # that circle-packing exhibited. A constant element count removes
    # the trailing leftover entirely. (Same pattern as circle-packing.)
    n_rects = len(data["rects"])
    for slot_idx in range(MAX_RECTS):
        if slot_idx >= n_rects:
            st.empty()
            continue
        rid = data["rects"][slot_idx]
        idx = slot_idx + 1
        cols = st.columns(_editor_cols, vertical_alignment="center")
        color = _PALETTE[(int(idx) - 1) % len(_PALETTE)]
        cols[0].markdown(
            f'<div style="display:inline-flex;align-items:center;'
            f'justify-content:center;width:1.6rem;height:1.6rem;'
            f'border-radius:0.3rem;background:{color};color:#fff;'
            f'font-weight:700;font-size:0.85rem;">{idx}</div>',
            unsafe_allow_html=True,
        )
        # Integer inputs by construction — the S2 degeneracy-breaking
        # disjuncts in build_model rely on integer dimensions (their +1
        # would wrongly cut fractional-data packings), so the editor
        # guarantees integrality rather than assuming it.
        new_w = cols[1].number_input(
            "Width", min_value=1, max_value=int(_W), step=1,
            value=int(data["w"][rid]),
            key=f"w_{rid}_{ver}",
            label_visibility="collapsed",
        )
        new_l = cols[2].number_input(
            "Length", min_value=1, max_value=30, step=1,
            value=int(data["length"][rid]),
            key=f"l_{rid}_{ver}",
            label_visibility="collapsed",
        )
        cols[3].button(
            "🗑", key=f"del_{rid}_{ver}",
            on_click=_delete_rect, args=(rid,),
        )
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
                else f"Max {MAX_RECTS} rectangles."
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
                r"$x_i \ge 0$ near corner of rectangle $i$ along the length"
                "  \n"
                r"$y_i \ge 0$ near corner across the width" "  \n"
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
\text{s.t.} \quad x_i + \ell_i \le L \quad \forall i \in \mathcal{I} \\
y_i + w_i \le W \quad \forall i \in \mathcal{I} \\
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
            "the left of, right of, below, or above rectangle $j$. The "
            "four-way disjunction is the GDP strip-packing model of "
            "Sawaya & Grossmann [4]:"
        )
        st.latex(
            r"""
            \begin{bmatrix} x_i + \ell_i \le x_j \end{bmatrix}
            \;\vee\;
            \begin{bmatrix} x_j + \ell_j \le x_i \end{bmatrix}
            \;\vee\;
            \begin{bmatrix} y_i + w_i \le y_j \\
                            x_i + \ell_i \ge x_j + 1 \\
                            x_j + \ell_j \ge x_i + 1 \end{bmatrix}
            \;\vee\;
            \begin{bmatrix} y_j + w_j \le y_i \\
                            x_i + \ell_i \ge x_j + 1 \\
                            x_j + \ell_j \ge x_i + 1 \end{bmatrix}
            \quad \forall i < j
            """
        )
        st.markdown(
            "The two extra inequalities in the above/below disjuncts are "
            "the degeneracy-breaking refinement of Trespalacios & "
            "Grossmann [5]. In the plain four-disjunct model the regions "
            "*overlap*: a pair separated both across the width and along "
            "the length satisfies two different disjuncts, so the same "
            "physical packing has multiple Boolean encodings and "
            "branch-and-bound explores each one. Requiring the "
            "above/below disjuncts to also overlap *lengthwise* routes "
            "every such arrangement uniquely through left/right; the $+1$ "
            "(rather than $\\ge 0$) closes the edge-flush tie as well, "
            "and is valid because all dimensions here are integer — the "
            "editor only accepts integer values."
        )

        st.markdown("**Symmetry breaking**")
        st.markdown(
            "The model breaks three distinct symmetry layers. The first "
            "— *encoding degeneracy*, where one physical packing has "
            "several valid disjunct selections — is handled by the extra "
            "lengthwise-overlap inequalities built into the disjunction "
            "above [5]. The remaining two are geometric and handled by "
            "dedicated constraints below."
        )
        st.markdown(
            "Every packing has mirror images — reflect it across the "
            "strip's centerline or along its length and the strip length "
            "$L$ is unchanged — so branch-and-bound would prove the same "
            "bound separately for each reflection. Pinning the *center* "
            "of one reference rectangle $r$ (the largest by area) into "
            "the lower-left quadrant keeps exactly one representative of "
            "each mirror class without cutting off the optimum — a "
            "standard domain-reduction device from the exact "
            "strip-packing literature:"
        )
        st.latex(
            r"x_r + \tfrac{\ell_r}{2} \le \tfrac{L}{2}, "
            r"\qquad y_r + \tfrac{w_r}{2} \le \tfrac{W}{2}"
        )
        st.markdown(
            "Rectangles with identical dimensions are also "
            "interchangeable — swapping two identical pieces gives a "
            '"different" solution with the same $L$ — so the members of '
            "each identical group are ordered along the length:"
        )
        st.latex(
            r"x_i \le x_j \quad \text{for consecutive } i < j"
            r"\ \text{ within each group of identical rectangles}"
        )
        st.markdown(
            "The reference rectangle's group is exempt from the "
            "ordering — pinning $r$'s position *and* forcing its place "
            "in an ordering could jointly cut off every optimum."
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

        st.markdown("**Solution method**")
        st.markdown(
            "Once the GDP is reformulated to a MILP, the selected solver "
            "runs branch-and-bound: relax the binary disjunct indicators "
            "$z_k$ to $[0, 1]$, solve the resulting LP, and either accept "
            "the solution if all indicators are integer or branch on a "
            "fractional one. Either solver runs under the selected time cap "
            "(10 s default) — if it can't "
            "prove optimality in that time, the Optimizer tab labels the "
            "result **Best length** instead of *Optimal length* and "
            "surfaces the remaining optimality **Gap**. HiGHS is a modern "
            "open-source LP/MILP solver from Edinburgh's ERGO group, "
            "distributed as a pip wheel via `highspy`; Gurobi is the "
            "commercial benchmark. Same model, same time cap — so the Gap "
            "each leaves is a fair head-to-head."
        )
        st.markdown(
            "See the [companion Jupyter notebook]"
            "(https://github.com/devin-griff/strip_packing/blob/main/Strip%20packing.ipynb) "
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
            "[4] N. W. Sawaya and I. E. Grossmann, "
            '"A cutting plane method for solving linear generalized '
            'disjunctive programming problems," *Computers & Chemical '
            "Engineering*, vol. 29, no. 9, pp. 1891–1913, 2005. "
            "[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0098135405000992)"
        )
        st.markdown(
            "[5] F. Trespalacios and I. E. Grossmann, "
            '"Symmetry breaking for generalized disjunctive programming '
            'formulation of the strip packing problem," *Annals of '
            "Operations Research*, vol. 258, pp. 747–759, 2017. "
            "[Springer](https://link.springer.com/article/10.1007/s10479-016-2112-9)"
        )
        st.markdown(
            "[6] Q. Huangfu and J. A. J. Hall, "
            '"Parallelizing the dual revised simplex method," *Mathematical '
            "Programming Computation*, vol. 10, no. 1, pp. 119–142, 2018. "
            "[Springer](https://link.springer.com/article/10.1007/s12532-017-0130-5)"
        )
        st.markdown(
            "[7] Gurobi Optimization, LLC, *Gurobi Optimizer Reference "
            "Manual*, 2026. [gurobi.com](https://www.gurobi.com)"
        )
        st.markdown(
            "[8] M. L. Bynum, G. A. Hackebeil, W. E. Hart, C. D. Laird, "
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
            f"**Total area $\\sum_i w_i \\ell_i$** &nbsp; {area:g}  \n"
            f"**Lower bound on L** &nbsp; "
            f"$\\max(\\max_i \\ell_i,\\ \\sum_i w_i \\ell_i / W) = {L_lb:g}$  \n"
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
                rf"& \underbrace{{x_{i_} + {li:g} \le x_{j_}}}_{{{i_}\text{{ left of }}{j_}}} \\"
                rf"\vee & \underbrace{{x_{j_} + {lj:g} \le x_{i_}}}_{{{j_}\text{{ left of }}{i_}}} \\"
                rf"\vee & \underbrace{{y_{i_} + {wi:g} \le y_{j_} \;\wedge\; "
                rf"x_{i_} + {li:g} \ge x_{j_} + 1 \;\wedge\; "
                rf"x_{j_} + {lj:g} \ge x_{i_} + 1}}"
                rf"_{{{i_}\text{{ below }}{j_}\text{{, lengthwise overlap}}}} \\"
                rf"\vee & \underbrace{{y_{j_} + {wj:g} \le y_{i_} \;\wedge\; "
                rf"x_{i_} + {li:g} \ge x_{j_} + 1 \;\wedge\; "
                rf"x_{j_} + {lj:g} \ge x_{i_} + 1}}"
                rf"_{{{j_}\text{{ below }}{i_}\text{{, lengthwise overlap}}}}"
                r"\end{array}"
            )
            st.caption(
                f"This is one of the {n_disj} pairwise disjunctions in "
                "the model; the GDP transformation rewrites each one "
                "into standard MILP constraints (Big-M / Hull). The "
                "lengthwise-overlap inequalities in the last two terms "
                "are the degeneracy-breaking refinement — without them, "
                "a diagonally-separated pair satisfies two terms at "
                "once, and the solver explores the same packing under "
                "multiple encodings."
            )

        # Symmetry-breaking pin, instantiated with the reference rectangle
        # build_model actually picks (largest area), numbers substituted.
        rects = data["rects"]
        ref = max(
            rects,
            key=lambda i: float(data["w"][i]) * float(data["length"][i]),
        )
        r_ = int(ref)
        wr = float(data["w"][ref])
        lr = float(data["length"][ref])
        W_val = float(data["W"])
        st.markdown("---")
        st.markdown(
            rf"**Symmetry breaking (instantiated)** &nbsp; rectangle "
            rf"${r_}$ is the largest by area "
            rf"($w_{r_} \cdot \ell_{r_} = {wr:g} \times {lr:g} = "
            rf"{wr * lr:g}$), so its center is pinned to the lower-left "
            "quadrant:"
        )
        st.latex(
            rf"x_{{{r_}}} + {lr / 2.0:g} \le \tfrac{{L}}{{2}}, "
            rf"\qquad y_{{{r_}}} + {wr / 2.0:g} \le {W_val / 2.0:g}"
        )
        st.caption(
            "Each inequality discards one mirror image of every packing "
            "(reflection across the strip's centerline / along its "
            "length), shrinking the branch-and-bound tree without "
            "cutting off the optimum."
        )

        # Identical-rectangle orderings, instantiated. Mirrors the
        # grouping logic in build_model (including the reference-group
        # exemption) so this display always matches the live model.
        groups = {}
        for i in rects:
            key = (float(data["w"][i]), float(data["length"][i]))
            groups.setdefault(key, []).append(i)
        chains = {
            dims: members for dims, members in groups.items()
            if len(members) >= 2 and ref not in members
        }
        exempt = {
            dims: members for dims, members in groups.items()
            if len(members) >= 2 and ref in members
        }
        st.markdown("---")
        st.markdown("**Identical-rectangle ordering (instantiated)**")
        if chains:
            for (w_, l_), members in sorted(chains.items()):
                ids = ", ".join(str(int(a)) for a in members)
                st.markdown(
                    rf"Rectangles {ids} all measure "
                    rf"${w_:g} \times {l_:g}$, so they are ordered along "
                    "the length:"
                )
                st.latex(
                    r" \le ".join(rf"x_{{{int(a)}}}" for a in members)
                )
        for (w_, l_), members in sorted(exempt.items()):
            ids = ", ".join(str(int(a)) for a in members)
            st.caption(
                f"Rectangles {ids} ({w_:g} × {l_:g}) form an identical "
                f"group containing the pinned reference rectangle {r_}, "
                "so this group is exempt from ordering — combining the "
                "pin with an ordering could cut off every optimum."
            )
        if not chains and not exempt:
            st.markdown(
                "*No two rectangles share identical dimensions in this "
                "instance, so no ordering constraints are active — give "
                "two rectangles the same width and length to see the "
                "chain appear here.*"
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
    " + "
    "<a href='https://www.gurobi.com' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>Gurobi</a>"
    "</span>"
    "</h2>",
    unsafe_allow_html=True,
)
_caption_col, _ = st.columns([5, 3])
with _caption_col:
    st.markdown(
        "Pack $N$ rectangles into a strip of fixed width $W$ to minimize "
        "the strip length $L$. Edit the object data directly on the "
        "Optimizer tab. Pick a GDP transformation, solver, and time "
        "limit, then click **Solve**. The **Gap** is returned if the "
        "time limit is reached. The **Formulation** and **Logs** tabs "
        "show the underlying GDP and solver output."
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
