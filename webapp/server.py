"""StudyPlanner web app. Owner: Person B.

Run from the project folder (only the project's own requirements are needed):
    python -m webapp            # opens http://localhost:8000

The server is the Python standard library's http.server. It serves webapp/static/ and a
small JSON API around the same planner as run_demo.py:

    GET  /api/presets           the demo students (example, tight, impossible)
    POST /api/plan              run search + NN + CSP for one student and method
    POST /api/compare           run all three searches and both predictors on one student
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config  # noqa: E402
from common.types import Assignment, StudentProfile  # noqa: E402
from csp.solver import MODES, schedule  # noqa: E402
from eval import harness  # noqa: E402
from integration.planner import plan_for  # noqa: E402
from run_demo import get_profile  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
STRATEGIES = ("ucs", "greedy", "astar")
PREDICTORS = ("nn", "avg")
CSP_LABELS = {"bt": "BT", "fc": "BT+FC", "ac3": "BT+FC+AC-3"}
MAX_WEEKS = 30
_lock = threading.Lock()          # the planner and torch are not shared between requests
_cache: dict[str, object] = {}


def predictor(kind: str):
    if kind not in _cache:
        _cache[kind] = harness.load_predictor(kind)
    return _cache[kind]


def search_fn():
    if "search" not in _cache:
        _cache["search"] = harness.load_search()
    return _cache["search"]


# ---------------------------------------------------------------- JSON <-> profile
def profile_to_json(p: StudentProfile) -> dict:
    return {
        "target": p.target,
        "slots_per_week": p.slots_per_week,
        "prev_score_mean": float(p.history.get("prev_score_mean", 30.0)),
        "clicks_per_session": p.clicks_per_session,
        "assignments": [{"id": a.id, "kind": a.kind, "weight": a.weight, "deadline_week": a.deadline_week}
                        for a in p.assignments],
    }


class BadRequest(ValueError):
    pass


def _num(d: dict, key: str, lo: float, hi: float, cast=float):
    try:
        v = cast(d[key])
    except (KeyError, TypeError, ValueError):
        raise BadRequest(f"'{key}' is missing or not a number")
    if not lo <= v <= hi:
        raise BadRequest(f"'{key}' must be between {lo} and {hi}")
    return v


def profile_from_json(d: dict) -> StudentProfile:
    base = get_profile("example", real=False)
    raw = d.get("assignments") or []
    if not 1 <= len(raw) <= 3:
        raise BadRequest("give 1 to 3 assignments")
    assignments = []
    for i, a in enumerate(raw):
        kind = str(a.get("kind", "TMA")).upper()
        if kind not in ("TMA", "CMA"):
            raise BadRequest("assignment type must be TMA or CMA")
        assignments.append(Assignment(f"A{i + 1}", kind, _num(a, "weight", 0, 100),
                                      _num(a, "deadline_week", 1, MAX_WEEKS, int)))
    assignments.sort(key=lambda a: a.deadline_week)
    assignments = [dataclasses.replace(a, id=f"A{i + 1}") for i, a in enumerate(assignments)]
    return dataclasses.replace(
        base,
        target=_num(d, "target", 0, 100),
        slots_per_week=_num(d, "slots_per_week", 1, 7, int),
        history={"prev_score_mean": _num(d, "prev_score_mean", 0, 100)},
        clicks_per_session=_num(d, "clicks_per_session", 1, 500, int),
        assignments=tuple(assignments),
    )


def method_from_json(d: dict) -> tuple[str, str, str]:
    strategy, csp, kind = d.get("strategy", "astar"), d.get("csp", "ac3"), d.get("predictor", "nn")
    if strategy not in STRATEGIES or csp not in MODES or kind not in PREDICTORS:
        raise BadRequest("unknown strategy, csp or predictor")
    return strategy, csp, kind


# ---------------------------------------------------------------- planner calls
def _floats(xs):
    return None if xs is None else [round(float(x), 2) for x in xs]


def run_plan(profile: StudentProfile, strategy: str, csp: str, kind: str) -> dict:
    pred = predictor(kind)
    fc, ac3 = MODES[csp]
    out = plan_for(profile, pred, strategy, use_fc=fc, use_ac3=ac3, search_fn=search_fn())
    plan = list(out.plan) if out.plan is not None else None

    timetable = []
    if out.schedule and out.schedule.timetable:
        for (aid, k), (w, s) in sorted(out.schedule.timetable.items(), key=lambda kv: kv[1]):
            timetable.append({"assignment": aid, "session": k + 1, "week": w + 1, "slot": s + 1})

    n = len(profile.assignments)
    whatif = []
    for i, a in enumerate(profile.assignments):
        curve = [float(pred.predict(profile, tuple(m if j == i else 0 for j in range(n)))[i])
                 for m in range(config.MAX_SESSIONS_PER_ASSIGNMENT + 1)]
        whatif.append({"assignment": a.id, "marks": _floats(curve)})

    csp_plan = out.nn_yes_csp_no[0] if out.nn_yes_csp_no else out.plan
    csp_rows = []
    if csp_plan is not None:
        for mode, (m_fc, m_ac3) in MODES.items():
            t0 = time.perf_counter()
            r = schedule(csp_plan, profile, use_fc=m_fc, use_ac3=m_ac3)
            csp_rows.append({"mode": mode, "label": CSP_LABELS[mode], "ok": r.ok,
                             "assignments": r.assignments, "backtracks": r.backtracks,
                             "pruned": r.values_pruned, "ms": round((time.perf_counter() - t0) * 1000, 2)})

    trace = [t.get("min_margin") for t in out.search.trace]
    return {
        "found": out.found,
        "message": out.message,
        "plan": plan,
        "sessions": sum(plan) if plan else 0,
        "predicted": _floats(out.predicted),
        "baseline": _floats(pred.predict(profile, tuple([0] * n))),
        "target": profile.target,
        "nodes_expanded": out.search.nodes_expanded,
        "nodes_generated": out.search.nodes_generated,
        "csp_checked": len(out.csp_calls),
        "nn_yes_csp_no": [list(p) for p in out.nn_yes_csp_no[:20]],
        "nn_yes_csp_no_count": len(out.nn_yes_csp_no),
        "schedule_ok": bool(out.schedule and out.schedule.ok),
        "timetable": timetable,
        "whatif": whatif,
        "csp_plan": list(csp_plan) if csp_plan is not None else None,
        "csp_compare": csp_rows,
        "trace": [None if m is None else round(float(m), 2) for m in trace[:2000]],
        "seconds": round(out.seconds, 3),
        "predictor": getattr(pred, "name", kind),
        "strategy": strategy,
        "csp": csp,
        "profile": profile_to_json(profile),
    }


def run_compare(profile: StudentProfile, csp: str) -> dict:
    fc, ac3 = MODES[csp]
    rows = []
    for kind in PREDICTORS:
        pred = predictor(kind)
        for strategy in STRATEGIES:
            out = plan_for(profile, pred, strategy, use_fc=fc, use_ac3=ac3, search_fn=search_fn())
            rows.append({"predictor": kind, "strategy": strategy, "found": out.found,
                         "plan": list(out.plan) if out.plan is not None else None,
                         "sessions": sum(out.plan) if out.plan is not None else None,
                         "nodes_expanded": out.search.nodes_expanded,
                         "csp_checked": len(out.csp_calls), "seconds": round(out.seconds, 3)})
    return {"rows": rows, "csp": csp}


# ---------------------------------------------------------------- HTTP
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "application/javascript; charset=utf-8", ".svg": "image/svg+xml",
                 ".png": "image/png", ".ico": "image/x-icon"}


class Handler(BaseHTTPRequestHandler):
    server_version = "StudyPlanner/1.0"

    def log_message(self, fmt, *args):      # keep the console quiet except for errors
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/presets":
            presets = {name: profile_to_json(get_profile(name, real=False))
                       for name in ("example", "tight", "impossible")}
            return self._json(200, {"presets": presets, "course": config.COURSE,
                                    "max_sessions": config.MAX_SESSIONS_PER_ASSIGNMENT})
        if path == "/api/health":
            return self._json(200, {"ok": True})
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        f = (STATIC / rel).resolve()
        if STATIC.resolve() not in f.parents or not f.is_file():
            return self._send(404, b"not found", "text/plain")
        self._send(200, f.read_bytes(), CONTENT_TYPES.get(f.suffix, "application/octet-stream"))

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 100_000:
                raise BadRequest("request too large")
            body = json.loads(self.rfile.read(length) or b"{}")
            profile = profile_from_json(body.get("profile") or {})
            strategy, csp, kind = method_from_json(body)
            with _lock:
                if path == "/api/plan":
                    return self._json(200, run_plan(profile, strategy, csp, kind))
                if path == "/api/compare":
                    return self._json(200, run_compare(profile, csp))
            return self._json(404, {"error": "unknown endpoint"})
        except (BadRequest, json.JSONDecodeError) as e:
            return self._json(400, {"error": str(e)})
        except Exception as e:      # report planner errors to the page instead of dropping the request
            traceback.print_exc()
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})


def make_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    ap = argparse.ArgumentParser(description="StudyPlanner web app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    print("Loading the neural network and the search...")
    predictor("nn"), predictor("avg"), search_fn()
    srv = make_server(args.host, args.port)
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
    print(f"StudyPlanner is running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, (url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
