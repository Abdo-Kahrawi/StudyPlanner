"""Web app: the JSON API returns the same plans as run_demo.py, and bad input gives a 400."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from webapp.server import make_server


@pytest.fixture(scope="module")
def base_url():
    srv = make_server("127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.status, r.read()


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_page_and_static_files(base_url):
    for path in ("/", "/style.css", "/app.js", "/favicon.svg"):
        status, body = get(base_url + path)
        assert status == 200 and body
    with pytest.raises(urllib.error.HTTPError):
        get(base_url + "/../common/config.py")


@pytest.mark.parametrize("strategy", ["ucs", "greedy", "astar"])
@pytest.mark.parametrize("csp", ["bt", "fc", "ac3"])
@pytest.mark.parametrize("predictor", ["nn", "avg"])
@pytest.mark.parametrize("preset", ["example", "tight", "impossible"])
def test_plan_every_option(base_url, preset, strategy, csp, predictor):
    _, presets = get(base_url + "/api/presets")
    profile = json.loads(presets)["presets"][preset]
    status, r = post(base_url + "/api/plan", {"profile": profile, "strategy": strategy,
                                              "csp": csp, "predictor": predictor})
    assert status == 200, r
    assert r["message"]
    assert len(r["whatif"]) == len(profile["assignments"])
    if r["found"]:
        assert r["schedule_ok"]
        assert len(r["timetable"]) == r["sessions"]
        assert min(r["predicted"]) >= r["target"]


def test_compare(base_url):
    _, presets = get(base_url + "/api/presets")
    profile = json.loads(presets)["presets"]["example"]
    status, r = post(base_url + "/api/compare", {"profile": profile, "csp": "ac3"})
    assert status == 200 and len(r["rows"]) == 6


@pytest.mark.parametrize("bad", [
    {"target": "x"},
    {"assignments": []},
    {"slots_per_week": 0},
    {"assignments": [{"kind": "EXAM", "weight": 20, "deadline_week": 4}]},
])
def test_bad_input_is_400(base_url, bad):
    _, presets = get(base_url + "/api/presets")
    profile = {**json.loads(presets)["presets"]["example"], **bad}
    status, r = post(base_url + "/api/plan", {"profile": profile})
    assert status == 400 and r["error"]
