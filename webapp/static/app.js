/* StudyPlanner web app front end. Talks to webapp/server.py. */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const A_COLORS = ["--a1", "--a2", "--a3"];
  const acol = (i) => css(A_COLORS[i % 3]);

  /* ---------------------------------------------------------- state & form */
  const state = { presets: {}, preset: "example", strategy: "astar", csp: "ac3", predictor: "nn", last: null };
  const sliders = ["target", "slots", "prev", "cps"];

  function setSlider(id, v) { $("#" + id).value = v; $("#" + id + "-o").textContent = v; }

  function renderAssignments(list) {
    $("#assignments").innerHTML = list.map((a, i) => `
      <div class="assign" data-i="${i}">
        <span class="tagA" style="background:${acol(i)}">A${i + 1}</span>
        <label>type<select data-k="kind"><option ${a.kind === "TMA" ? "selected" : ""}>TMA</option><option ${a.kind === "CMA" ? "selected" : ""}>CMA</option></select></label>
        <label>weight<input type="number" data-k="weight" min="0" max="100" value="${a.weight}"></label>
        <label>due week<input type="number" data-k="deadline_week" min="1" max="30" value="${a.deadline_week}"></label>
      </div>`).join("");
    $$("#assignments input, #assignments select").forEach((el) => el.addEventListener("input", () => markCustom()));
  }

  function loadPreset(name) {
    state.preset = name;
    $$(".preset").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.preset === name)));
    if (name === "custom") return;
    const p = state.presets[name];
    setSlider("target", p.target); setSlider("slots", p.slots_per_week);
    setSlider("prev", p.prev_score_mean); setSlider("cps", p.clicks_per_session);
    renderAssignments(p.assignments);
  }

  function markCustom() {
    if (state.preset !== "custom") {
      state.preset = "custom";
      $$(".preset").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.preset === "custom")));
    }
  }

  function readProfile() {
    return {
      target: +$("#target").value,
      slots_per_week: +$("#slots").value,
      prev_score_mean: +$("#prev").value,
      clicks_per_session: +$("#cps").value,
      assignments: $$("#assignments .assign").map((row) => ({
        kind: $("[data-k=kind]", row).value,
        weight: +$("[data-k=weight]", row).value,
        deadline_week: Math.round(+$("[data-k=deadline_week]", row).value),
      })),
    };
  }

  function setSeg(name, v) {
    state[name] = v;
    $$(`.seg[data-name=${name}] button`).forEach((b) => b.classList.toggle("on", b.dataset.v === v));
  }

  /* ---------------------------------------------------------- API */
  async function api(path, body) {
    const res = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
    let data;
    try { data = await res.json(); } catch { throw new Error(`server answered ${res.status}`); }
    if (!res.ok) throw new Error(data.error || `server answered ${res.status}`);
    return data;
  }

  function show(which) {
    ["#empty", "#loading", "#error", "#out"].forEach((s) => { $(s).hidden = s !== which; });
  }

  async function plan() {
    const btn = $("#go");
    btn.disabled = true; show("#loading");
    try {
      const r = await api("/api/plan", { profile: readProfile(), strategy: state.strategy, csp: state.csp, predictor: state.predictor });
      state.last = r;
      render(r);
      show("#out");
      $("#cmp-out").innerHTML = "";
    } catch (e) {
      $("#error").innerHTML = `<b>Error</b><p>${esc(e.message)}</p><p>Is the server still running? Start it with <code>python -m webapp</code>.</p>`;
      show("#error");
    } finally { btn.disabled = false; }
  }

  /* ---------------------------------------------------------- rendering */
  const fmt = (x, d = 1) => (x == null ? "-" : Number(x).toFixed(d));

  function render(r) {
    const b = $("#banner");
    b.className = "banner" + (r.found ? "" : " fail");
    const planTxt = r.plan ? `(${r.plan.join(", ")})` : "none";
    b.innerHTML = `<div><b>${r.found ? "Plan found" : "No solution"}</b>
      <p>${r.found ? `Plan ${planTxt}: ${r.sessions} session${r.sessions === 1 ? "" : "s"} reach the target of ${fmt(r.target, 0)}.`
                   : `The target of ${fmt(r.target, 0)} cannot be reached with this capacity. The closest plan, ${planTxt}, is shown below.`}</p></div>`;

    $("#s-sessions").textContent = r.plan ? r.sessions : "-";
    $("#s-nodes").textContent = r.nodes_expanded;
    $("#s-csp").textContent = r.csp_checked;
    $("#s-reject").textContent = r.nn_yes_csp_no_count;
    $("#s-time").textContent = `${fmt(r.seconds, 2)}s`;

    $("#marks").innerHTML = marksChart(r);
    $("#timetable").innerHTML = timetable(r);
    $("#tt-note").innerHTML = r.schedule_ok
      ? r.profile.assignments.map((a, i) => `<span><i class="sw" style="background:${acol(i)}"></i>${a.id} due end of week ${a.deadline_week}</span>`).join("")
      : "No timetable: the plan could not be placed in the free slots.";
    $("#whatif").innerHTML = whatifChart(r);
    $("#whatif-legend").innerHTML = r.whatif.map((w, i) => `<span><i class="sw" style="background:${acol(i)}"></i>${w.assignment}</span>`).join("")
      + `<span><i class="sw tgt"></i>target</span><span>ringed point: sessions in the chosen plan</span>`;
    $("#t-csp").innerHTML = cspTable(r);
    $("#t-trace").innerHTML = traceChart(r);
    $("#t-rej").innerHTML = rejected(r);
  }

  function marksChart(r) {
    const W = 520, H = 280, L = 40, B = 40, T = 16, R = 10;
    const n = r.profile.assignments.length, gw = (W - L - R) / n;
    const y = (m) => T + (H - T - B) * (1 - Math.max(0, Math.min(100, m)) / 100);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Predicted marks per assignment">`;
    for (let m = 0; m <= 100; m += 20) {
      s += `<rect x="${L}" y="${y(m)}" width="${W - L - R}" height="1" fill="${css("--rule")}"/>`;
      s += `<text x="${L - 8}" y="${y(m) + 5}" text-anchor="end" font-size="16">${m}</text>`;
    }
    r.profile.assignments.forEach((a, i) => {
      const x0 = L + i * gw + gw * 0.14, bw = gw * 0.32;
      const bars = [[r.baseline?.[i], css("--base"), x0], [r.predicted?.[i], acol(i), x0 + bw + 6]];
      bars.forEach(([v, col, x]) => {
        if (v == null) return;
        const h = Math.max(1, y(0) - y(v));
        s += `<rect x="${x}" y="${y(v)}" width="${bw}" height="${h}" rx="4" fill="${col}"/>`;
      });
      if (r.predicted) s += `<text x="${x0 + bw * 1.5 + 6}" y="${y(r.predicted[i]) - 6}" text-anchor="middle" font-size="18" fill="${css("--ink")}">${fmt(r.predicted[i])}</text>`;
      s += `<text x="${x0 + bw + 3}" y="${H - 18}" text-anchor="middle" font-size="18" fill="${css("--ink")}">${a.id}</text>`;
      if (r.plan) s += `<text x="${x0 + bw + 3}" y="${H - 2}" text-anchor="middle" font-size="16">${r.plan[i]} sessions</text>`;
    });
    s += `<line x1="${L}" x2="${W - R}" y1="${y(r.target)}" y2="${y(r.target)}" stroke="${css("--accent")}" stroke-width="2" stroke-dasharray="6 5"/>`;
    return s + "</svg>";
  }

  function timetable(r) {
    const as = r.profile.assignments;
    const weeks = Math.max(...as.map((a) => a.deadline_week));
    const slots = r.profile.slots_per_week;
    const idx = Object.fromEntries(as.map((a, i) => [a.id, i]));
    const cell = {};
    r.timetable.forEach((t) => { cell[`${t.week}-${t.slot}`] = t; });
    const due = new Set(as.map((a) => a.deadline_week));
    let h = `<table class="tt"><thead><tr><th></th>`;
    for (let w = 1; w <= weeks; w++) h += `<th class="${due.has(w) ? "due" : ""}">W${w}${due.has(w) ? "<span>due</span>" : ""}</th>`;
    h += `</tr></thead><tbody>`;
    for (let s = 1; s <= slots; s++) {
      h += `<tr><th>S${s}</th>`;
      for (let w = 1; w <= weeks; w++) {
        const t = cell[`${w}-${s}`];
        h += t ? `<td class="s" style="background:${acol(idx[t.assignment])}" title="${t.assignment} session ${t.session}, week ${w}, slot ${s}">${t.assignment}</td>`
               : `<td title="free: week ${w}, slot ${s}"></td>`;
      }
      h += `</tr>`;
    }
    return h + `</tbody></table>`;
  }

  function whatifChart(r) {
    const W = 1000, H = 300, L = 44, B = 36, T = 14, R = 14;
    const N = r.whatif[0]?.marks.length - 1 || 10;
    const x = (k) => L + (W - L - R) * k / N;
    const all = r.whatif.flatMap((w) => w.marks).concat([r.target]);
    let lo = Math.max(0, Math.floor((Math.min(...all) - 5) / 10) * 10), hi = Math.min(100, Math.ceil((Math.max(...all) + 5) / 10) * 10);
    if (hi - lo < 20) hi = Math.min(100, lo + 20);
    const y = (m) => T + (H - T - B) * (1 - (m - lo) / (hi - lo));
    let s = `<svg class="wide" viewBox="0 0 ${W} ${H}" role="img" aria-label="Predicted mark as sessions increase">`;
    for (let m = lo; m <= hi; m += 10) {
      s += `<rect x="${L}" y="${y(m)}" width="${W - L - R}" height="1" fill="${css("--rule")}"/>`;
      s += `<text x="${L - 8}" y="${y(m) + 5}" text-anchor="end" font-size="16">${m}</text>`;
    }
    for (let k = 0; k <= N; k++) s += `<text x="${x(k)}" y="${H - 10}" text-anchor="middle" font-size="16">${k}</text>`;
    s += `<line x1="${L}" x2="${W - R}" y1="${y(r.target)}" y2="${y(r.target)}" stroke="${css("--accent")}" stroke-width="2" stroke-dasharray="6 5"/>`;
    r.whatif.forEach((w, i) => {
      const col = acol(i);
      let d = `M${x(0)} ${y(w.marks[0])}`;
      w.marks.forEach((m, k) => { if (k) d += ` L${x(k)} ${y(m)}`; });
      s += `<path d="${d}" fill="none" stroke="${col}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>`;
      w.marks.forEach((m, k) => {
        const big = r.plan && r.plan[i] === k;
        s += `<circle cx="${x(k)}" cy="${y(m)}" r="${big ? 7 : 3.5}" fill="${big ? css("--card") : col}" ${big ? `stroke="${col}" stroke-width="3"` : ""}><title>${w.assignment}: ${k} sessions gives ${fmt(m)}</title></circle>`;
      });
    });
    return s + "</svg>";
  }

  function cspTable(r) {
    if (!r.csp_compare.length) return "<p>No plan to schedule.</p>";
    const max = Math.max(1, ...r.csp_compare.map((c) => c.assignments));
    return `<p>Plan (${r.csp_plan.join(", ")}) scheduled by each solver${r.nn_yes_csp_no_count ? " (the first plan the NN accepted but the CSP rejected)" : ""}.</p>
      <div class="scroll"><table class="data"><thead><tr><th>SOLVER</th><th>RESULT</th><th>ASSIGNMENTS</th><th>BACKTRACKS</th><th>PRUNED</th><th>MS</th></tr></thead><tbody>
      ${r.csp_compare.map((c) => `<tr class="${c.mode === r.csp ? "cur" : ""}"><td>${c.label}</td>
        <td class="${c.ok ? "ok" : "no"}">${c.ok ? "scheduled" : "rejected"}</td>
        <td><span class="minibar" style="width:${Math.max(2, 120 * c.assignments / max)}px"></span>${c.assignments}</td>
        <td>${c.backtracks}</td><td>${c.pruned}</td><td>${fmt(c.ms, 2)}</td></tr>`).join("")}
      </tbody></table></div>`;
  }

  function traceChart(r) {
    const tr = r.trace.filter((v) => v != null);
    if (!tr.length) return "<p>No trace recorded.</p>";
    const W = 1000, H = 220, L = 50, B = 30, T = 10, R = 10;
    const lo = Math.min(0, ...tr), hi = Math.max(0, ...tr);
    const span = hi - lo || 1;
    const x = (k) => L + (W - L - R) * k / Math.max(1, tr.length - 1);
    const y = (m) => T + (H - T - B) * (1 - (m - lo) / span);
    let s = `<p>Lowest (predicted mark - target) for each plan the search expanded. It must reach 0 for a plan to pass the NN check.</p>
      <div class="scroll"><svg class="wide" viewBox="0 0 ${W} ${H}" role="img" aria-label="Search trace">`;
    s += `<rect x="${L}" y="${y(0)}" width="${W - L - R}" height="2" fill="${css("--accent")}"/>`;
    s += `<text x="${L - 8}" y="${y(hi) + 12}" text-anchor="end" font-size="16">${fmt(hi, 0)}</text><text x="${L - 8}" y="${y(lo)}" text-anchor="end" font-size="16">${fmt(lo, 0)}</text>`;
    s += `<text x="${L - 8}" y="${y(0) + 5}" text-anchor="end" font-size="16">0</text>`;
    const step = Math.max(1, Math.floor(tr.length / 400));
    for (let k = 0; k < tr.length; k += step) s += `<circle cx="${x(k)}" cy="${y(tr[k])}" r="2.5" fill="${tr[k] >= 0 ? css("--ok") : css("--a1")}"/>`;
    s += `<text x="${W - R}" y="${H - 6}" text-anchor="end" font-size="16">${tr.length} nodes expanded</text>`;
    return s + "</svg></div>";
  }

  function rejected(r) {
    if (!r.nn_yes_csp_no_count) return "<p>None: every plan the NN accepted also fitted in the timetable.</p>";
    return `<p>${r.nn_yes_csp_no_count} plan(s) reached the target in the NN but could not be placed in the free slots before the deadlines. First ${r.nn_yes_csp_no.length}:</p>
      <div class="chips">${r.nn_yes_csp_no.map((p) => `<span class="chip">(${p.join(", ")})</span>`).join("")}</div>`;
  }

  async function compare() {
    const btn = $("#cmp-go");
    btn.disabled = true;
    $("#cmp-out").innerHTML = `<div class="loader"><i></i><i></i><i></i><i></i><i></i></div>`;
    try {
      const r = await api("/api/compare", { profile: readProfile(), strategy: state.strategy, csp: state.csp, predictor: state.predictor });
      const max = Math.max(1, ...r.rows.map((x) => x.nodes_expanded));
      const names = { ucs: "UCS", greedy: "Greedy", astar: "A*", nn: "Neural net", avg: "Average" };
      $("#cmp-out").innerHTML = `<div class="scroll"><table class="data"><thead><tr><th>MARKS</th><th>SEARCH</th><th>RESULT</th><th>PLAN</th><th>SESSIONS</th><th>NODES EXPANDED</th><th>TIME</th></tr></thead><tbody>
        ${r.rows.map((x) => `<tr><td>${names[x.predictor]}</td><td>${names[x.strategy]}</td>
          <td class="${x.found ? "ok" : "no"}">${x.found ? "found" : "no solution"}</td>
          <td>${x.plan ? "(" + x.plan.join(", ") + ")" : "-"}</td><td>${x.sessions ?? "-"}</td>
          <td><span class="minibar" style="width:${Math.max(2, 140 * x.nodes_expanded / max)}px"></span>${x.nodes_expanded}</td>
          <td>${fmt(x.seconds, 2)}s</td></tr>`).join("")}</tbody></table></div>
        <p class="legend">UCS and A* always return the fewest sessions. Greedy is fastest but has no such guarantee. The average predictor ignores study, so extra sessions never raise its marks.</p>`;
    } catch (e) {
      $("#cmp-out").innerHTML = `<p class="no">${esc(e.message)}</p>`;
    } finally { btn.disabled = false; }
  }

  /* ---------------------------------------------------------- wiring */
  async function init() {
    sliders.forEach((id) => $("#" + id).addEventListener("input", (e) => { $("#" + id + "-o").textContent = e.target.value; markCustom(); }));
    $$(".preset").forEach((b) => b.addEventListener("click", () => loadPreset(b.dataset.preset)));
    $$(".seg").forEach((seg) => $$("button", seg).forEach((b) => b.addEventListener("click", () => setSeg(seg.dataset.name, b.dataset.v))));
    $$(".tab").forEach((t) => t.addEventListener("click", () => {
      $$(".tab").forEach((x) => x.classList.toggle("active", x === t));
      $$(".tabpane").forEach((p) => { p.hidden = p.id !== t.dataset.tab; });
    }));
    $("#go").addEventListener("click", plan);
    $("#cmp-go").addEventListener("click", compare);
    document.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) plan(); });
    setSeg("strategy", "astar"); setSeg("csp", "ac3"); setSeg("predictor", "nn");
    try {
      const p = await api("/api/presets");
      state.presets = p.presets;
      $("#course").textContent = p.course;
      loadPreset("example");
    } catch (e) {
      $("#error").innerHTML = `<b>Cannot reach the server</b><p>${esc(e.message)}</p><p>Start it from the project folder with <code>python -m webapp</code> and open the address it prints.</p>`;
      show("#error");
    }
  }
  init();
})();
