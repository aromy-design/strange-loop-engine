// Scientific dashboard — dense numeric view of agent + experiment results.

const el = (id) => document.getElementById(id);
const fmt = (x, d=2) => (x == null || Number.isNaN(x)) ? "—" : Number(x).toFixed(d);
const pct = (x) => Math.max(0, Math.min(100, x * 100));
const clear = (n) => { while (n.firstChild) n.removeChild(n.firstChild); };

// ---------- canvases ----------
const wcv = el("world"), wctx = wcv.getContext("2d");
const vcv = el("value"), vctx = vcv.getContext("2d");
const xcv = el("weights"), xctx = xcv.getContext("2d");
const lcv = el("learn-cv"), lctx = lcv.getContext("2d");

function drawWorld(s) {
  if (!s) return;
  const N = s.size, cell = wcv.width / N;
  const lit = s.lightLevel ?? 1;
  wctx.fillStyle = `rgb(${2+lit*8},${4+lit*12},${10+lit*18})`;
  wctx.fillRect(0, 0, wcv.width, wcv.height);
  wctx.strokeStyle = "rgba(122,215,255,0.05)"; wctx.lineWidth = 1;
  for (let i = 0; i <= N; i++) {
    wctx.beginPath(); wctx.moveTo(i*cell, 0); wctx.lineTo(i*cell, wcv.height); wctx.stroke();
    wctx.beginPath(); wctx.moveTo(0, i*cell); wctx.lineTo(wcv.width, i*cell); wctx.stroke();
  }
  for (const sg of s.signals) {
    const c = ["#3affc8","#ff3a3a","#ffd47a","#a87aff"][(sg.kind-1)%4] || "#7ad7ff";
    wctx.fillStyle = c; wctx.globalAlpha = 0.5;
    wctx.fillRect(sg.c*cell+2, sg.r*cell+2, cell-4, cell-4); wctx.globalAlpha = 1;
  }
  for (const f of s.food) {
    if (f.kind === 2) { wctx.fillStyle = "#fff8b8"; wctx.shadowColor = "#fff8b8"; wctx.shadowBlur = 8;
      wctx.beginPath(); wctx.arc(f.c*cell+cell/2, f.r*cell+cell/2, cell*0.32, 0, Math.PI*2); wctx.fill();
      wctx.shadowBlur = 0;
    } else { wctx.fillStyle = "#e8eef7";
      wctx.beginPath(); wctx.arc(f.c*cell+cell/2, f.r*cell+cell/2, cell*0.20, 0, Math.PI*2); wctx.fill();
    }
  }
  const [ar, ac] = s.agent;
  wctx.fillStyle = "#ff5d8f"; wctx.shadowColor = "#ff5d8f"; wctx.shadowBlur = 10;
  wctx.beginPath(); wctx.arc(ac*cell+cell/2, ar*cell+cell/2, cell*0.34, 0, Math.PI*2); wctx.fill();
  wctx.shadowBlur = 0;
}

function drawValueMap(matrix) {
  if (!matrix) return;
  const N = matrix.length, cell = vcv.width / N;
  vctx.fillStyle = "#02040a"; vctx.fillRect(0, 0, vcv.width, vcv.height);
  let mn = Infinity, mx = -Infinity;
  for (const row of matrix) for (const v of row) { if (v<mn)mn=v; if (v>mx)mx=v; }
  const range = Math.max(1e-4, mx - mn);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const norm = (matrix[i][j] - mn) / range;
    let r,g,b;
    if (norm < 0.4) { r = 80 + 175*(0.4-norm)/0.4; g=30; b=60; }
    else if (norm > 0.6) { r=30; g = 100 + 155*(norm-0.6)/0.4; b = 200 + 55*(norm-0.6)/0.4; }
    else { r=60; g=80; b=100; }
    vctx.fillStyle = `rgb(${Math.floor(r)},${Math.floor(g)},${Math.floor(b)})`;
    vctx.fillRect(j*cell, i*cell, cell+1, cell+1);
  }
}

function drawWeights(matrix) {
  if (!matrix) return;
  const N = matrix.length, cell = xcv.width / N;
  xctx.fillStyle = "#02040a"; xctx.fillRect(0, 0, xcv.width, xcv.height);
  let max = 0;
  for (const row of matrix) for (const v of row) if (v > max) max = v;
  if (max < 1e-6) max = 1;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const v = matrix[i][j] / max;
    xctx.fillStyle = `rgb(${Math.floor(122+v*100)},${Math.floor(215*v+30)},${Math.floor(255*v+40)})`;
    xctx.globalAlpha = Math.min(1, v + 0.15);
    xctx.fillRect(j*cell, i*cell, cell+1, cell+1);
  }
  xctx.globalAlpha = 1;
}

function drawLearning(curve) {
  const W = lcv.width, H = lcv.height;
  lctx.fillStyle = "#02040a"; lctx.fillRect(0, 0, W, H);
  if (!curve || !curve.buckets || curve.buckets.length < 2) return;
  let max = 1;
  for (const b of curve.buckets) { if (b.eats>max)max=b.eats; if (b.danger>max)max=b.danger; }
  for (const [k, c] of [["eats", "#7ad7ff"], ["danger", "#ff5d8f"]]) {
    lctx.strokeStyle = c; lctx.lineWidth = 1.4;
    lctx.beginPath();
    for (let i = 0; i < curve.buckets.length; i++) {
      const x = (i/(curve.buckets.length-1))*W;
      const y = H - (curve.buckets[i][k]/max)*(H-6) - 3;
      if (i===0) lctx.moveTo(x, y); else lctx.lineTo(x, y);
    }
    lctx.stroke();
  }
}

// ---------- sparklines ----------
const sparkBuffers = {};
function spark(id, val, color) {
  if (!sparkBuffers[id]) sparkBuffers[id] = [];
  sparkBuffers[id].push(val);
  if (sparkBuffers[id].length > 200) sparkBuffers[id].shift();
  const cv = el("sp-" + id);
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.fillStyle = "#02040a"; ctx.fillRect(0, 0, W, H);
  const arr = sparkBuffers[id];
  if (arr.length < 2) return;
  let mn = Math.min(...arr), mx = Math.max(...arr);
  const r = Math.max(1e-4, mx - mn);
  ctx.strokeStyle = color; ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (let i = 0; i < arr.length; i++) {
    const x = (i/(arr.length-1))*W;
    const y = H - ((arr[i]-mn)/r)*(H-2) - 1;
    if (i===0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// ---------- battery bars ----------
function renderBattery(c) {
  const box = el("batt-bars"); clear(box);
  const order = ["phiSubset","closureDepth","collapseIndex","identityPersistence","mirror","surpriseReact","alignment","causalDensity","temporalBinding","presence","awareness","ignitionRate","selfRecognition"];
  const labels = {phiSubset:"Φ subset", closureDepth:"closure", collapseIndex:"obs collapse", identityPersistence:"identity", mirror:"mirror", surpriseReact:"surprise", alignment:"sym→state", causalDensity:"causal", temporalBinding:"temporal", presence:"presence", awareness:"awareness", ignitionRate:"ignitions", selfRecognition:"self-recog"};
  for (const k of order) {
    if (c[k] === undefined) continue;
    const v = c[k];
    const row = document.createElement("div"); row.className = "bcomp";
    const sp = document.createElement("span"); sp.textContent = labels[k];
    const wr = document.createElement("div"); wr.className = "barwrap";
    const inner = document.createElement("div"); inner.style.width = pct(v) + "%";
    wr.appendChild(inner);
    const b = document.createElement("b"); b.textContent = fmt(v);
    row.appendChild(sp); row.appendChild(wr); row.appendChild(b);
    box.appendChild(row);
  }
}

// ---------- circuits table ----------
function renderCircuits(circuits) {
  const tb = el("circuits-tbl"); clear(tb);
  const order = ["vision","antennal","lateral_horn","mushroom","central_complex","subesophageal","motor"];
  const sizes = {vision:800, antennal:200, lateral_horn:150, mushroom:3000, central_complex:500, subesophageal:200, motor:300};
  for (const name of order) {
    const c = circuits[name];
    if (!c) continue;
    const tr = document.createElement("tr");
    tr.innerHTML = `<th>${name}</th><td>${sizes[name]}</td><td>${(c.firingRate*100).toFixed(2)}</td><td>${c.activeNow}</td>`;
    tb.appendChild(tr);
  }
}

// ---------- event log ----------
const eventLog = el("event-log");
let eventLogCount = 0;
function addEvent(text, cls) {
  const div = document.createElement("div");
  div.className = "ev-line " + cls;
  div.textContent = text;
  eventLog.insertBefore(div, eventLog.firstChild);
  eventLogCount++;
  while (eventLog.children.length > 30) eventLog.removeChild(eventLog.lastChild);
}

// ---------- experiment loader ----------
async function loadExperiment(name) {
  const status = el("exp-status");
  status.textContent = `loading experiments/results/${name}/summary.csv ...`;
  try {
    const resp = await fetch(`/experiments/results/${name}/summary.csv`);
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const text = await resp.text();
    renderExperimentTable(text);
    status.textContent = `loaded ${name}`;
  } catch (e) {
    status.textContent = `error: ${e.message} — try running experiment first`;
  }
}

function renderExperimentTable(csv) {
  const lines = csv.trim().split(/\r?\n/);
  const header = lines[0].split(",");
  const rows = lines.slice(1).map(l => l.split(","));
  // group by condition (first column)
  const condIdx = header.indexOf("condition");
  if (condIdx < 0) {
    el("exp-table").textContent = "no 'condition' column";
    return;
  }
  const groups = {};
  for (const r of rows) {
    const c = r[condIdx];
    if (!groups[c]) groups[c] = [];
    groups[c].push(r);
  }
  // compute mean of numeric cols
  const numCols = ["total_eats","total_danger","deaths","final_mirror","final_awareness_idx","final_collapse_index","final_closure_depth"];
  const colIdx = numCols.map(c => header.indexOf(c)).filter(i => i >= 0);
  const html = ["<table>"];
  html.push("<tr><th>condition</th><th>n</th>");
  for (const c of numCols) if (header.indexOf(c) >= 0) html.push(`<th>${c.replace("final_","").replace("_"," ")}</th>`);
  html.push("</tr>");
  // sort: FULL first
  const condKeys = Object.keys(groups).sort((a,b) => (a==="FULL"?-1:1) - (b==="FULL"?-1:1));
  for (const cond of condKeys) {
    const g = groups[cond];
    html.push("<tr>");
    html.push(`<td>${cond}</td><td>${g.length}</td>`);
    for (const cidx of colIdx) {
      const vals = g.map(r => parseFloat(r[cidx])).filter(v => !Number.isNaN(v));
      const mean = vals.length ? (vals.reduce((a,b)=>a+b,0)/vals.length) : 0;
      html.push(`<td>${mean.toFixed(2)}</td>`);
    }
    html.push("</tr>");
  }
  html.push("</table>");
  el("exp-table").innerHTML = html.join("");
}

el("btn-load-exp").addEventListener("click", () => {
  loadExperiment(el("exp-select").value);
});

// ---------- ws ----------
let ws;
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (ev) => {
    const d = JSON.parse(ev.data);
    const m = d.metrics || {}, mind = d.mind || {}, w = d.world || {}, ib = d.insectBrain || {};
    const md = d.mode || {}, sl = d.sleep || {}, nm = d.neuromods || {};
    const mt = d.mirrorTest || {}, aw = d.awareness || {}, batt = d.battery || {components:{}, composite:0};
    const val = d.validation || {trust:0}, meta = d.meta || {}, ws_ = d.workspace || {};
    const cont = d.continuity || {}, learning = d.learning || {};

    // header
    el("t").textContent = d.t;
    el("msStep").textContent = fmt(m.actionTemperature ? (1000/20).toFixed(1) : 50, 1);
    if (d.totalNeurons) {
      const n = d.totalNeurons;
      el("neuron-count").textContent = (n >= 1000 ? (n/1000).toFixed(1) + "k" : n) + (d.insectBrainVersion === "V2" ? " (V2 81k)" : "");
    }
    el("energy").textContent = fmt(m.energy, 2);
    el("light").textContent = (w.isNight ? "🌙 " : "☀️ ") + fmt(w.lightLevel, 2);
    el("mode").textContent = md.current || "—";
    el("pos").textContent = w.agent ? `${w.agent[0]},${w.agent[1]}` : "—";

    // composite
    el("composite").textContent = fmt(batt.composite, 3);
    el("trust").textContent = fmt(val.trust, 2);
    if (batt.components) renderBattery(batt.components);

    // circuits
    if (ib.circuits) renderCircuits(ib.circuits);

    // meta
    el("depth").textContent = (meta.closureDepth ?? 0) + "/3";
    el("err1").textContent = fmt(meta.err1, 3);
    el("err2").textContent = fmt(meta.err2, 3);
    el("err3").textContent = fmt(meta.err3, 3);

    // neuromod
    el("da").textContent = fmt(nm.dopamine, 2);
    el("ht").textContent = fmt(nm.serotonin, 2);
    el("ne").textContent = fmt(nm.norepinephrine, 2);
    el("bda").style.width = pct(Math.min(1, (nm.dopamine ?? 0)/2)) + "%";
    el("bht").style.width = pct(nm.serotonin ?? 0) + "%";
    el("bne").style.width = pct(Math.min(1, (nm.norepinephrine ?? 0)/3)) + "%";

    // sparklines
    spark("energy", m.energy ?? 0, "#3affc8"); el("sp-energy-v").textContent = fmt(m.energy, 2);
    spark("selfpe", m.selfPredictionError ?? 0, "#ff5d8f"); el("sp-selfpe-v").textContent = fmt(m.selfPredictionError, 2);
    spark("worldpe", m.predictionError ?? 0, "#ffd47a"); el("sp-worldpe-v").textContent = fmt(m.predictionError, 2);
    spark("mirror", mt.recognitionScore ?? 0, "#a87aff"); el("sp-mirror-v").textContent = fmt(mt.recognitionScore, 2);
    spark("aware", aw.awarenessIndex ?? 0, "#7ad7ff"); el("sp-aware-v").textContent = fmt(aw.awarenessIndex, 2);
    spark("collapse", (d.coupling||{}).collapseIndex ?? 0, "#3affc8"); el("sp-collapse-v").textContent = fmt((d.coupling||{}).collapseIndex, 2);
    spark("cont", cont.continuity ?? 0, "#a87aff"); el("sp-cont-v").textContent = fmt(cont.continuity, 2);
    spark("presence", m.presence ?? 0, "#ff5d8f"); el("sp-presence-v").textContent = fmt(m.presence, 2);

    // cumulative
    if (typeof window._cum === "undefined") window._cum = {eats:0, danger:0, shelter:0, sleep:0, deaths:0, wake:0};
    if (d.ate) window._cum.eats++;
    if (d.inDanger) window._cum.danger++;
    if (d.inShelter) window._cum.shelter++;
    if (sl.isSleeping) window._cum.sleep++;
    window._cum.deaths = m.deaths ?? 0;
    el("cum-eats").textContent = window._cum.eats;
    el("cum-danger").textContent = window._cum.danger;
    el("cum-shelter").textContent = window._cum.shelter;
    el("cum-sleep").textContent = window._cum.sleep;
    el("cum-deaths").textContent = window._cum.deaths;
    el("cum-wake").textContent = (d.awakening||{}).totalAwakenings ?? 0;

    // event log
    const ev = aw.currentEvent;
    if (ev === "DISSOCIATION") addEvent(`t=${d.t} DISSOCIATION (s=${aw.currentSelfZ.toFixed(1)}σ)`, "diss");
    else if (ev === "SELF_SURPRISE") addEvent(`t=${d.t} SELF_SURPRISE`, "self");
    else if (ev === "WORLD_SURPRISE") addEvent(`t=${d.t} WORLD_SURPRISE`, "world");
    if ((d.awakening||{}).awakened) addEvent(`t=${d.t} ⚡ AWAKENING (score=${d.awakening.currentScore}/4)`, "wake");

    // symbols
    if (d.symbolStream) {
      const box = el("symbols"); clear(box);
      for (let i = 0; i < d.symbolStream.length; i++) {
        const s = document.createElement("div");
        s.className = "sym" + (i === d.symbolStream.length-1 ? " fresh" : "");
        s.textContent = String(d.symbolStream[i]).padStart(2, "0");
        box.appendChild(s);
      }
    }

    // ignitions
    const ignBox = el("ign"); clear(ignBox);
    (ws_.recentIgnitions || []).slice().reverse().slice(0, 12).forEach((i, idx) => {
      const p = document.createElement("div"); p.className = "ign-pill" + (idx===0?" fresh":"");
      p.textContent = i.module; ignBox.appendChild(p);
    });

    // awareness totals
    el("aw-self").textContent = aw.totalEvents?.SELF_SURPRISE ?? 0;
    el("aw-world").textContent = aw.totalEvents?.WORLD_SURPRISE ?? 0;
    el("aw-diss").textContent = aw.totalEvents?.DISSOCIATION ?? 0;
    el("aw-idx").textContent = fmt(aw.awarenessIndex, 2);

    // mirror
    el("mirror").textContent = fmt(mt.recognitionScore, 3);
    el("mirror-runs").textContent = mt.runs ?? 0;

    drawWorld(w);
    drawValueMap(d.spatialHeatmap);
    drawWeights(d.weightHeatmap);
    drawLearning(learning);
  };
  ws.onclose = () => setTimeout(connect, 1500);
}
connect();

el("btn-perturb").addEventListener("click", () => ws?.send(JSON.stringify({type:"perturb"})));
el("btn-reset").addEventListener("click", () => {
  ws?.send(JSON.stringify({type:"reset"}));
  window._cum = {eats:0, danger:0, shelter:0, sleep:0, deaths:0, wake:0};
});
