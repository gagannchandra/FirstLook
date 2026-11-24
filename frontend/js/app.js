/* FirstLook web app. Assess (the form and a readiness checklist) and Result (the estimate) are separate pages.
   T1/T2 run in the browser (js/model.js); when the local backend answers (backend/server.py, same origin) the
   record model T3-lite runs through POST api/predict. */
"use strict";
const M = self.FirstLookModel, C = self.FirstLookCharts;
const NB = " ";
// key, label, unit, usual range, strip scale, soft (outer) range, inputmode
const VITAL = [
  ["triage_vital_hr", "Heart rate", "bpm", [60, 100], [30, 200], [40, 180], "numeric"],
  ["triage_vital_rr", "Respiratory rate", "/min", [12, 20], [6, 44], [10, 40], "numeric"],
  ["triage_vital_sbp", "Systolic BP", "mmHg", [100, 140], [60, 240], [70, 230], "numeric"],
  ["triage_vital_dbp", "Diastolic BP", "mmHg", [60, 90], [30, 140], [40, 120], "numeric"],
  ["triage_vital_o2", "SpO₂", "%", [95, 100], [78, 100], [80, 100], "numeric"],
  ["triage_vital_temp", "Temperature", "°F", [97, 99.5], [93, 106], [95, 104], "decimal"],
];
const ARR = { "Walk-in": "Walk-in", Car: "Car or taxi", ambulance: "Ambulance", Wheelchair: "Wheelchair",
  "Public Transportation": "Public transport", Police: "Police", Other: "Other" };
const CONF = {
  likely_admission: ["#i-bed", "Likely admission", "FirstLook is confident about this patient: its check points to admission."],
  likely_discharge: ["#i-home", "Likely discharge", "FirstLook is confident about this patient: its check points to going home."],
  uncertain: ["#i-split", "Unsure — use your judgement", "Patients like this one could go either way. Rely on your clinical judgement, not the percentage."],
};
const PAGES = ["home", "assess", "result", "history", "about"];
const LEVEL = { t1: "Triage data only", t2: "Triage data and triage level", t3lite: "Triage data, triage level and patient record" };
const STORE = "firstlook.form.v3";
let META = null, DASH = null, SERVER = null, selected = [], last = null, submitted = false, panelTimer = 0, sayTimer = 0, reqId = 0;
let undoFn = null, undoTimer = 0, showAll = false, savedSig = null, savedId = null;
const touched = new Set();
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const announce = (msg) => { $("live").textContent = ""; requestAnimationFrame(() => { $("live").textContent = msg; }); };
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const pct = (p) => Math.min(99, Math.max(1, Math.round(p * 100))); // never claim 0% or 100%: it is an estimate
const hid = (k) => "h_" + k.replace(/[^a-z0-9]/gi, "_");
const radio = (name) => document.querySelector(`input[name="${name}"]:checked`)?.value ?? null;
const setRadio = (name, v) => { for (const r of document.querySelectorAll(`input[name="${name}"]`)) r.checked = r.value === String(v ?? ""); };
const hourBin = (h) => { const b = META.hour_bins; return h >= 23 || h < 3 ? b[0] : b[1 + Math.floor((h - 3) / 4)]; };

/* ---------------------------------------------------------------- routing */
function route() {
  const p = (location.hash.match(/^#\/(\w+)/) ?? [])[1];
  const page = p === "evidence" ? "about" : p === "patient" ? "assess" : p === "explain" ? "result" : PAGES.includes(p) ? p : "home"; // old links still work
  for (const s of PAGES) $(s).hidden = s !== page;
  const navp = page === "result" ? "assess" : page;
  for (const a of $("nav").children) a.dataset.p === navp ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current");
  const name = { home: "Home", assess: "Assess", result: "Estimate", history: "History", about: "About the project" }[page];
  document.title = page === "home" ? "FirstLook · Admission chance at triage" : page === "result" && last ? `${pct(last.r.probability)}% chance of admission · FirstLook` : `${name} · FirstLook`;
  preview.active(page === "home");
  if (page === "about") evidence();
  if (page === "history") renderHistory();
  if (page === "result") { $("res-none").hidden = !!last; $("res-body").hidden = !last; }
}
/* in-page links to About sections (#ab-…, or data-to) scroll there without breaking the page route */
document.addEventListener("click", (e) => {
  const a = e.target.closest("a"); if (!a) return;
  if (a.classList.contains("skip")) { e.preventDefault(); $("main").focus(); return; } // keep the #/route in the URL
  const to = a.dataset.to ?? (a.getAttribute("href")?.startsWith("#ab-") ? a.getAttribute("href").slice(1) : null);
  if (!to) return;
  e.preventDefault();
  if (location.hash !== "#/about") { history.pushState(null, "", "#/about"); route(); }
  $(to)?.scrollIntoView({ behavior: reduced ? "auto" : "smooth" });
});
addEventListener("hashchange", () => {
  if (!location.hash.startsWith("#/") && location.hash !== "") return; // #main, #estimate: in-page anchors
  route(); scrollTo(0, 0); $("main").focus({ preventScroll: true });
});

function pill() {
  $("led").classList.toggle("on", true);
  $("nettext").textContent = SERVER ? "Works offline, patient records on" : navigator.onLine ? "Works offline" : "Offline, working normally";
}
addEventListener("online", pill); addEventListener("offline", pill);

async function detectServer() {
  try {
    const ctl = new AbortController(); const t = setTimeout(() => ctl.abort(), 2500);
    const r = await fetch("api/health", { signal: ctl.signal, cache: "no-store" }); clearTimeout(t);
    const j = r.ok ? await r.json() : null;
    SERVER = j && j.ok && j.models?.t3lite ? j : null;
  } catch { SERVER = null; }
  pill(); histAvailability();
  $("srvstate").textContent = SERVER ? "on: patient records can be added" : "off: estimates use triage data and triage level";
}

/* ---------------------------------------------------------------- home */
const dots = (p) => { let s = '<svg viewBox="0 0 200 50">';
  for (let i = 0; i < 100; i++) s += `<circle class="p${i < p ? " on" : ""}" style="--i:${i}" cx="${5 + Math.floor(i / 5) * 10}" cy="${5 + (i % 5) * 10}" r="3.6"/>`;
  return s + "</svg>"; };
const preview = (() => {
  let keys = [], i = 0, timer = 0, user = false, cur = null;
  const show = (k) => {
    cur = k; const f = META.examples[k].form, r = M.predict(f, META.models.t1, META), p = pct(r.probability);
    $("pv-who").textContent = `${f.gender}, ${f.age}`;
    $("pv-sub").textContent = `${ARR[f.arrivalmode] ?? f.arrivalmode}. ${f.complaints.map((c) => META.complaint_labels[c]).join(", ")}.`;
    $("pv-vit").innerHTML = [["Heart rate", f.triage_vital_hr, " bpm"], ["Breathing", f.triage_vital_rr, "/min"], ["Blood pressure", f.triage_vital_sbp && `${f.triage_vital_sbp}/${f.triage_vital_dbp}`, ""], ["SpO₂", f.triage_vital_o2, "%"], ["Temperature", f.triage_vital_temp, " °F"]]
      .map(([l, v, u]) => `<div><dt>${l}</dt><dd class="num">${v === null || v === undefined ? "Not measured" : v + u}</dd></div>`).join("");
    $("pv-pct").textContent = p;
    $("pv-say").innerHTML = `About <b>${p} in 100</b> similar patients were admitted. <b>${r.band} risk.</b>`;
    $("pv-array").innerHTML = dots(p);
    $("pv-array").closest(".demo").dataset.band = r.band;
    const [, t] = CONF[r.conformal]; $("pv-verdict").textContent = t; $("pv-verdict").classList.toggle("unc", r.conformal === "uncertain");
    $("pv-f").innerHTML = r.contributions.slice(0, 3).map((d) => { const up = d.value > 0;
      return `<li class="${up ? "up" : "dn"}"><span class="nm"><span class="dir"><svg class="ic" aria-hidden="true"><use href="#i-${up ? "up" : "down"}"/></svg></span>${esc(d.label)}</span><span class="val">${strength(d.value)}</span></li>`; }).join("");
    for (const b of $("pv-tabs").children) b.setAttribute("aria-pressed", String(b.dataset.k === k));
  };
  return {
    init() {
      keys = ["high", "uncertain", "low"].filter((k) => META.examples[k]);
      $("pv-tabs").innerHTML = keys.map((k) => `<button type="button" data-k="${k}" aria-pressed="false">${esc(META.examples[k].label.replace(" risk", "").replace("Uncertain", "Unsure"))}</button>`).join("");
      $("pv-tabs").addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; user = true; clearInterval(timer); i = keys.indexOf(b.dataset.k); show(b.dataset.k); });
      $("pv-open").addEventListener("click", () => { if (cur) load(cur); });
      show(keys[0]);
    },
    active(on) { clearInterval(timer); if (on && !user && !reduced && keys.length) timer = setInterval(() => { i = (i + 1) % keys.length; show(keys[i]); }, 4500); },
  };
})();
function home() {
  preview.init();
  const t1 = META.models.t1, f = META.examples.high.form, n = 200, t0 = performance.now();
  for (let k = 0; k < n; k++) M.predict(f, t1, META);
  const ms = (performance.now() - t0) / n;
  $("h-lat").textContent = ms < 1 ? "under a millisecond" : `${ms.toFixed(1)} milliseconds`;
  const lv = [["t1", t1.metrics, "Age, sex, arrival, main complaint, six vital signs"], ["t2", META.models.t2.metrics, "All of that, plus the triage level (ESI)"]];
  if (META.history) lv.push(["t3lite", META.history.metrics, "All of that, plus past visits, medicines, conditions and labs"]);
  $("h-levels-rows").innerHTML = lv.map(([k, m, d]) => `<tr><th scope="row">${LEVEL[k]}</th><td>${d}</td><td><b class="num">${pct(m.auc)}</b> in 100</td><td><b class="num">${pct(1 - m["conformal_alpha_0.10"].abstention)}</b> in 100 patients</td></tr>`).join("");
}

/* ---------------------------------------------------------------- history: saved estimates, this device only */
const HKEY = "firstlook.history.v1", HMAX = 200;
const hAll = () => { try { const v = JSON.parse(localStorage.getItem(HKEY) ?? "[]"); return Array.isArray(v) ? v : []; } catch { return []; } };
const hPut = (list) => { try { localStorage.setItem(HKEY, JSON.stringify(list.slice(0, HMAX))); return true; } catch { return false; } };
const formSig = () => JSON.stringify(snapshot());
const sig = () => JSON.stringify({ f: formSig(), note: $("hnote").value.trim() });
function hBadge() { const n = hAll().length; $("hcount").textContent = n; $("hcount").hidden = !n; }
function who(f) { return `${f.gender ?? "?"}, ${f.age ?? "?"}. ${f.complaints.map((c) => META.complaint_labels[c]).join(", ")}`; }
function saveEstimate() {
  if (!last) return false;
  const s = sig(), list = hAll(), r = last.r;
  if (savedSig === s && list.length) { toast("Already in history."); return true; }
  const prev = savedId && list.find((e) => e.id === savedId);
  if (prev && JSON.parse(savedSig).f === formSig()) { // same patient, only the note changed: update that entry
    prev.note = $("hnote").value.trim();
    if (!hPut(list)) { toast("Couldn’t save: this browser’s storage is full or switched off."); return false; }
    savedSig = s; saveState(); toast("Note saved."); return true;
  }
  const entry = { id: `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`, at: Date.now(), p: r.probability, band: r.band, conformal: r.conformal,
    slot: last.slot, note: $("hnote").value.trim(), who: who(last.f), age: last.f.age, sex: last.f.gender,
    complaints: last.f.complaints.map((c) => META.complaint_labels[c]), top: r.contributions.slice(0, 3).map((d) => ({ label: d.label, up: d.value > 0 })), form: snapshot() };
  if (!hPut([entry, ...list])) { toast("Couldn’t save: this browser’s storage is full or switched off."); return false; }
  savedSig = s; savedId = entry.id; hBadge(); saveState(); toast("Saved to history.");
  return true;
}
function saveState() {
  const t = new Date(hAll()[0]?.at ?? Date.now());
  const isSaved = !!last && savedSig === sig();
  $("hsaved").hidden = !isSaved; $("hsaved").textContent = `Saved to history at ${t.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}.`;
  const noteOnly = !!last && !isSaved && !!savedSig && JSON.parse(savedSig).f === formSig() && hAll().some((e) => e.id === savedId);
  $("hsave").textContent = isSaved ? "Saved" : noteOnly ? "Save note" : "Save to history"; $("hsave").disabled = isSaved;
}
const dayLabel = (t) => { const d = new Date(t), today = new Date(); const k = (x) => x.toDateString();
  const y = new Date(today); y.setDate(today.getDate() - 1);
  return k(d) === k(today) ? "Today" : k(d) === k(y) ? "Yesterday" : d.toLocaleDateString([], { weekday: "long", day: "numeric", month: "long", year: "numeric" }); };
function renderHistory() {
  const all = hAll(), q = $("hist-q").value.trim().toLowerCase(), band = radio("hband") ?? "";
  const list = all.filter((e) => (!band || (band === "unsure" ? e.conformal === "uncertain" : e.band === band))
    && (!q || `${e.who} ${e.note} ${e.age}`.toLowerCase().includes(q)));
  $("hempty").hidden = all.length > 0; $("hist-tools").hidden = !all.length; $("hist-actions").hidden = !all.length;
  const n = (b) => all.filter((e) => e.band === b).length;
  $("hist-sum").textContent = all.length ? `${list.length === all.length ? all.length : `${list.length} of ${all.length}`} saved. ${n("High")} high, ${n("Medium")} medium, ${n("Low")} low.` : "";
  let day = "", html = "";
  for (const e of list) {
    const d = dayLabel(e.at); if (d !== day) { day = d; html += `<li class="hday">${esc(d)}</li>`; }
    const [, vt] = CONF[e.conformal] ?? CONF.uncertain;
    html += `<li class="hitem" data-band="${esc(e.band)}">
      <div class="h-pct"><b class="num">${pct(e.p)}%</b><span>${esc(e.band)}</span></div>
      <div class="h-main"><p class="h-who">${esc(e.who)}</p>
        <p class="h-meta"><time datetime="${new Date(e.at).toISOString()}">${new Date(e.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time><span class="${e.conformal === "uncertain" ? "unc" : ""}">${esc(vt)}</span><span>${esc(LEVEL[e.slot] ?? "")}</span></p>
        ${e.note ? `<p class="h-note">${esc(e.note)}</p>` : ""}
        ${e.top?.length ? `<p class="h-top">Moved most by ${e.top.map((t) => `${t.up ? "↑" : "↓"} ${esc(t.label)}`).join(", ")}</p>` : ""}</div>
      <div class="h-act"><button type="button" class="btn" data-open="${e.id}">Open</button><button type="button" class="btn quiet" data-del="${e.id}" aria-label="Delete the estimate for ${esc(e.who)}">Delete</button></div></li>`;
  }
  $("hlist").innerHTML = html || (all.length ? '<li class="hint none">No saved estimate matches. Try another search or filter.</li>' : "");
}
function historyUI() {
  hBadge();
  $("hsave").onclick = saveEstimate;
  $("hnote").addEventListener("input", () => { saveState(); });
  $("hnote").addEventListener("keydown", (e) => { if (e.key === "Enter" && !$("hsave").disabled) { e.preventDefault(); saveEstimate(); } });
  $("hist-q").addEventListener("input", renderHistory);
  $("hist-band").addEventListener("change", renderHistory);
  $("hlist").addEventListener("click", (e) => {
    const o = e.target.closest("[data-open]")?.dataset.open, d = e.target.closest("[data-del]")?.dataset.del;
    if (o) { const en = hAll().find((x) => x.id === o); if (!en) return;
      withUndo("Opened a saved estimate.", () => { apply(en.form); touched.clear(); submitted = false; $("hnote").value = en.note ?? ""; });
      savedSig = sig(); savedId = en.id; saveState(); location.hash = "#/result"; }
    if (d) { const before = hAll(); hPut(before.filter((x) => x.id !== d)); hBadge(); renderHistory();
      toast("Deleted from history.", () => { hPut(before); hBadge(); renderHistory(); }); }
  });
  $("hist-clear").onclick = () => { const before = hAll(); hPut([]); hBadge(); renderHistory(); toast("History cleared.", () => { hPut(before); hBadge(); renderHistory(); }); };
  $("hist-csv").onclick = () => {
    const cell = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
    const rows = [["saved_at", "note", "age", "sex", "complaints", "chance_percent", "risk", "confidence", "based_on"],
      ...hAll().map((e) => [new Date(e.at).toISOString(), e.note, e.age, e.sex, e.complaints?.join("; "), pct(e.p), e.band, CONF[e.conformal]?.[1], LEVEL[e.slot]])];
    const url = URL.createObjectURL(new Blob([rows.map((r) => r.map(cell).join(",")).join("\r\n")], { type: "text/csv" }));
    const a = document.createElement("a"); a.href = url; a.download = `firstlook-history-${new Date().toISOString().slice(0, 10)}.csv`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000); toast("Downloaded history as CSV.");
  };
  addEventListener("storage", (e) => { if (e.key === HKEY) { hBadge(); if (!$("history").hidden) renderHistory(); } }); // other tabs
}

/* ---------------------------------------------------------------- build the form */
function build() {
  $("arrival").innerHTML = Object.keys(ARR).filter((a) => META.arrival_levels.includes(a))
    .map((a) => `<label><input type="radio" name="arrival" value="${esc(a)}"${a === "Walk-in" ? " checked" : ""}>${esc(ARR[a])}</label>`).join("");
  for (const h of META.hour_bins) $("hour").add(new Option(`${h.replace("-", ":00 to ")}:59`, h));
  $("hour").value = hourBin(new Date().getHours());
  $("vitals").innerHTML = VITAL.map(([k, l, u, ok, , , mode]) => `<div class="vital" id="row-${k}">
    <label class="l" for="${k}">${l}<small>usual ${ok[0]}–${ok[1]}${u === "%" ? "%" : NB + u}</small></label>
    <div class="unit-in"><input id="${k}" name="${k}" type="number" step="any" inputmode="${mode}" aria-describedby="err-${k}"><span aria-hidden="true">${u}</span></div>
    <div class="strip" id="strip-${k}" aria-hidden="true"></div>
    <label class="nmt"><input type="checkbox" id="${k}_nm">Not measured</label>
    <p class="msg" id="err-${k}" hidden></p></div>`).join("");
  for (const [k] of VITAL) { $(k + "_nm").addEventListener("change", (e) => { setMissing(k, e.target.checked); touched.add(k); changed(true); }); strip(k); }
  const ex = $("examples");
  for (const [key, e] of Object.entries(META.examples)) {
    const b = document.createElement("button"); b.type = "button"; b.className = "btn"; b.textContent = e.label; b.onclick = () => load(key); ex.appendChild(b);
  }
  $("cccommon").innerHTML = META.complaint_order.filter((c) => c !== "cc_other").slice(0, 6)
    .map((c) => `<button type="button" class="common" data-c="${c}">${esc(META.complaint_labels[c])}</button>`).join(", ");
  $("cccommon").addEventListener("click", (e) => { const c = e.target.closest("[data-c]")?.dataset.c; if (c) addComplaint(c); });
  combobox(); historyForm();
  const form = $("pform");
  form.addEventListener("input", (e) => { if (e.target.type === "number") strip(e.target.id); changed(false); });
  form.addEventListener("change", (e) => { if (e.target.type === "radio" || e.target.tagName === "SELECT") { touched.add(e.target.name); changed(true); } });
  form.addEventListener("focusout", (e) => { if (e.target.type === "number") { touched.add(e.target.id); changed(true); } });
  form.addEventListener("submit", (e) => { e.preventDefault(); check(); });
  $("usehist").addEventListener("change", () => { $("histbox").hidden = !$("usehist").checked; changed(true); });
  $("random").onclick = randomFill;
  $("clear").onclick = () => withUndo("Form cleared.", clearForm);
  $("res-new").onclick = () => { withUndo("Ready for the next patient.", clearForm); location.hash = "#/assess"; };
  $("undo").onclick = undo;
  $("morefactors").onclick = () => { showAll = !showAll; renderFactors(); };
  restore(); update(true);
}

function strip(k) {
  const v = VITAL.find((x) => x[0] === k); if (!v) return;
  const [, , , ok, [a, b], soft] = v, pos = (x) => ((Math.min(Math.max(x, a), b) - a) / (b - a)) * 100;
  const zones = [[a, soft[0], "z2"], [soft[0], ok[0], "z1"], [ok[0], ok[1], "z0"], [ok[1], soft[1], "z1"], [soft[1], b, "z2"]].filter(([x, y]) => y > x);
  const val = $(k).value === "" || $(k + "_nm").checked ? null : Number($(k).value);
  $("strip-" + k).innerHTML = `<div class="zones">${zones.map(([x, y, z]) => `<i class="${z}" style="left:${pos(x)}%;width:${pos(y) - pos(x)}%"></i>`).join("")}</div>` +
    `<span class="tk" style="left:${pos(ok[0])}%">${ok[0]}</span>${ok[1] < b ? `<span class="tk" style="left:${pos(ok[1])}%">${ok[1]}</span>` : ""}` +
    (val === null || Number.isNaN(val) ? "" : `<b class="dot" style="left:${pos(val)}%"></b>`);
}
function setMissing(k, on) {
  $(k + "_nm").checked = on; $(k).disabled = on; if (on) $(k).value = "";
  $(k).placeholder = on ? "Not measured" : ""; $("row-" + k).classList.toggle("nm", on); strip(k);
}

function historyForm() {
  const H = META.history;
  if (!H) { $("usehist").closest(".hist").hidden = true; return; }
  const field = (f) => {
    const id = hid(f.key);
    if (f.kind === "binary") return `<label class="tick" for="${id}"><input type="checkbox" id="${id}">${esc(f.label)}</label>`;
    return `<div><label class="l" for="${id}">${esc(f.label)}</label><input id="${id}" type="number" ${f.kind === "lab" ? 'step="any" placeholder="Not available"' : 'min="0" step="1" inputmode="numeric" placeholder="0"'} aria-describedby="err-${id}"><p class="msg" id="err-${id}" hidden></p></div>`;
  };
  const group = (g) => H.fields.filter((f) => f.group === g).map(field).join("");
  let s = '<p class="hint" style="margin-top:12px">Counts cover the past year; leave blank if none on record.</p>';
  if (H.prev_dispo.length) s += `<div class="f" style="margin-top:12px"><label class="l" for="h_prevdispo">Outcome of the previous ED visit</label><select id="h_prevdispo">${H.prev_dispo.map((v) => `<option value="${esc(v)}">${esc(v === "No previous dispo" ? "No previous ED visit" : v)}</option>`).join("")}</select></div>`;
  s += `<div class="hist-grid">${group("key")}</div><details class="more"><summary>Conditions, medicines and past labs</summary>
    <p class="sub-h">Conditions on record</p><div class="hist-grid">${group("binary")}</div>
    <p class="sub-h">Outpatient medicines, by class</p><div class="hist-grid">${group("count")}</div>
    <p class="sub-h">Past-year labs</p><div class="hist-grid">${group("lab")}</div></details>`;
  $("histbox").innerHTML = s;
}
function histAvailability() {
  const ok = !!SERVER && !!META?.history;
  $("usehist").disabled = !ok; $("histoff").hidden = ok;
  if (!ok) { $("usehist").checked = false; $("histbox").hidden = true; }
  else { try { if (JSON.parse(sessionStorage.getItem(STORE) ?? "null")?.usehist) $("usehist").checked = true; } catch { /* ignore */ } $("histbox").hidden = !$("usehist").checked; }
  if (META?.history) $("histsub").textContent = "Past visits, medicines, conditions and lab results give the sharpest estimate.";
}

/* complaint combobox — WAI-ARIA APG "list autocomplete": ↑ ↓ Enter Esc, Backspace removes the last chip */
function combobox() {
  const s = $("ccsearch"), o = $("ccopts");
  let hits = [], act = 0;
  const mark = (t, q) => { const i = t.toLowerCase().indexOf(q); return i < 0 ? esc(t) : esc(t.slice(0, i)) + "<mark>" + esc(t.slice(i, i + q.length)) + "</mark>" + esc(t.slice(i + q.length)); };
  const render = () => {
    const q = s.value.trim().toLowerCase(), open = q !== "";
    o.innerHTML = hits.length ? hits.map((c, i) => `<li role="option" id="opt-${i}" data-c="${c}" aria-selected="${i === act}">${mark(META.complaint_labels[c], q)}</li>`).join("")
      : `<li class="none" role="option" aria-disabled="true">${selected.length >= 3 ? "Three complaints chosen; remove one to add another." : "No match. Try another word, or choose “Other”."}</li>`;
    o.hidden = !open; s.setAttribute("aria-expanded", String(open));
    open && hits.length ? s.setAttribute("aria-activedescendant", `opt-${act}`) : s.removeAttribute("aria-activedescendant");
    o.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: "nearest" });
  };
  s.addEventListener("input", () => {
    const q = s.value.trim().toLowerCase(); act = 0;
    hits = selected.length >= 3 ? [] : META.complaint_order.filter((c) => !selected.includes(c) && META.complaint_labels[c].toLowerCase().includes(q))
      .sort((a, b) => META.complaint_labels[a].toLowerCase().indexOf(q) - META.complaint_labels[b].toLowerCase().indexOf(q)).slice(0, 30);
    render();
  });
  s.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { act = Math.min(act + 1, hits.length - 1); render(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { act = Math.max(act - 1, 0); render(); e.preventDefault(); }
    else if (e.key === "Enter") { if (hits.length && !o.hidden) { addComplaint(hits[act]); e.preventDefault(); } }
    else if (e.key === "Escape") { s.value = ""; hits = []; render(); }
    else if (e.key === "Backspace" && s.value === "" && selected.length) removeComplaint(selected[selected.length - 1]);
  });
  s.addEventListener("blur", () => setTimeout(() => { o.hidden = true; s.setAttribute("aria-expanded", "false"); }, 150));
  o.addEventListener("mousedown", (e) => { const c = e.target.closest("[data-c]")?.dataset.c; if (c) { addComplaint(c); e.preventDefault(); } });
  $("ccchips").addEventListener("click", (e) => { const c = e.target.closest("[data-c]")?.dataset.c; if (c) { removeComplaint(c); $("ccsearch").focus(); } });
  $("ccbox").addEventListener("click", (e) => { if (e.target === $("ccbox")) s.focus(); });
  combobox.reset = () => { s.value = ""; hits = []; render(); };
}
function addComplaint(c) {
  if (c && selected.length < 3 && !selected.includes(c)) { selected = [...selected, c]; touched.add("complaints"); announce(`${META.complaint_labels[c]} added.`); }
  chips(); combobox.reset(); changed(true);
}
function removeComplaint(c) { selected = selected.filter((x) => x !== c); announce(`${META.complaint_labels[c]} removed.`); chips(); changed(true); }
function chips() {
  $("ccchips").innerHTML = selected.map((c) => `<li><button type="button" data-c="${c}" aria-label="Remove ${esc(META.complaint_labels[c])}">${esc(META.complaint_labels[c])}<svg class="ic" aria-hidden="true"><use href="#i-x"/></svg></button></li>`).join("");
}

/* ---------------------------------------------------------------- state */
const histIds = () => (META.history ? [...META.history.fields.map((f) => hid(f.key)), ...(META.history.prev_dispo.length ? ["h_prevdispo"] : [])] : []);
function snapshot() {
  const v = { selected: [...selected], note: $("exnote").innerHTML, usehist: $("usehist").checked, age: $("age").value, hour: $("hour").value,
    gender: radio("gender"), arrival: radio("arrival"), esi: radio("esi"), o2dev: radio("o2dev") };
  for (const [k] of VITAL) { v[k] = $(k).value; v[k + "_nm"] = $(k + "_nm").checked; }
  for (const id of histIds()) v[id] = $(id).type === "checkbox" ? $(id).checked : $(id).value;
  return v;
}
function apply(v) {
  $("age").value = v.age ?? ""; if (v.hour) $("hour").value = v.hour;
  setRadio("gender", v.gender); setRadio("arrival", v.arrival ?? "Walk-in"); setRadio("esi", v.esi ?? ""); setRadio("o2dev", v.o2dev ?? "0");
  for (const [k] of VITAL) { setMissing(k, !!v[k + "_nm"]); if (!v[k + "_nm"]) $(k).value = v[k] ?? ""; strip(k); }
  for (const id of histIds()) if (v[id] !== undefined) { if ($(id).type === "checkbox") $(id).checked = !!v[id]; else $(id).value = v[id]; }
  $("usehist").checked = !!v.usehist && !$("usehist").disabled; $("histbox").hidden = !$("usehist").checked;
  selected = [...(v.selected ?? [])]; chips(); note(v.note ?? "");
}
function save() { try { sessionStorage.setItem(STORE, JSON.stringify(snapshot())); } catch { /* storage unavailable */ } }
function restore() { try { const v = JSON.parse(sessionStorage.getItem(STORE) ?? "null"); if (v) apply(v); } catch { /* ignore corrupt state */ } }
const isBlank = () => !$("age").value && !radio("gender") && !selected.length && VITAL.every(([k]) => !$(k).value && !$(k + "_nm").checked);
function note(html) { $("exnote").innerHTML = html; $("exnote").hidden = !html; }

function clearForm() {
  apply({ hour: hourBin(new Date().getHours()) }); for (const id of histIds()) if (id === "h_prevdispo") $(id).selectedIndex = 0;
  touched.clear(); submitted = false; last = null; $("hnote").value = ""; savedSig = null; savedId = null;
}
/** One toast for the whole app. With onUndo it offers Undo for 8 seconds. */
function toast(msg, onUndo = null) {
  undoFn = onUndo; $("toastmsg").textContent = msg; $("undo").hidden = !onUndo; $("toast").hidden = false;
  announce(onUndo ? `${msg} Undo is available.` : msg);
  clearTimeout(undoTimer); undoTimer = setTimeout(() => { $("toast").hidden = true; undoFn = null; }, onUndo ? 8000 : 3500);
}
function undo() { if (!undoFn) return; const fn = undoFn; undoFn = null; $("toast").hidden = true; clearTimeout(undoTimer); fn(); announce("Restored."); }
function withUndo(msg, fn) {
  const had = !isBlank(), snap = { form: snapshot(), touched: [...touched], submitted };
  fn(); save(); update(true); route();
  if (had) toast(msg, () => { apply(snap.form); touched.clear(); snap.touched.forEach((t) => touched.add(t)); submitted = snap.submitted; save(); update(true); });
}
function fill(f) {
  clearForm();
  $("age").value = f.age; setRadio("gender", f.gender); setRadio("arrival", f.arrivalmode); $("hour").value = f.arrivalhour_bin;
  for (const [k] of VITAL) { setMissing(k, f[k] === null); if (f[k] !== null) $(k).value = f[k]; strip(k); }
  setRadio("o2dev", f.triage_vital_o2_device === null ? "" : f.triage_vital_o2_device);
  selected = [...f.complaints]; chips();
}
function load(key) {
  const e = META.examples[key];
  withUndo(`Loaded “${e.label}”.`, () => { fill(e.form); note(`<b>Synthetic patient: ${esc(e.label)}.</b> Invented for the demo, not a real person.`); });
}
function randomFill() {
  const seed = crypto.getRandomValues(new Uint32Array(1))[0] % 100000, esiOn = radio("esi") !== "";
  const { form, meta } = M.randomPatient(seed, $("acuity").value || null, $("rmiss").checked, META);
  withUndo("Loaded a random patient.", () => {
    fill(form); if (esiOn) setRadio("esi", meta.esi);
    note(`<b>Random synthetic patient, ${meta.acuity} acuity</b> (seed ${seed}). Invented for the demo, not a real person.`);
  });
}

function readForm() {
  const num = (id) => { const v = $(id).value.trim(); return v === "" ? null : Number(v); };
  const o2 = radio("o2dev"), esi = radio("esi");
  const f = { age: num("age"), gender: radio("gender"), arrivalmode: radio("arrival"), arrivalhour_bin: $("hour").value,
    triage_vital_o2_device: o2 === "" || o2 === null ? null : Number(o2), complaints: [...selected], esi: esi ? Number(esi) : null };
  for (const [k] of VITAL) f[k] = $(k + "_nm").checked ? null : num(k);
  if ($("usehist").checked && META.history) {
    for (const h of META.history.fields) { const el = $(hid(h.key)); f[h.key] = h.kind === "binary" ? (el.checked ? 1 : 0) : h.kind === "lab" ? num(el.id) : (num(el.id) ?? 0); }
    if ($("h_prevdispo")) f.previousdispo = $("h_prevdispo").value;
  }
  return f;
}

/** Every finding about the form: {field, level: error|warning|missing, msg}. */
function findings(f) {
  const out = [];
  for (const d of M.validateFields(f, META)) {
    const blank = (d.field === "age" && f.age === null) || (d.field === "gender" && !f.gender) || (d.field === "complaints" && !f.complaints.length);
    out.push({ ...d, level: blank ? "missing" : d.level });
  }
  for (const [k] of VITAL) if (f[k] === null && !$(k + "_nm").checked) out.push({ field: k, level: "missing", msg: "Enter a value, or tick “Not measured”." });
  if ($("usehist").checked && META.history) for (const h of META.history.fields) {
    const v = f[h.key];
    if (h.kind === "count" && v !== null && (v < 0 || !Number.isInteger(v) || v > 500)) out.push({ field: hid(h.key), level: "error", msg: "Enter a whole number from 0 to 500." });
  }
  const order = ["age", "gender", "complaints", ...VITAL.map((v) => v[0])];
  return out.sort((a, b) => (order.indexOf(a.field) + 1 || 99) - (order.indexOf(b.field) + 1 || 99));
}
const INPUT = { complaints: "ccsearch" };
const focusField = (field) => {
  const el = field === "gender" ? document.querySelector('input[name="gender"]') : $(INPUT[field] ?? field);
  el?.focus({ preventScroll: true }); el?.scrollIntoView({ block: "center", behavior: reduced ? "auto" : "smooth" });
};
const fieldName = (field) => ({ age: "Age", gender: "Sex", complaints: "Chief complaint" })[field] ?? VITAL.find((v) => v[0] === field)?.[1]
  ?? META.history?.fields.find((h) => hid(h.key) === field)?.label ?? field;

/** Inline messages: errors and warnings once a field is touched (and cleared the moment it is fixed); blanks only after "Check". */
function showMessages(items) {
  for (const el of document.querySelectorAll(".msg")) { el.hidden = true; el.textContent = ""; el.classList.remove("warn"); }
  for (const el of document.querySelectorAll("[aria-invalid]")) el.removeAttribute("aria-invalid");
  $("ccbox").classList.remove("bad");
  const seen = new Set();
  for (const d of items) {
    if (seen.has(d.field)) continue;
    const show = submitted || (d.level !== "missing" && touched.has(d.field)) || d.field.startsWith("h_");
    const p = $(`err-${d.field}`); if (!show || !p) continue;
    seen.add(d.field); p.textContent = d.msg; p.classList.toggle("warn", d.level === "warning"); p.hidden = false;
    if (d.level !== "warning") {
      if (d.field === "gender") $("gender").setAttribute("aria-invalid", "true");
      else if (d.field === "complaints") { $("ccsearch").setAttribute("aria-invalid", "true"); $("ccbox").classList.add("bad"); }
      else $(d.field)?.setAttribute("aria-invalid", "true");
    }
  }
  const n = items.filter((d) => d.level !== "warning").length;
  $("errsum").hidden = !(submitted && n);
  $("errsum").textContent = `${n} field${n > 1 ? "s need" : " needs"} attention. See the messages next to them.`;
}

function changed(now) { save(); update(now); }
function update(now) {
  const f = readForm(), items = findings(f);
  showMessages(items);
  const vit = VITAL.filter(([k]) => $(k + "_nm").checked || $(k).value !== "").length;
  $("vcount").textContent = `${vit} of ${VITAL.length}`;
  clearTimeout(panelTimer);
  if (now) panel(f, items); else panelTimer = setTimeout(() => panel(f, items), 250);
}

/* ---------------------------------------------------------------- checklist (assess) and estimate (result) */
function setState(s) {
  const ready = s === "result" || s === "busy";
  $("readycard").dataset.state = ready ? "ready" : s; $("tag").dataset.state = s;
  $("st-empty").hidden = s !== "empty"; $("st-fix").hidden = s !== "fix"; $("st-ready").hidden = !ready;
  $("readyhint").hidden = !!(radio("esi") && $("usehist").checked);
}
async function panel(f, items) {
  const errs = items.filter((d) => d.level === "error"), miss = items.filter((d) => d.level === "missing");
  const id = ++reqId;
  if (errs.length) {
    last = null; setState("fix"); delete $("tag").dataset.band;
    $("fixlist").innerHTML = errs.map((d) => `<li><button type="button" class="err" data-f="${d.field}">${d.msg.startsWith(fieldName(d.field)) ? "" : esc(fieldName(d.field)) + ": "}${esc(d.msg)}</button></li>`).join("");
  } else if (miss.length) {
    last = null; setState("empty"); delete $("tag").dataset.band;
    const need = new Set(miss.map((d) => d.field));
    const all = ["age", "gender", "complaints", ...VITAL.map((v) => v[0])];
    $("todo").innerHTML = all.map((k) => need.has(k) ? `<li><button type="button" data-f="${k}">${esc(fieldName(k))}</button></li>`
      : `<li class="done">${esc(fieldName(k))}</li>`).join("");
  } else {
    if ($("usehist").checked && SERVER) {
      if (last) setState("busy");
      try { const res = await predictServer(f); if (id !== reqId) return; last = { ...res, f }; }
      catch (e) {
        if (id !== reqId) return;
        last = null; setState("fix");
        $("fixlist").innerHTML = `<li class="err">The patient record couldn’t be used just now (${esc(e.message)}). Switch “Add the patient’s record” off to get an estimate without it.</li>`;
        route(); return;
      }
    } else {
      const slot = f.esi ? "t2" : "t1", m = META.models[slot];
      last = { r: M.predict(f, m, META), m, f, slot, engine: "This device", kind: "linear" };
    }
    render();
  }
  route();
}
for (const id of ["todo", "fixlist"]) document.getElementById(id).addEventListener("click", (e) => { const b = e.target.closest("[data-f]"); if (b) focusField(b.dataset.f); });

async function predictServer(f) {
  const t0 = performance.now();
  const res = await fetch("api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ form: f, slot: META.history.slot }) });
  const j = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(j.fields?.[0]?.msg ?? j.error ?? `HTTP ${res.status}`);
  return { r: { probability: j.probability, band: j.band, conformal: j.conformal, contributions: j.contributions.filter((d) => Math.abs(d.value) > 0.05), ms: j.latency_ms, roundtrip: performance.now() - t0 },
    m: { name: j.name, features: { length: j.features }, thresholds: j.thresholds }, slot: j.slot, engine: "This computer", kind: j.explain_kind };
}

function render() {
  const { r, m, slot, engine, kind } = last, p = pct(r.probability);
  setState("result"); $("tag").dataset.band = r.band;
  renderInputs();
  $("pct").textContent = p;
  $("pctsay").innerHTML = `About <b>${p} in 100</b> similar patients were admitted. <b>${r.band} risk.</b>`;
  // icon array: 20 × 5, filled column by column so the admitted share reads like a bar
  let s = '<svg viewBox="0 0 200 50">';
  for (let i = 0; i < 100; i++) { const c = Math.floor(i / 5), row = i % 5; s += `<circle class="p${i < p ? " on" : ""}" style="--i:${i}" cx="${5 + c * 10}" cy="${5 + row * 10}" r="3.6"/>`; }
  $("array").innerHTML = s + "</svg>";
  const tl = m.thresholds.t_low, th = m.thresholds.t_high, x = Math.min(Math.max(r.probability * 100, 1), 99);
  $("scale").innerHTML = `<div class="track"><div class="segs"><i style="width:${tl * 100}%;background:var(--low-fill)"></i><i style="width:${(th - tl) * 100}%;background:var(--med-fill)"></i><i style="flex:1;background:var(--high)"></i></div><span class="mk" style="left:${x}%"></span></div>
    <div class="lbl"><span${r.band === "Low" ? ' class="on"' : ""}>Low under ${pct(tl)}%</span><span>Medium</span><span>High from ${pct(th)}%</span></div>`;
  const [ic, t, d] = CONF[r.conformal];
  $("verdict").classList.toggle("unc", r.conformal === "uncertain");
  $("vicon").setAttribute("href", ic); $("conftitle").textContent = t; $("conftext").textContent = d;
  $("mtier").textContent = LEVEL[slot] ?? m.name; $("mengine").textContent = engine;
  $("mms").textContent = r.ms < 1 ? "Under 1 ms" : `${Math.round(r.ms)}${NB}ms`;
  $("exnote2").textContent = "Longer bars mattered more. These are patterns in past patients, not causes.";
  renderFactors(); saveState();
  clearTimeout(sayTimer);
  if (!$("result").hidden) sayTimer = setTimeout(() => announce(`Chance of admission ${p} percent, ${r.band.toLowerCase()}. ${t}.`), 600);
}
/* result page: who the patient is and exactly what was entered, so the estimate can be checked against the chart */
function renderInputs() {
  const f = last.f, arr = document.querySelector('input[name="arrival"]:checked')?.parentElement.textContent.trim();
  const hour = $("hour").selectedOptions[0]?.textContent;
  $("res-who").textContent = `${who(f)}. ${arr ? `Arrived by ${arr.toLowerCase()}, ` : ""}${hour ?? ""}.`;
  const rows = [["Age", `${f.age} years`], ["Sex", f.gender], ["Arrived by", arr], ["Arrival time", hour],
    ["Chief complaint", f.complaints.map((c) => META.complaint_labels[c]).join(", ")],
    ...VITAL.map(([k, name, unit]) => [name, $(k + "_nm").checked ? "Not measured" : `${$(k).value}${unit ? NB + unit : ""}`]),
    ["Triage level (ESI)", f.esi ? String(f.esi) : "Not set"], ["Patient record", $("usehist").checked && SERVER ? "Added" : "Not added"]];
  $("res-input").innerHTML = rows.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v ?? "")}</dd></div>`).join("");
}
const strength = (v) => { const a = Math.abs(v); return a >= 0.5 ? "Strong" : a >= 0.2 ? "Moderate" : "Slight"; };
function renderFactors() {
  const all = last.r.contributions, list = showAll ? all.slice(0, 15) : all.slice(0, 5);
  const mx = Math.max(...list.map((d) => Math.abs(d.value)), 0.5);
  $("factors").innerHTML = list.map((d) => { const up = d.value > 0;
    return `<li class="${up ? "up" : "dn"}"><span class="nm"><span class="dir"><svg class="ic" aria-hidden="true"><use href="#i-${up ? "up" : "down"}"/></svg></span><span class="sr-only">${up ? "Raises the chance:" : "Lowers the chance:"}</span>${esc(d.label)}</span>
      <span class="bar" aria-hidden="true"><i style="width:${Math.abs(d.value) / mx * 50}%"></i></span><span class="val">${strength(d.value)}</span></li>`; }).join("")
    || '<li class="hint">No single factor stood out.</li>';
  $("morefactors").hidden = all.length <= 5;
  $("morefactors").textContent = showAll ? "Show the top 5" : `Show all ${Math.min(all.length, 15)} factors`;
  $("morefactors").setAttribute("aria-expanded", String(showAll));
}

/** Enter / "Check and save": reveal every message, move to the first problem, or to the estimate. */
async function check() {
  submitted = true;
  const f = readForm(), items = findings(f), bad = items.filter((d) => d.level !== "warning");
  showMessages(items); clearTimeout(panelTimer); await panel(f, items);
  if (bad.length) { focusField(bad[0].field); announce(`${bad.length} field${bad.length > 1 ? "s need" : " needs"} attention. ${bad[0].msg}`); return; }
  saveEstimate();
  location.hash = "#/result";
}

/* ---------------------------------------------------------------- evidence */
let evDone = false;
function evidence() {
  if (evDone) return; evDone = true;
  if (!DASH) { $("kpis").innerHTML = '<p style="padding:14px">Research results (data/dashboard.json) are not in this build.</p>'; return; }
  
  const colors = ["var(--s1)", "var(--s2)", "var(--s3)"], f3 = (v) => v.toFixed(3);
  $("kpis").innerHTML = `<table><caption class="sr-only">Shipped models on the test set</caption><thead><tr><th scope="col">Model</th><th scope="col">ROC-AUC (95% CI)</th><th scope="col">Calibration error</th><th scope="col">Coverage at 90%</th><th scope="col">Answered with confidence</th><th scope="col">Size</th></tr></thead><tbody>` +
    DASH.edge.map((e, i) => `<tr><td><span class="sw" style="background:${colors[i]}"></span><b>${esc(e.slot)}</b></td><td>${f3(e.auc[0])} <span class="ci">(${f3(e.auc[1])}–${f3(e.auc[2])})</span></td><td>${f3(e.ece[0])}</td><td>${(e.coverage * 100).toFixed(1)}%</td><td>${pct(1 - e.abstention)}%</td><td>${e.size_kb < 100 ? e.size_kb.toFixed(1) : Math.round(e.size_kb)}${NB}KB</td></tr>`).join("") + "</tbody></table>";
  const SERIES = [["t1", "Triage data only", "Triage", colors[0]], ["t2", "+ triage level", "+ level", colors[1]], ["t3", "+ patient record", "+ record", colors[2]]];
  const pc = (v) => `${Math.round(v * 100)}%`;
  C.line($("c-cal"), { aria: "FirstLook’s estimate against the share actually admitted, for the three levels of detail; all lines lie on the diagonal.", diagonal: true,
    series: SERIES.map(([k, name, short, color]) => ({ name, short, color, label: false, points: DASH.calibration[k].pred.map((p, i) => [+p.toFixed(3), DASH.calibration[k].obs[i]]) })),
    x: { min: 0, max: 1, ticks: [0, .2, .4, .6, .8, 1], fmt: pc, title: "FirstLook’s estimate" }, y: { min: 0, max: 1, ticks: [0, .2, .4, .6, .8, 1], fmt: pc, title: "Actually admitted" } });
  const dc = DASH.decision_curve, th = dc.threshold;
  C.line($("c-dca"), { aria: "Decision curves: net benefit of T1, T2, T3 against admitting everyone or no one, thresholds 10% to 70%.",
    series: [...SERIES.map(([k, name, short, color]) => ({ name, short, color, points: th.map((t, i) => [t, dc[k][i]]) })),
      { name: "Admit everyone", short: "All", color: "var(--axis)", dash: true, label: false, points: th.map((t, i) => [t, dc.treat_all[i]]).filter((q) => q[1] >= -0.05) },
      { name: "Admit no one", short: "None", color: "#6b7280", dash: true, label: false, points: th.map((t, i) => [t, dc.treat_none[i]]) }],
    x: { min: 0.1, max: 0.7, ticks: [.1, .2, .3, .4, .5, .6, .7], fmt: pc, title: "Decision threshold" },
    y: { min: -0.05, max: 0.3, ticks: [0, .1, .2, .3], fmt: (v, t) => t ? v.toFixed(3) : v.toFixed(2), title: "Net benefit" } });
  const la = DASH.local_adaptation, site = ["#15803d", "#c98a00", "#8a24c4"];
  C.line($("c-la"), { aria: "How often the confidence check was right at each new hospital as local visits were added; all reach the 9-in-10 target by about 1,000 visits.", band: [0.88, 0.92],
    series: Object.entries(la.sites).map(([s, d], i) => ({ name: `Hospital ${s}`, short: s, color: site[i], points: d.n.map((n, j) => [n, d.coverage_admit[j]]) })),
    x: { min: 50, max: 5000, log: true, ticks: [50, 100, 250, 500, 1000, 2000, 5000], fmt: (v) => v >= 1000 ? `${v / 1000}k` : String(v), title: "Local patients used to tune it" },
    y: { min: 0.85, max: 0.97, ticks: [.86, .88, .9, .92, .94, .96], fmt: (v, t) => t ? `${(v * 100).toFixed(1)}%` : pc(v), title: "Confidence check right" } });
  $("c-la-cap").textContent = "At each hospital it had never seen, how often the confidence check was right for admitted patients, as local visits were added. The shaded band is the 9-in-10 target; every hospital reaches it within about 1,000 visits, with no retraining.";
  const rows = DASH.benchmark.filter((b) => b.setting.includes("network none"));
  $("t-bench").innerHTML = `<table><caption class="sr-only">Edge benchmark</caption><thead><tr><th scope="col">Model</th><th scope="col">Size</th><th scope="col">p50</th><th scope="col">p95</th><th scope="col">Memory</th><th scope="col">Network</th></tr></thead><tbody>` +
    rows.map((b) => `<tr><td><b>${b.model === "t3lite" ? "T3-lite" : b.model.toUpperCase()}</b></td><td>${b.size_kb < 100 ? b.size_kb.toFixed(1) : Math.round(b.size_kb)}${NB}KB</td><td>${b.p50_ms.toFixed(1)}${NB}ms</td><td>${b.p95_ms.toFixed(1)}${NB}ms</td><td>${Math.round(b.peak_rss_mb)}${NB}MB</td><td>${b.net_bytes}${NB}bytes</td></tr>`).join("") +
    `</tbody></table><p class="hint">${esc(rows[0]?.setting ?? "")}. Latency is end to end, form to result, including the explanation.</p>`;
  $("evfoot").textContent = DASH.footnote;
}

function about() {
  const ms = [...Object.values(META.models)];
  if (META.history) ms.push({ name: META.history.name, metrics: META.history.metrics, features: { length: META.history.features }, server: true });
  const rows = ms.map((m) => { const t = m.metrics ?? {}, c = t["conformal_alpha_0.10"] ?? {};
    const f = (k) => t[k] !== undefined ? `${t[k].toFixed(3)} <span class="ci">(${t[k + "_ci"][0].toFixed(3)}–${t[k + "_ci"][1].toFixed(3)})</span>` : "—";
    return `<tr><td><b>${esc(m.name)}</b></td><td>${m.server ? "Local server" : "This browser"}</td><td>${m.features.length}</td><td>${f("auc")}</td><td>${f("ece")}</td><td>${c.abstention !== undefined ? Math.round(c.abstention * 100) + "%" : "—"}</td></tr>`; }).join("");
  $("mt").innerHTML = `<caption class="sr-only">Test-set metrics of the shipped models</caption><thead><tr><th scope="col">Model</th><th scope="col">Runs in</th><th scope="col">Inputs</th><th scope="col">ROC-AUC (95% CI)</th><th scope="col">Calibration error</th><th scope="col">Uncertain</th></tr></thead><tbody>${rows}</tbody>`;
  $("aboutnote").textContent = "Exported by backend/export.py from each bundle’s metrics.json. The browser models compute the same function as the ONNX models; tests/test_pwa.py checks agreement to 1e-4.";
  $("fver").textContent = META.version;
  const t1 = META.models.t1, tl = Math.round(t1.thresholds.t_low * 100), th = Math.round(t1.thresholds.t_high * 100);
  $("ab-bands").textContent = `With triage data only, low means under ${tl}% and high means ${th}% or more. Among patients marked low, at least 95 in 100 went home; among those marked high, at least 80 in 100 were admitted.`;
  const lv = [["t1", t1, "Age, sex, arrival, main complaint and six vital signs.", "This device"], ["t2", META.models.t2, "All of that, plus the triage level (ESI) if the hospital uses one.", "This device"]];
  if (META.history) lv.push(["t3lite", { metrics: META.history.metrics }, "All of that, plus past visits, medicines, conditions and lab results.", "A helper on this computer"]);
  $("ab-levelcards").innerHTML = lv.map(([k, m, uses, where]) => { const c = m.metrics["conformal_alpha_0.10"];
    return `<div class="lvl"><h3>${LEVEL[k]}</h3><p>${uses}</p><dl><div><dt>Ranks patients correctly</dt><dd><b class="num">${pct(m.metrics.auc)}</b> in 100</dd></div>
      <div><dt>Gives a clear answer for</dt><dd><b class="num">${pct(1 - c.abstention)}</b> in 100</dd></div><div><dt>Runs on</dt><dd>${where}</dd></div></dl></div>`; }).join("");
  const aucs = lv.map(([, m]) => pct(m.metrics.auc)), clear = lv.map(([, m]) => pct(1 - m.metrics["conformal_alpha_0.10"].abstention));
  const ece = Math.max(...lv.map(([, m]) => m.metrics.ece)) * 100;
  const f = META.examples.high.form, n = 200, t0 = performance.now();
  for (let k = 0; k < n; k++) M.predict(f, t1, META);
  const perMs = (performance.now() - t0) / n;
  $("ab-wins").innerHTML = [
    ["Tells patients apart well.", `Shown one patient who was admitted and one who went home, it picks the right one ${aucs.join(", ")} times in 100 at the three levels of detail.`],
    ["Its percentages are honest.", `On average its estimate was within ${ece < 1 ? "1 percentage point" : `${Math.ceil(ece)} percentage points`} of what actually happened.`],
    ["It knows when it is unsure.", `The confidence check keeps its 9-in-10 promise, and still gives a clear answer for ${Math.min(...clear)} to ${Math.max(...clear)} in 100 patients.`],
    ["It copes with missing readings.", "Even with every vital sign missing, its ability to tell patients apart drops only from 86 to 84 in 100."],
    ["It travels to new hospitals.", "Moved to a hospital it had never seen, it needed only 500 to 1,000 local visits to be reliable again, without retraining."],
    ["It is tiny and fast.", `The everyday model is about ${Math.round(DASH?.edge?.[0]?.size_kb ?? 9)} KB and answers in ${perMs < 1 ? "under a millisecond" : `${perMs.toFixed(1)} ms`} on this device.`],
  ].map(([b, t]) => `<li><b>${b}</b> ${t}</li>`).join("");
  const bytes = DASH?.benchmark?.find((b) => b.setting.includes("network none"))?.net_bytes;
  if (bytes !== undefined) $("ab-bytes").textContent = `${bytes} bytes`;
}

async function init() {
  if ("scrollRestoration" in history) history.scrollRestoration = "manual";
  try {
    const res = await fetch("data/models.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    META = await res.json();
    try { const d = await fetch("data/dashboard.json"); DASH = d.ok ? await d.json() : null; } catch { DASH = null; }
    build(); historyUI(); about(); home(); pill(); histAvailability(); route();
  } catch (e) {
    $("main").innerHTML = `<div class="wrap"><p class="errsum">Couldn’t load the model files (${esc(e.message)}). Reload the page. If it keeps failing, open it once while the server is running so it can cache itself.</p></div>`;
    return;
  }
  detectServer();
  if (!("serviceWorker" in navigator)) { $("swstate").textContent = "not in this browser; keep the internet on"; return; }
  try { await navigator.serviceWorker.register("sw.js"); await navigator.serviceWorker.ready; $("swstate").textContent = "yes, saved on this device"; }
  catch (e) { $("swstate").textContent = `not available (${e.message})`; }
}
init();
