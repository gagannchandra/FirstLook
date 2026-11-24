/* FirstLook offline model (PWA, L3). Mirrors src/features.py + the shipped LR bundle exactly.
   Works in the browser (window.FirstLookModel) and in Node (module.exports) for parity tests. */
(function (root) {
  "use strict";
  const RANGES = {
    triage_vital_hr: [30, 300], triage_vital_sbp: [30, 400], triage_vital_dbp: [25, 300],
    triage_vital_rr: [8, 70], triage_vital_o2: [60, 100], triage_vital_temp: [90, 110],
  };

  function cleanVital(name, v) {
    if (v === null || v === undefined || Number.isNaN(v)) return NaN;
    if (name === "triage_vital_o2" && v === 100) v = 99; // D9: SpO2 100 -> 99
    const r = RANGES[name];
    if (r && (v < r[0] || v >= r[1])) return NaN;
    return v;
  }

  function featureValue(f, form, meta) {
    if (f === "age") return form.age;
    if (f === "sex_male") return form.gender === "Male" ? 1 : 0;
    if (f.startsWith("arrival=")) return form.arrivalmode === f.slice(8) ? 1 : 0;
    if (f.startsWith("hour=")) return form.arrivalhour_bin === f.slice(5) ? 1 : 0;
    if (f === "triage_vital_o2_device") {
      const v = form[f];
      return v === null || v === undefined ? NaN : v;
    }
    if (meta.vitals.includes(f)) return cleanVital(f, form[f]);
    if (f.startsWith("cc_")) return (form.complaints || []).includes(f) ? 1 : 0;
    if (f === "esi") return form.esi === null || form.esi === undefined ? NaN : Number(form.esi);
    throw new Error("unknown feature " + f);
  }

  function interp(x, xs, ys) {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
    let lo = 0, hi = xs.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= x) lo = mid; else hi = mid; }
    const t = (x - xs[lo]) / (xs[hi] - xs[lo] || 1);
    return ys[lo] + t * (ys[hi] - ys[lo]);
  }

  function calibrate(p, c) {
    if (c.kind === "platt") {
      const q = Math.min(Math.max(p, 1e-7), 1 - 1e-7);
      return 1 / (1 + Math.exp(-(c.a * Math.log(q / (1 - q)) + c.b)));
    }
    if (c.kind === "isotonic") return Math.min(Math.max(interp(p, c.x, c.y), 1e-4), 1 - 1e-4);
    return p;
  }

  function predict(form, model, meta) {
    const t0 = (typeof performance !== "undefined" ? performance : Date).now();
    const ex = model.explain, feats = model.features, n = feats.length;
    const x = feats.map((f) => featureValue(f, form, meta));
    let z = ex.intercept;
    const contrib = new Array(n).fill(0);
    for (let j = 0; j < n; j++) {
      const xi = Number.isNaN(x[j]) ? ex.median[j] : x[j];
      contrib[j] = ex.coef[j] * (xi - ex.mean[j]) / ex.scale[j];
    }
    ex.indicator_features.forEach((f, k) => {
      const j = n + k, v = Number.isNaN(x[f]) ? 1 : 0;
      contrib[f] += ex.coef[j] * (v - ex.mean[j]) / ex.scale[j];
    });
    for (let j = 0; j < n; j++) z += contrib[j];
    const raw = 1 / (1 + Math.exp(-z));
    const p = calibrate(raw, model.calibrator);
    const cf = model.conformal;
    const admit = (1 - p) <= cf.qhat_admit, discharge = p <= cf.qhat_discharge;
    const conformal = admit && !discharge ? "likely_admission" : discharge && !admit ? "likely_discharge" : "uncertain";
    const th = model.thresholds;
    const band = p < th.t_low ? "Low" : p >= th.t_high ? "High" : "Medium";
    const ms = (typeof performance !== "undefined" ? performance : Date).now() - t0;
    return { probability: p, raw, band, conformal, contributions: groupContrib(feats, contrib, form, meta), ms };
  }

  function groupContrib(feats, contrib, form, meta) {
    const g = {};
    const add = (k, label, v) => { if (!g[k]) g[k] = { label, value: 0 }; g[k].value += v; };
    const names = { triage_vital_hr: ["Heart rate", " bpm"], triage_vital_sbp: ["Systolic BP", " mmHg"],
      triage_vital_dbp: ["Diastolic BP", " mmHg"], triage_vital_rr: ["Respiratory rate", "/min"],
      triage_vital_o2: ["SpO₂", "%"], triage_vital_temp: ["Temperature", " °F"] };
    feats.forEach((f, j) => {
      const v = contrib[j];
      if (f === "age") add("age", "Age " + form.age, v);
      else if (f === "sex_male") add("sex", "Sex: " + form.gender, v);
      else if (f.startsWith("arrival=")) add("arrival", "Arrival: " + (form.arrivalmode || "not recorded").toLowerCase(), v);
      else if (f.startsWith("hour=")) add("hour", "Arrival time " + form.arrivalhour_bin + "h", v);
      else if (f === "triage_vital_o2_device") add(f, form[f] === 1 ? "On supplemental oxygen" : form[f] === 0 ? "Room air" : "Oxygen use not recorded", v);
      else if (names[f]) add(f, form[f] === null || form[f] === undefined ? names[f][0] + " not measured" : names[f][0] + " " + form[f] + names[f][1], v);
      else if (f.startsWith("cc_")) add("cc", "Chief complaint: " + ((form.complaints || []).map((c) => meta.complaint_labels[c]).join(", ") || "none"), v);
      else if (f === "esi") add("esi", form.esi ? "ESI level " + form.esi : "ESI not recorded", v);
    });
    return Object.values(g).filter((d) => Math.abs(d.value) > 0.05).sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  }

  /** Seeded PRNG (mulberry32): reproducible random patients from a 32-bit seed. */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /**
   * Random synthetic patient (same hand-set acuity profiles as app/random_patient.py, exported to
   * models.json). Never sampled from data. Returns { form, meta }.
   * @param {number} seed  @param {"low"|"medium"|"high"|null} acuity  @param {boolean} allowMissing
   */
  function randomPatient(seed, acuity, allowMissing, meta) {
    const R = meta.random, rnd = mulberry32(seed);
    const level = R.profiles[acuity] ? acuity : ["low", "medium", "high"][Math.floor(rnd() * 3)];
    const P = R.profiles[level];
    const u = ([lo, hi]) => lo + rnd() * (hi - lo);
    const pick = (w) => {
      const keys = Object.keys(w), tot = keys.reduce((s, k) => s + w[k], 0);
      let r = rnd() * tot;
      for (const k of keys) { r -= w[k]; if (r <= 0) return k; }
      return keys[keys.length - 1];
    };
    const sbp = Math.round(u(P.sbp));
    const dbp = Math.min(Math.max(Math.round(sbp * u([0.55, 0.68])), 40), sbp - 15);
    const pool = P.complaints.filter((c) => meta.complaint_order.includes(c));
    const nCc = rnd() < 0.8 ? 1 : 2, complaints = [];
    while (complaints.length < Math.min(nCc, pool.length)) {
      const c = pool[Math.floor(rnd() * pool.length)];
      if (!complaints.includes(c)) complaints.push(c);
    }
    const form = {
      age: Math.round(u(P.age)), gender: rnd() < 0.5 ? "Female" : "Male", arrivalmode: pick(P.arrival),
      arrivalhour_bin: R.hour_bins[Math.floor(rnd() * R.hour_bins.length)],
      triage_vital_hr: Math.round(u(P.hr)), triage_vital_rr: Math.round(u(P.rr)), triage_vital_sbp: sbp,
      triage_vital_dbp: dbp, triage_vital_o2: Math.round(u(P.o2)), triage_vital_o2_device: rnd() < P.oxygen_p ? 1 : 0,
      triage_vital_temp: Math.round(u(P.temp) * 10) / 10, complaints, esi: null,
    };
    const missing = [];
    if (allowMissing && rnd() < 0.6) {
      const v = R.missable[Math.floor(rnd() * R.missable.length)];
      form[v] = null; if (v === "triage_vital_o2") form.triage_vital_o2_device = null;
      missing.push(v);
    }
    return { form, meta: { seed, acuity: level, missing, esi: Number(pick(P.esi)) } };
  }

  const api = { predict, featureValue, calibrate, randomPatient };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.FirstLookModel = api;
})(typeof self !== "undefined" ? self : this);
