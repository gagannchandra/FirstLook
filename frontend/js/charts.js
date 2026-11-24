/* FirstLook charts: dependency-free SVG line charts (offline). One y-axis, thin 2px lines, recessive grid,
   legend + direct end labels, crosshair tooltip on hover/touch, and a data-table view for every chart. */
(function (root) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

  /**
   * @param {HTMLElement} el
   * @param {{series: {name: string, color: string, points: number[][], dash?: boolean, label?: boolean}[],
   *          x: {min: number, max: number, ticks: number[], fmt: (v:number)=>string, title: string, log?: boolean},
   *          y: {min: number, max: number, ticks: number[], fmt: (v:number)=>string, title: string},
   *          diagonal?: boolean, band?: number[], aria: string}} o
   */
  function line(el, o) {
    const W = 520, H = 300, m = { l: 46, r: 92, t: 12, b: 40 };
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const tx = (v) => o.x.log ? Math.log10(Math.max(v, 1)) : v;
    const sx = (v) => m.l + (tx(v) - tx(o.x.min)) / (tx(o.x.max) - tx(o.x.min)) * pw;
    const sy = (v) => m.t + ph - (v - o.y.min) / (o.y.max - o.y.min) * ph;
    const clampY = (v) => Math.min(Math.max(v, o.y.min), o.y.max);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.aria)}">`;
    for (const t of o.y.ticks) s += `<line x1="${m.l}" x2="${m.l + pw}" y1="${sy(t)}" y2="${sy(t)}" stroke="var(--grid)"/>` +
      `<text x="${m.l - 8}" y="${sy(t) + 4}" text-anchor="end">${o.y.fmt(t)}</text>`;
    for (const t of o.x.ticks) s += `<text x="${sx(t)}" y="${m.t + ph + 18}" text-anchor="middle">${o.x.fmt(t)}</text>`;
    s += `<line x1="${m.l}" x2="${m.l + pw}" y1="${m.t + ph}" y2="${m.t + ph}" stroke="var(--axis)"/>`;
    s += `<text x="${m.l + pw / 2}" y="${H - 4}" text-anchor="middle">${esc(o.x.title)}</text>`;
    s += `<text transform="translate(12 ${m.t + ph / 2}) rotate(-90)" text-anchor="middle">${esc(o.y.title)}</text>`;
    if (o.band) s += `<rect x="${m.l}" width="${pw}" y="${sy(o.band[1])}" height="${sy(o.band[0]) - sy(o.band[1])}" fill="var(--brand-soft)" opacity=".7"/>`;
    if (o.diagonal) s += `<line x1="${sx(o.x.min)}" y1="${sy(o.y.min)}" x2="${sx(o.x.max)}" y2="${sy(o.y.max)}" stroke="var(--axis)" stroke-dasharray="4 4"/>`;
    for (const se of o.series) {
      const d = se.points.map(([x, y], i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(clampY(y)).toFixed(1)}`).join("");
      s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"${se.dash ? ' stroke-dasharray="5 4"' : ""}/>`;
    }
    // direct end labels (≤ 4 labelled series), nudged apart so they never collide
    const ends = o.series.filter((se) => se.label !== false).map((se) => { const p = se.points[se.points.length - 1]; return { se, y: sy(clampY(p[1])), x: sx(p[0]) }; })
      .sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 14) ends[i].y = ends[i - 1].y + 14;
    for (const e of ends) s += `<circle cx="${e.x}" cy="${sy(clampY(e.se.points[e.se.points.length - 1][1]))}" r="3.5" fill="${e.se.color}" stroke="#fff" stroke-width="2"/>` +
      `<text class="dl" x="${e.x + 8}" y="${e.y + 4}">${esc(e.se.short ?? e.se.name)}</text>`;
    s += `<line class="xh" y1="${m.t}" y2="${m.t + ph}" stroke="var(--ink2)" stroke-width="1" opacity="0"/>`;
    s += `<g class="dots"></g><rect class="hit" x="${m.l}" y="${m.t}" width="${pw}" height="${ph}" fill="transparent"/></svg>`;

    const legend = `<div class="legend">${o.series.map((se) => `<span><i class="${se.dash ? "dash" : ""}" style="background:${se.color}"></i>${esc(se.name)}</span>`).join("")}</div>`;
    const xs = [...new Set(o.series.flatMap((se) => se.points.map((p) => p[0])))].sort((a, b) => a - b);
    const table = `<details class="data"><summary>Show data table</summary><div class="tablewrap"><table><thead><tr><th scope="col">${esc(o.x.title)}</th>${o.series.map((se) => `<th scope="col">${esc(se.name)}</th>`).join("")}</tr></thead><tbody>` +
      xs.map((x) => `<tr><td>${o.x.fmt(x)}</td>${o.series.map((se) => { const p = se.points.find((q) => q[0] === x); return `<td>${p ? o.y.fmt(p[1], true) : "—"}</td>`; }).join("")}</tr>`).join("") + "</tbody></table></div></details>";
    el.innerHTML = `${legend}<div class="plot">${s}<div class="tip" hidden></div></div>${table}`;

    // crosshair tooltip: nearest x across all series
    const svg = el.querySelector("svg"), tip = el.querySelector(".tip"), xh = svg.querySelector(".xh"), dots = svg.querySelector(".dots");
    const move = (ev) => {
      const r = svg.getBoundingClientRect(), k = W / r.width;
      const px = ((ev.touches ? ev.touches[0].clientX : ev.clientX) - r.left) * k;
      let best = xs[0];
      for (const x of xs) if (Math.abs(sx(x) - px) < Math.abs(sx(best) - px)) best = x;
      const rows = o.series.map((se) => ({ se, p: se.points.find((q) => q[0] === best) })).filter((d) => d.p);
      xh.setAttribute("x1", sx(best)); xh.setAttribute("x2", sx(best)); xh.setAttribute("opacity", ".35");
      dots.innerHTML = rows.map((d) => `<circle cx="${sx(best)}" cy="${sy(clampY(d.p[1]))}" r="4.5" fill="${d.se.color}" stroke="#fff" stroke-width="2"/>`).join("");
      tip.innerHTML = `<b>${esc(o.x.title)}: ${o.x.fmt(best, true)}</b>` + rows.map((d) => `<div><i style="background:${d.se.color}"></i>${esc(d.se.name)} <b style="display:inline">${o.y.fmt(d.p[1], true)}</b></div>`).join("");
      tip.hidden = false;
      const top = Math.min(...rows.map((d) => sy(clampY(d.p[1]))));
      tip.style.left = `${Math.min(Math.max(sx(best) / k, 80), r.width - 80)}px`; tip.style.top = `${top / k}px`;
    };
    const leave = () => { tip.hidden = true; xh.setAttribute("opacity", "0"); dots.innerHTML = ""; };
    const hit = svg.querySelector(".hit");
    hit.addEventListener("pointermove", move); hit.addEventListener("pointerleave", leave);
    hit.addEventListener("touchstart", move, { passive: true });
  }

  /** Ring gauge (risk dial) as an SVG string. */
  function ring(p, color, label) {
    const r = 70, c = 2 * Math.PI * r;
    return `<svg viewBox="0 0 170 170"><circle cx="85" cy="85" r="${r}" fill="none" stroke="#e8ecf4" stroke-width="14"/>` +
      `<circle cx="85" cy="85" r="${r}" fill="none" stroke="${color}" stroke-width="14" stroke-linecap="round" stroke-dasharray="${c}" stroke-dashoffset="${c * (1 - Math.max(p, 0.005))}"/></svg>` +
      `<div class="v"><span><b class="num">${Math.round(p * 100)}<span style="font-size:.5em">%</span></b><small>${esc(label)}</small></span></div>`;
  }

  root.FirstLookCharts = { line, ring };
})(self);
