/* Parameter-tuning page. Read-only: every number is rendered from /api/tuning. */
(function () {
  "use strict";

  const { api, esc, fmt, strategyLabel } = window.TSR;

  const METRIC_LABEL = {
    mAP: "mAP",
    MRR: "MRR",
    "ndcg@10": "nDCG@10",
    "precision@1": "P@1",
    "recall@10": "R@10",
  };

  const state = { data: null, byId: {}, strategy: "", family: "", metric: "mAP" };

  function metricOf(run, metric) {
    const value = run && run.metrics ? run.metrics[metric] : null;
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  }

  function numLabel(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return String(value);
    return String(Number(number.toFixed(4)));
  }

  function deltaText(delta, digits) {
    if (typeof delta !== "number" || !Number.isFinite(delta)) return "—";
    return `${delta >= 0 ? "+" : ""}${delta.toFixed(digits == null ? 4 : digits)}`;
  }

  function deltaClass(delta) {
    if (typeof delta !== "number" || !Number.isFinite(delta) || Math.abs(delta) < 1e-9) return "";
    return delta > 0 ? "good" : "bad";
  }

  function visibleRuns() {
    return state.data.runs.filter(
      (run) =>
        (!state.strategy || run.strategy === state.strategy) &&
        (!state.family || run.family === state.family)
    );
  }

  function stamp(value) {
    if (!value) return "—";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    const pad = (number) => String(number).padStart(2, "0");
    const day = `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(
      date.getUTCDate()
    )}`;
    return `generated ${day} ${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())} UTC`;
  }

  /* ------------------------------------------------------------------ */
  /* Header                                                              */
  /* ------------------------------------------------------------------ */
  function renderHead(report) {
    const ready = report.runs.filter((run) => run.ready).length;
    document.getElementById("tuning-intro").textContent =
      `${report.num_queries || 0} frozen queries from “${report.eval_set || "eval"}”, ` +
      `${ready}/${report.runs.length} runs complete, ` +
      `${report.bootstrap || 0}× bootstrap (seed ${report.seed}).`;

    document.getElementById("tuning-generated").textContent = stamp(report.generated_at);

    const integrity = report.integrity || {};
    const integrityPill = document.getElementById("tuning-integrity");
    if (integrity.unchanged === true) {
      integrityPill.textContent = "protected data untouched";
      integrityPill.dataset.tone = "ok";
    } else if (Array.isArray(integrity.changes) && integrity.changes.length) {
      integrityPill.textContent = `${integrity.changes.length} protected change(s)`;
      integrityPill.dataset.tone = "error";
    } else {
      integrityPill.textContent = "integrity not checked";
    }

    const checks = Object.values(report.baseline_check || {});
    const baselinePill = document.getElementById("tuning-baseline");
    if (!checks.length) {
      baselinePill.textContent = "baseline not compared";
    } else if (checks.every((check) => check.status === "match")) {
      baselinePill.textContent = "baseline reproduces /evaluation";
      baselinePill.dataset.tone = "ok";
    } else {
      const bad = checks.filter((check) => check.status !== "match").length;
      baselinePill.textContent = `${bad} baseline check(s) not matching`;
      baselinePill.dataset.tone = "error";
    }

    const notice = document.getElementById("protocol-notice");
    const protocol = report.protocol || {};
    notice.innerHTML =
      "<strong>Exploratory sweep, not a new benchmark.</strong> " +
      esc(protocol.caveat || "") +
      " Isolation: every run wrote to its own directory under <code>data/tuning/</code>.";
    if (protocol.exploratory === false) notice.classList.add("error");
  }

  /* ------------------------------------------------------------------ */
  /* Filters                                                             */
  /* ------------------------------------------------------------------ */
  function fillFilters(report) {
    const strategies = Array.from(new Set(report.runs.map((run) => run.strategy)));
    const families = Array.from(
      new Set(report.runs.filter((run) => !run.is_baseline).map((run) => run.family))
    );
    const strategySelect = document.getElementById("filter-strategy");
    strategySelect.innerHTML =
      '<option value="">All strategies</option>' +
      strategies
        .map((key) => `<option value="${esc(key)}">${esc(strategyLabel(key))}</option>`)
        .join("");
    const familySelect = document.getElementById("filter-family");
    familySelect.innerHTML =
      '<option value="">All families</option>' +
      families.map((family) => `<option value="${esc(family)}">${esc(family)}</option>`).join("");
  }

  /* ------------------------------------------------------------------ */
  /* Charts                                                              */
  /* ------------------------------------------------------------------ */
  function tipText(group, item, metric) {
    const run = item.run;
    const lines = [run.is_baseline ? `${run.id} · default` : run.id];
    lines.push(`${group.x_label}: ${run.value_label || numLabel(item.x)}`);
    lines.push(`${METRIC_LABEL[metric] || metric}: ${item.value.toFixed(4)}`);
    const delta = (run.delta_vs_baseline || {})[metric];
    if (typeof delta === "number") lines.push(`Δ vs baseline: ${deltaText(delta, 4)}`);
    return lines.join("\n");
  }

  function barPath(x, y, width, height) {
    const radius = Math.min(4, width / 2, Math.max(0, height));
    const bottom = y + height;
    return (
      `M${x.toFixed(1)},${bottom.toFixed(1)} L${x.toFixed(1)},${(y + radius).toFixed(1)} ` +
      `Q${x.toFixed(1)},${y.toFixed(1)} ${(x + radius).toFixed(1)},${y.toFixed(1)} ` +
      `L${(x + width - radius).toFixed(1)},${y.toFixed(1)} ` +
      `Q${(x + width).toFixed(1)},${y.toFixed(1)} ${(x + width).toFixed(1)},${(y + radius).toFixed(1)} ` +
      `L${(x + width).toFixed(1)},${bottom.toFixed(1)} Z`
    );
  }

  function lineChart(group, points, metric) {
    const W = 380, H = 232, ml = 50, mr = 16, mt = 26, mb = 38;
    const pw = W - ml - mr, ph = H - mt - mb;

    const series = points
      .map((point) => ({
        x: Number(point.x),
        run: point.run,
        is_baseline: Boolean(point.is_baseline),
        value: metricOf(point.run, metric),
      }))
      .filter((item) => Number.isFinite(item.x) && item.value !== null)
      .sort((a, b) => a.x - b.x);
    if (!series.length) return "";

    let minX = Math.min(...series.map((item) => item.x));
    let maxX = Math.max(...series.map((item) => item.x));
    if (minX === maxX) {
      minX -= 1;
      maxX += 1;
    }
    const values = series.map((item) => item.value);
    const span = Math.max(...values) - Math.min(...values) || 0.02;
    let minV = Math.min(...values) - span * 0.25;
    if (minV < 0 && Math.min(...values) >= 0) minV = 0;
    let maxV = Math.max(...values) + span * 0.25;
    if (maxV === minV) maxV = minV + 0.01;

    const sx = (x) => ml + ((x - minX) / (maxX - minX)) * pw;
    const sy = (v) => mt + ((maxV - v) / (maxV - minV)) * ph;

    const ticks = [maxV, (maxV + minV) / 2, minV];
    const grid = ticks
      .map((value) => {
        const y = sy(value);
        return `<line x1="${ml}" y1="${y.toFixed(1)}" x2="${ml + pw}" y2="${y.toFixed(1)}" class="viz-grid"/>
          <text x="${ml - 8}" y="${(y + 3.5).toFixed(1)}" text-anchor="end" class="viz-axis">${value.toFixed(3)}</text>`;
      })
      .join("");

    const base = series.find((item) => item.is_baseline);
    const baseLine = base
      ? `<line x1="${ml}" y1="${sy(base.value).toFixed(1)}" x2="${ml + pw}" y2="${sy(base.value).toFixed(1)}" class="viz-baseline"/>`
      : "";
    const path = series
      .map((item, index) => `${index ? "L" : "M"}${sx(item.x).toFixed(1)},${sy(item.value).toFixed(1)}`)
      .join(" ");
    const dots = series
      .map((item) => {
        const cx = sx(item.x), cy = sy(item.value);
        const tip = tipText(group, item, metric);
        return `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="4.5" class="viz-dot${item.is_baseline ? " is-baseline" : ""}"/>
          <circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="13" class="viz-hit" data-tip="${esc(tip)}" tabindex="0" aria-label="${esc(tip)}"><title>${esc(tip)}</title></circle>`;
      })
      .join("");
    const xTicks = series
      .map(
        (item) =>
          `<text x="${sx(item.x).toFixed(1)}" y="${(mt + ph + 18).toFixed(1)}" text-anchor="middle" class="viz-axis">${esc(numLabel(item.x))}</text>`
      )
      .join("");
    const best = series.reduce((a, b) => (b.value > a.value ? b : a));
    const bestLabel = `<text x="${sx(best.x).toFixed(1)}" y="${(sy(best.value) - 10).toFixed(1)}" text-anchor="middle" class="viz-value">${best.value.toFixed(3)}</text>`;

    return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(group.family)} — ${esc(METRIC_LABEL[metric] || metric)}">
      ${grid}${baseLine}
      <path d="${path}" class="viz-line"/>
      ${dots}${bestLabel}${xTicks}
      <text x="${ml + pw / 2}" y="${H - 4}" text-anchor="middle" class="viz-axis-dim">${esc(group.x_label)}</text>
    </svg>`;
  }

  function barChart(group, points, metric) {
    const W = 380, H = 232, ml = 50, mr = 16, mt = 26, mb = 42;
    const pw = W - ml - mr, ph = H - mt - mb;

    const series = points
      .map((point) => ({
        x: point.x,
        run: point.run,
        is_baseline: Boolean(point.is_baseline),
        value: metricOf(point.run, metric),
      }))
      .filter((item) => item.value !== null);
    if (!series.length) return "";

    const maxV = Math.max(...series.map((item) => item.value), 0);
    const top = maxV > 0 ? maxV * 1.18 : 1;
    const sy = (v) => mt + ((top - v) / top) * ph;
    const ticks = [top, top / 2, 0];
    const grid = ticks
      .map((value) => {
        const y = sy(value);
        return `<line x1="${ml}" y1="${y.toFixed(1)}" x2="${ml + pw}" y2="${y.toFixed(1)}" class="viz-grid"/>
          <text x="${ml - 8}" y="${(y + 3.5).toFixed(1)}" text-anchor="end" class="viz-axis">${value.toFixed(3)}</text>`;
      })
      .join("");

    const slot = pw / series.length;
    const width = Math.min(72, slot * 0.55);
    const bars = series
      .map((item, index) => {
        const cx = ml + slot * (index + 0.5);
        const x = cx - width / 2;
        const y = sy(item.value);
        const height = Math.max(0, mt + ph - y);
        const tip = tipText(group, item, metric);
        const label = item.is_baseline ? "default" : item.run.value_label || "variant";
        return `<path d="${barPath(x, y, width, height)}" class="viz-bar"/>
          <rect x="${(x - 5).toFixed(1)}" y="${mt}" width="${(width + 10).toFixed(1)}" height="${ph}" class="viz-hit" data-tip="${esc(tip)}" tabindex="0" aria-label="${esc(tip)}"><title>${esc(tip)}</title></rect>
          <text x="${cx.toFixed(1)}" y="${(y - 8).toFixed(1)}" text-anchor="middle" class="viz-value">${item.value.toFixed(3)}</text>
          <text x="${cx.toFixed(1)}" y="${(mt + ph + 18).toFixed(1)}" text-anchor="middle" class="viz-axis">${esc(label)}</text>`;
      })
      .join("");

    return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(group.family)} — ${esc(METRIC_LABEL[metric] || metric)}">
      ${grid}${bars}
      <text x="${ml + pw / 2}" y="${H - 4}" text-anchor="middle" class="viz-axis-dim">${esc(group.x_label)}</text>
    </svg>`;
  }

  function renderCharts() {
    const grid = document.getElementById("chart-grid");
    const groups = state.data.groups.filter(
      (group) =>
        (!state.strategy || group.strategy === state.strategy) &&
        (!state.family || group.family === state.family)
    );
    if (!groups.length) {
      grid.innerHTML =
        '<p class="empty chart-empty">No parameter family matches the current filters.</p>';
      return;
    }
    grid.innerHTML = groups
      .map((group) => {
        const points = (group.points || [])
          .map((point) => ({ ...point, run: state.byId[point.run_id] }))
          .filter((point) => point.run && point.run.ready);
        const header = `<header><h3>${esc(group.family)}</h3>
          <p class="chart-sub">${esc(strategyLabel(group.strategy))} · ${esc(group.x_label)}</p></header>`;
        if (!points.length) {
          return `<article class="chart-card">${header}
            <p class="chart-foot">No completed run for this family yet.</p></article>`;
        }
        const svg =
          group.kind === "bar"
            ? barChart(group, points, state.metric)
            : lineChart(group, points, state.metric);
        const foot =
          group.kind === "bar"
            ? "Bars compare the default with the variant; the y-axis starts at zero."
            : `Dashed line: default configuration (${esc(numLabel(group.baseline_x))}).`;
        return `<article class="chart-card" data-group="${esc(group.id)}">${header}
          <div class="chart-shell">${svg}</div>
          <p class="chart-foot">${foot}</p></article>`;
      })
      .join("");
  }

  /* ------------------------------------------------------------------ */
  /* Tables                                                              */
  /* ------------------------------------------------------------------ */
  function renderStats() {
    const runs = visibleRuns();
    const grid = document.getElementById("stat-grid");
    const strategies = Array.from(new Set(runs.map((run) => run.strategy)));
    if (!strategies.length) {
      grid.innerHTML = '<p class="empty chart-empty">No run matches the current filters.</p>';
      return;
    }
    const metric = state.metric;
    grid.innerHTML = strategies
      .map((key) => {
        const base = state.data.baselines[key];
        const baseValue = metricOf(base, metric);
        const best = runs
          .filter((run) => !run.is_baseline && run.strategy === key && run.ready)
          .reduce((winner, run) => {
            const value = metricOf(run, metric);
            if (value === null) return winner;
            return winner === null || value > metricOf(winner, metric) ? run : winner;
          }, null);
        const bestValue = best ? metricOf(best, metric) : null;
        const delta = bestValue !== null && baseValue !== null ? bestValue - baseValue : null;
        return `<article class="stat" data-strategy="${esc(key)}">
          <p class="stat-label">${esc(strategyLabel(key))}</p>
          <p class="stat-value">${fmt(baseValue, 4)}</p>
          <p class="stat-note">baseline ${esc(METRIC_LABEL[metric] || metric)}</p>
          <p class="stat-note">${
            best
              ? `best ${esc(best.id)} · ${fmt(bestValue, 4)} ` +
                `<span class="${deltaClass(delta)}">${deltaText(delta, 4)}</span>`
              : "no completed sweep run under these filters"
          }</p>
        </article>`;
      })
      .join("");
  }

  function runsRow(run) {
    const cells = (count) => `<td colspan="${count}">${esc(run.error || "run not ready")}</td>`;
    const runCell = `<td class="strategy-name">${esc(run.id)}<small>${esc(
      run.is_baseline ? "default config" : run.parameter || ""
    )}</small></td>`;
    if (!run.ready) {
      return `<tr>${runCell}<td>${esc(strategyLabel(run.strategy))}</td>
        <td>${esc(run.family)}</td><td class="num">${esc(run.value_label)}</td>${cells(9)}</tr>`;
    }
    const metrics = run.metrics || {};
    const interval = (run.ci || {}).mAP;
    const latency = run.latency || {};
    const delta = (run.delta_vs_baseline || {}).mAP;
    return `<tr>${runCell}
      <td>${esc(strategyLabel(run.strategy))}</td>
      <td>${esc(run.family)}</td>
      <td class="num">${esc(run.value_label)}</td>
      <td class="num"><span class="value">${fmt(metrics.mAP, 4)}</span>${
        interval
          ? `<span class="ci"> [${fmt(interval.low, 3)}, ${fmt(interval.high, 3)}]</span>`
          : ""
      }</td>
      <td class="num ${deltaClass(delta)}">${deltaText(delta, 4)}</td>
      <td class="num">${fmt(metrics.MRR, 4)}</td>
      <td class="num">${fmt(metrics["ndcg@10"], 4)}</td>
      <td class="num">${fmt(metrics["precision@1"], 4)}</td>
      <td class="num">${fmt(metrics["recall@10"], 4)}</td>
      <td class="num">${fmt(metrics.median_rank, 0)}</td>
      <td class="num">${fmt(latency.p50_ms, 1)} / ${fmt(latency.p95_ms, 1)} ms</td>
      <td class="num">${fmt(run.num_errors, 0)}</td>
    </tr>`;
  }

  function renderRunsTable() {
    const table = document.getElementById("runs-table");
    table.innerHTML = `<thead><tr>
      <th>Run / parameter</th><th>Strategy</th><th>Family</th><th>Value</th>
      <th>mAP [95% CI]</th><th>ΔmAP</th><th>MRR</th><th>nDCG@10</th><th>P@1</th>
      <th>R@10</th><th>median rank</th><th>p50 / p95</th><th>errors</th>
    </tr></thead><tbody>${visibleRuns().map(runsRow).join("")}</tbody>`;
  }

  function renderCategories() {
    const table = document.getElementById("category-table");
    const runs = visibleRuns();
    const categories = Array.from(
      new Set(
        runs.flatMap((run) => Object.keys(run.per_category || {}))
      )
    ).sort();
    const head = `<thead><tr><th>Run</th><th>Strategy</th>${categories
      .map((category) => `<th>${esc(category)}</th>`)
      .join("")}</tr></thead>`;
    const rows = runs.map((run) => {
      const base = state.data.baselines[run.strategy];
      const cells = categories
        .map((category) => {
          const group = (run.per_category || {})[category];
          if (!group) return '<td class="num">—</td>';
          const baseGroup = (base && base.per_category ? base.per_category[category] : null) || null;
          const delta =
            !run.is_baseline &&
            baseGroup &&
            typeof group.mAP === "number" &&
            typeof baseGroup.mAP === "number"
              ? group.mAP - baseGroup.mAP
              : null;
          return `<td class="num"><span class="value">${fmt(group.mAP, 4)}</span>
            <span class="ci ${deltaClass(delta)}">${delta === null ? "" : "Δ " + deltaText(delta, 3)}</span></td>`;
        })
        .join("");
      return `<tr><td class="strategy-name">${esc(run.id)}${
        run.is_baseline ? "<small>default config</small>" : ""
      }</td><td>${esc(strategyLabel(run.strategy))}</td>${cells}</tr>`;
    });
    table.innerHTML =
      head +
      `<tbody>${rows.join("") || `<tr><td colspan="${categories.length + 2}">No run matches the filters.</td></tr>`}</tbody>`;
  }

  function renderFailures() {
    const failures = state.data.failures || [];
    const section = document.getElementById("failures-section");
    section.hidden = failures.length === 0;
    if (!failures.length) return;
    document.getElementById("failures-table").innerHTML = `<thead><tr>
      <th>Run</th><th>Error</th></tr></thead><tbody>${failures
      .map(
        (failure) =>
          `<tr><td class="strategy-name">${esc(failure.id)}</td><td>${esc(failure.error)}</td></tr>`
      )
      .join("")}</tbody>`;
  }

  /* ------------------------------------------------------------------ */
  /* Tooltip                                                             */
  /* ------------------------------------------------------------------ */
  function wireTooltips() {
    const grid = document.getElementById("chart-grid");
    let tip = document.getElementById("viz-tooltip");
    if (!tip) {
      tip = document.createElement("div");
      tip.id = "viz-tooltip";
      tip.className = "chart-tooltip";
      tip.setAttribute("role", "tooltip");
      tip.hidden = true;
      document.body.appendChild(tip);
    }

    function show(target, x, y) {
      const text = target.getAttribute("data-tip") || "";
      tip.innerHTML = text
        .split("\n")
        .map((line) => `<span>${esc(line)}</span>`)
        .join("");
      tip.hidden = false;
      const pad = 14;
      const rect = tip.getBoundingClientRect();
      let left = x + pad;
      let top = y + pad;
      if (left + rect.width > window.innerWidth - 8) left = x - rect.width - pad;
      if (top + rect.height > window.innerHeight - 8) top = y - rect.height - pad;
      tip.style.left = `${Math.max(8, left)}px`;
      tip.style.top = `${Math.max(8, top)}px`;
    }

    function hide() {
      tip.hidden = true;
    }

    grid.addEventListener("pointermove", (event) => {
      const target = event.target.closest("[data-tip]");
      if (!target) return hide();
      show(target, event.clientX, event.clientY);
    });
    grid.addEventListener("pointerleave", hide);
    grid.addEventListener("focusin", (event) => {
      const target = event.target.closest("[data-tip]");
      if (!target) return;
      const rect = target.getBoundingClientRect();
      show(target, rect.left + rect.width / 2, rect.bottom);
    });
    grid.addEventListener("focusout", hide);
  }

  /* ------------------------------------------------------------------ */
  /* Boot                                                                */
  /* ------------------------------------------------------------------ */
  function render() {
    renderStats();
    renderCharts();
    renderRunsTable();
    renderCategories();
  }

  function bindFilters() {
    document.getElementById("filter-strategy").addEventListener("change", (event) => {
      state.strategy = event.target.value;
      render();
    });
    document.getElementById("filter-family").addEventListener("change", (event) => {
      state.family = event.target.value;
      render();
    });
    document.getElementById("filter-metric").addEventListener("change", (event) => {
      state.metric = event.target.value;
      render();
    });
  }

  async function init() {
    const status = document.getElementById("status");
    try {
      const report = await api("/api/tuning");
      state.data = report;
      state.byId = Object.fromEntries(report.runs.map((run) => [run.id, run]));
      document.getElementById("tuning-report").hidden = false;
      document.getElementById("empty-state").hidden = true;
      renderHead(report);
      fillFilters(report);
      renderFailures();
      render();
      status.textContent = "";
    } catch (error) {
      document.getElementById("tuning-report").hidden = true;
      if (error.status === 404) {
        document.getElementById("empty-state").hidden = false;
      } else {
        status.textContent = error.message;
        status.classList.add("error");
      }
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    bindFilters();
    wireTooltips();
    init();
  });
})();
