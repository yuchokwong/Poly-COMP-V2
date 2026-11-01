/* Evaluation report page. */
(function () {
  "use strict";

  const { api, esc, fmt, strategyLabel } = window.TSR;

  function cell(value, digits) {
    return `<td class="num">${fmt(value, digits)}</td>`;
  }

  function withCi(metrics, ci, name) {
    const value = metrics[name];
    const interval = ci && ci[name];
    const main = `<span class="value">${fmt(value, 4)}</span>`;
    if (!interval) return `<td class="num">${main}</td>`;
    return `<td class="num">${main}
      <span class="ci">[${fmt(interval.low, 3)}, ${fmt(interval.high, 3)}]</span></td>`;
  }

  function renderOverall(report) {
    const table = document.getElementById("overall-table");
    const head = `<thead><tr>
      <th>Strategy</th><th>mAP [95% CI]</th><th>MRR</th><th>nDCG@10</th>
      <th>P@1</th><th>R@10</th><th>median rank</th><th>p50 / p95 latency</th>
    </tr></thead>`;
    const rows = Object.entries(report.strategies || {})
      .map(([key, result]) => {
        if (!result.ready) {
          return `<tr><td class="strategy-name">${esc(strategyLabel(key))}
            <small>${esc(key)}</small></td>
            <td colspan="7">unavailable — ${esc(result.error || "")}</td></tr>`;
        }
        const metrics = result.metrics || {};
        const latency = result.latency || {};
        return `<tr>
          <td class="strategy-name">${esc(result.title || strategyLabel(key))}
            <small>${esc(key)}</small></td>
          ${withCi(metrics, result.ci, "mAP")}
          ${withCi(metrics, result.ci, "MRR")}
          ${withCi(metrics, result.ci, "ndcg@10")}
          ${cell(metrics["precision@1"], 4)}
          ${cell(metrics["recall@10"], 4)}
          ${cell(metrics["median_rank"], 0)}
          <td class="num">${fmt(latency.p50_ms, 1)} / ${fmt(latency.p95_ms, 1)} ms</td>
        </tr>`;
      })
      .join("");
    table.innerHTML = head + `<tbody>${rows}</tbody>`;
  }

  function renderCategories(report) {
    const table = document.getElementById("category-table");
    const keys = Object.keys(report.strategies || {});
    const categories = Array.from(
      new Set(
        keys.flatMap((key) =>
          Object.keys((report.strategies[key] || {}).per_category || {})
        )
      )
    ).sort();
    const head = `<thead><tr><th>Category</th>${keys
      .map((key) => `<th>${esc(strategyLabel(key))}</th>`)
      .join("")}</tr></thead>`;
    const rows = categories
      .map((category) => {
        const cells = keys
          .map((key) => {
            const group = ((report.strategies[key] || {}).per_category || {})[category];
            if (!group) return `<td class="num">—</td>`;
            return `<td class="num"><span class="value">${fmt(group.mAP, 4)}</span>
              <span class="ci">n=${group.num_queries}</span></td>`;
          })
          .join("");
        return `<tr><td>${esc(category)}</td>${cells}</tr>`;
      })
      .join("");
    table.innerHTML = head + `<tbody>${rows || '<tr><td colspan="4">No grouped metrics.</td></tr>'}</tbody>`;
  }

  function renderSplits(report) {
    const table = document.getElementById("split-table");
    if (!table) return;
    const keys = Object.keys(report.strategies || {});
    const splits = Array.from(
      new Set(
        keys.flatMap((key) => Object.keys((report.strategies[key] || {}).per_split || {}))
      )
    ).sort();
    const head = `<thead><tr><th>Split</th>${keys
      .map((key) => `<th>${esc(strategyLabel(key))}</th>`)
      .join("")}</tr></thead>`;
    const rows = splits
      .map((split) => {
        const cells = keys
          .map((key) => {
            const group = ((report.strategies[key] || {}).per_split || {})[split];
            if (!group) return `<td class="num">—</td>`;
            return `<td class="num"><span class="value">${fmt(group.mAP, 4)}</span>
              <span class="ci">n=${group.num_queries}</span></td>`;
          })
          .join("");
        return `<tr><td>${esc(split)}</td>${cells}</tr>`;
      })
      .join("");
    table.innerHTML =
      head + `<tbody>${rows || '<tr><td colspan="4">No split metadata.</td></tr>'}</tbody>`;
  }

  function renderPairwise(report) {
    const table = document.getElementById("pairwise-table");
    const head = `<thead><tr><th>Pair</th><th>Metric</th><th>Δ</th><th>95% CI</th>
      <th>p raw</th><th>p Holm</th><th>Significant</th><th>W / T / L</th></tr></thead>`;
    const rows = (report.pairwise || []).flatMap((pair) =>
      (pair.comparisons || []).map((comp) => {
        const wins = comp.wins || {};
        const cls = comp.significant ? (comp.delta > 0 ? "good" : "bad") : "";
        return `<tr>
          <td>${esc(pair.a)} vs ${esc(pair.b)}</td>
          <td>${esc(comp.metric)}</td>
          <td class="num ${cls}">${comp.delta >= 0 ? "+" : ""}${fmt(comp.delta, 4)}</td>
          <td class="num ci">[${fmt(comp.ci_low, 4)}, ${fmt(comp.ci_high, 4)}]</td>
          <td class="num">${fmt(comp.p_raw, 4)}</td>
          <td class="num">${fmt(comp.p_holm, 4)}</td>
          <td class="${comp.significant ? "good" : ""}">${comp.significant ? "yes" : "no"}</td>
          <td class="num">${wins.win || 0} / ${wins.tie || 0} / ${wins.loss || 0}</td>
        </tr>`;
      })
    );
    table.innerHTML = head + `<tbody>${rows.join("") || '<tr><td colspan="8">No pairwise tests.</td></tr>'}</tbody>`;
  }

  function renderAblations(report) {
    const table = document.getElementById("ablation-table");
    const head = `<thead><tr><th>Variant</th><th>mAP</th><th>MRR</th><th>nDCG@10</th>
      <th>p50 latency</th></tr></thead>`;
    const rows = Object.entries(report.ablations || {}).map(([label, result]) => {
      if (!result.ready) {
        return `<tr><td>${esc(label)}</td><td colspan="4">${esc(result.error || "failed")}</td></tr>`;
      }
      const metrics = result.metrics || {};
      return `<tr><td>${esc(label)}</td>
        ${cell(metrics.mAP, 4)}${cell(metrics.MRR, 4)}${cell(metrics["ndcg@10"], 4)}
        <td class="num">${fmt((result.latency || {}).p50_ms, 1)} ms</td></tr>`;
    });
    table.innerHTML = head + `<tbody>${rows.join("") || '<tr><td colspan="5">Run \`tsr compare --ablations\`.</td></tr>'}</tbody>`;
  }

  function renderDiagnostics(report) {
    const table = document.getElementById("diagnostic-table");
    if (!table) return;
    const head = `<thead><tr><th>Strategy</th><th>mAP (diagnostic)</th>
      <th>Δ vs main mAP</th><th>MRR</th><th>nDCG@10</th><th>queries</th></tr></thead>`;
    const rows = Object.entries(report.diagnostics || {}).map(([key, data]) => {
      const main = (((report.strategies || {})[key] || {}).metrics || {}).mAP;
      const delta = typeof main === "number" ? data.mAP - main : null;
      const cls = delta === null ? "" : delta > 0 ? "bad" : "good";
      const text =
        delta === null ? "—" : `${delta >= 0 ? "+" : ""}${fmt(delta, 4)}`;
      return `<tr>
        <td class="strategy-name">${esc(strategyLabel(key))} <small>${esc(key)}</small></td>
        ${cell(data.mAP, 4)}
        <td class="num ${cls}">${text}</td>
        ${cell(data.MRR, 4)}${cell(data["ndcg@10"], 4)}${cell(data.num_queries, 0)}
      </tr>`;
    });
    table.innerHTML =
      head +
      `<tbody>${rows.join("") ||
        '<tr><td colspan="6">Run <code>tsr eval --queries data/eval/memecap_eval_v1.raw_captions.json</code>.</td></tr>'}</tbody>`;
  }

  function renderProbe(report) {
    const table = document.getElementById("probe-table");
    const probe = report.probe || {};
    const rows = [
      ["Images in gallery", fmt(probe.num_images, 0)],
      ["Evaluation queries", fmt(report.num_queries, 0)],
      ["Eval set", report.eval_set || "—"],
      ["Leakage checks passed", fmt(probe.leakage_passed, 0)],
      ["Leakage flagged", fmt(probe.leakage_flagged, 0)],
      ["VLM descriptions", fmt(probe.vlm_captions, 0)],
      ["Bootstrap resamples", fmt(report.n_boot, 0)],
      ["Seed", fmt(report.seed, 0)],
    ];
    table.innerHTML = `<tbody>${rows
      .map(([label, value]) => `<tr><td>${esc(label)}</td><td class="num">${esc(value)}</td></tr>`)
      .join("")}</tbody>`;
  }

  async function init() {
    const status = document.getElementById("status");
    try {
      const report = await api("/api/eval/summary");
      document.getElementById("report").hidden = false;
      document.getElementById("empty-state").hidden = true;
      document.getElementById("eval-intro").textContent =
        `${report.num_queries || 0} queries from “${report.eval_set || "eval"}”, ` +
        `each run through all ready strategies on the same gallery.`;
      document.getElementById("eval-generated").textContent =
        report.generated_at ? `generated ${report.generated_at.replace("T", " ").replace("+00:00", " UTC")}` : "—";
      renderOverall(report);
      renderCategories(report);
      renderSplits(report);
      renderPairwise(report);
      renderAblations(report);
      renderDiagnostics(report);
      renderProbe(report);
      status.textContent = "";
    } catch (error) {
      document.getElementById("report").hidden = true;
      document.getElementById("empty-state").hidden = false;
      if (error.status !== 404) {
        status.textContent = error.message;
        status.classList.add("error");
      }
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
