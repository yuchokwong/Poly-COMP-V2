/* Three-column comparison page. */
(function () {
  "use strict";

  const { api, esc, fmt, strategyLabel, queryFromUrl, setUrlState } = window.TSR;

  const COLUMN_TITLE = {
    clip: "CLIP only",
    clip_nlp: "CLIP + NLP",
    vlm: "VLM",
  };

  function status(message, isError) {
    const node = document.getElementById("status");
    if (!node) return;
    node.textContent = message || "";
    node.classList.toggle("error", Boolean(isError));
  }

  function captionFor(result) {
    return (result.img_captions || [])[0] || (result.meme_captions || [])[0] || "";
  }

  function miniCard(result, strategy, shared) {
    return `
      <button class="mini-card${shared ? " shared" : ""}" type="button"
              data-id="${result.id}" data-strategy="${esc(strategy)}"
              title="${esc(result.title || result.image_id)}">
        <img loading="lazy" src="${esc(result.thumbnail)}" alt="">
        <span>
          <span class="mini-rank">#${result.rank} · ${fmt(result.score, 3)}</span>
          <span class="mini-title">${esc(result.title || result.image_id)}</span>
          <span class="mini-caption">${esc(captionFor(result))}</span>
        </span>
      </button>`;
  }

  function render(payload) {
    const columns = payload.columns || {};
    const counts = new Map();
    for (const column of Object.values(columns)) {
      for (const result of column.results || []) {
        counts.set(result.id, (counts.get(result.id) || 0) + 1);
      }
    }
    let sharedCount = 0;
    let uniqueCount = 0;
    counts.forEach((count) => {
      uniqueCount += 1;
      if (count > 1) sharedCount += 1;
    });

    for (const element of document.querySelectorAll(".compare-column")) {
      const key = element.dataset.column;
      const column = columns[key] || { ready: false, results: [], error: "no response" };
      const pill = element.querySelector("[data-column-status]");
      const meta = element.querySelector("[data-column-meta]");
      const container = element.querySelector("[data-column-results]");

      pill.classList.remove("ready", "unavailable");
      if (column.ready) {
        pill.classList.add("ready");
        pill.textContent = "ready";
        meta.textContent = `${column.count} results · ${fmt(column.latency_ms, 1)} ms`;
      } else {
        pill.classList.add("unavailable");
        pill.textContent = "unavailable";
        meta.textContent = column.error || "This strategy is not built.";
      }

      if (!column.results || !column.results.length) {
        container.innerHTML = `<p class="text-link">${esc(
          column.ready ? "No results for this query." : column.error || "Not available."
        )}</p>`;
        continue;
      }
      container.innerHTML = column.results
        .map((result) => miniCard(result, key, (counts.get(result.id) || 0) > 1))
        .join("");
    }

    const agreement = document.getElementById("agreement");
    const summary = document.getElementById("agreement-summary");
    if (agreement && summary && uniqueCount) {
      agreement.hidden = false;
      summary.textContent =
        `${sharedCount} of ${uniqueCount} retrieved images appear in more than one column. ` +
        "Outlined cards are shared hits; use them to see where the pipelines agree.";
    }
    const pill = document.getElementById("compare-mode");
    if (pill) {
      pill.textContent = `${uniqueCount} unique hits`;
      pill.classList.add("ready");
    }
  }

  async function run(query, k) {
    const trimmed = query.trim();
    if (!trimmed) return;
    status("Running all ready strategies…");
    const button = document.querySelector("#compare-form button[type=submit]");
    if (button) button.disabled = true;
    try {
      const payload = await api(
        `/api/compare?q=${encodeURIComponent(trimmed)}&k=${encodeURIComponent(k)}`
      );
      render(payload);
      const latencies = Object.values(payload.columns || {})
        .filter((column) => column.ready)
        .map((column) => Number(column.latency_ms) || 0);
      const slowest = latencies.length ? Math.max(...latencies) : 0;
      const unavailable = Object.keys(payload.errors || {});
      status(
        `Query: “${trimmed}” · slowest column ${fmt(slowest, 1)} ms` +
          (unavailable.length ? ` · unavailable: ${unavailable.join(", ")}` : "")
      );
      setUrlState(trimmed, k);
      for (const element of document.querySelectorAll(".compare-column")) {
        const key = element.dataset.column;
        element.querySelector("h2").textContent = COLUMN_TITLE[key] || strategyLabel(key);
      }
    } catch (error) {
      status(error.message, true);
    } finally {
      if (button) button.disabled = false;
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("compare-form");
    const input = document.getElementById("query");
    const range = document.getElementById("top-k");
    const rangeOut = document.getElementById("top-k-value");

    range.addEventListener("input", () => {
      rangeOut.textContent = range.value;
    });
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      run(input.value, Number(range.value));
    });
    document.querySelectorAll("[data-example]").forEach((button) => {
      button.addEventListener("click", () => {
        input.value = button.dataset.example;
        run(input.value, Number(range.value));
      });
    });

    document.getElementById("columns").addEventListener("click", (event) => {
      const card = event.target.closest("[data-id]");
      if (!card) return;
      window.TSR.openDetail(Number(card.dataset.id), card.dataset.strategy);
    });

    const state = queryFromUrl();
    if (state.k) {
      range.value = String(Math.min(Number(range.max), Math.max(Number(range.min), state.k)));
      rangeOut.textContent = range.value;
    }
    if (state.q) {
      input.value = state.q;
      run(state.q, Number(range.value));
    }
  });
})();
