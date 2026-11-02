/* Home overview and the shared strategy search page. */
(function () {
  "use strict";

  const { api, esc, fmt, pct, strategyLabel, componentLabel, queryFromUrl, setUrlState } = window.TSR;

  const PAGE = document.body.dataset.page;

  const STRATEGY_PAGE = {
    "/clip": {
      key: "clip",
      kicker: "Strategy A · pixels only",
      title: "CLIP only",
      description:
        "The OpenCLIP text encoder searches the shared image index. No captions, " +
        "annotations or generated descriptions — a zero-knowledge baseline for visual similarity.",
    },
    "/clip-nlp": {
      key: "clip_nlp",
      kicker: "Strategy B · human annotations",
      title: "CLIP + NLP",
      description:
        "CLIP visual similarity fused with a dense index over Meme-Cap captions and " +
        "field-weighted BM25, then optionally reranked by a cross-encoder. Uses the human labels.",
    },
    "/vlm": {
      key: "vlm",
      kicker: "Strategy C · generated labels",
      title: "VLM descriptions",
      description:
        "Qwen2.5-VL reads each meme offline and writes scene, OCR and meaning text. " +
        "Queries are matched against those descriptions, optionally fused with CLIP.",
    },
  };

  const SCORE_LABEL = {
    clip: "CLIP",
    clip_nlp: "Fused score",
    vlm: "VLM score",
  };

  // ------------------------------------------------------------------ //
  // Home
  // ------------------------------------------------------------------ //
  function renderCardStatus(card, item, stats) {
    card.classList.remove("skeleton");
    const pill = card.querySelector("[data-ready]");
    if (pill) {
      pill.classList.remove("ready", "unavailable");
      pill.classList.add(item && item.ready ? "ready" : "unavailable");
      pill.textContent = item && item.ready ? "ready" : "unavailable";
      if (item && item.error) pill.title = item.error;
    }
    const existing = card.querySelector(".meta-line");
    if (existing) existing.remove();
    if (item && item.ready) {
      const parts = [];
      if (stats && stats.num_images) parts.push(`${fmt(stats.num_images)} images`);
      if (item.num_docs) parts.push(`${fmt(item.num_docs)} docs`);
      if (item.coverage != null) parts.push(`${fmt(item.coverage * 100, 0)}% coverage`);
      if (item.dim) parts.push(`dim ${item.dim}`);
      const line = document.createElement("p");
      line.className = "meta-line";
      line.textContent = parts.join(" · ");
      card.querySelector("footer").before(line);
    }
  }

  async function initHome() {
    try {
      const [health, stats] = await Promise.all([api("/api/health"), api("/api/stats")]);
      document.getElementById("stat-images").textContent = fmt(stats.dataset.num_images);
      document.getElementById("stat-vlm").textContent = fmt(stats.dataset.vlm_docs);
      const ready = (health.strategies || []).filter((item) => item.ready).length;
      document.getElementById("stat-ready").textContent = `${ready}/${(health.strategies || []).length}`;
      for (const card of document.querySelectorAll(".strategy-card")) {
        const key = card.dataset.strategy;
        const item = (stats.strategies || []).find((entry) => (entry.key || entry.strategy) === key);
        const summary = item && item.key ? item : item;
        renderCardStatus(card, summary || { ready: false }, summary);
      }
    } catch (error) {
      const box = document.getElementById("home-error");
      if (box) {
        box.hidden = false;
        box.querySelector("[data-error-text]").textContent = error.message;
      }
    }
  }

  // ------------------------------------------------------------------ //
  // Search page
  // ------------------------------------------------------------------ //
  let currentStrategy = "clip";
  let dialogStrategy = "clip";
  let lastQuery = "";

  function status(message, isError) {
    const node = document.getElementById("status");
    if (!node) return;
    node.textContent = message || "";
    node.classList.toggle("error", Boolean(isError));
  }

  function componentBars(components) {
    const order = ["clip", "doc", "bm25", "rerank"];
    const entries = order
      .filter((name) => components[name] != null && components[name] !== 0)
      .map((name) => [name, components[name]]);
    if (!entries.length) return "";
    const rows = entries
      .map(
        ([name, value]) => `
        <div class="component" data-name="${esc(name)}">
          <span>${esc(componentLabel(name))}</span>
          <span class="bar"><i style="width:${pct(value).toFixed(1)}%"></i></span>
          <span>${fmt(value, 3)}</span>
        </div>`
      )
      .join("");
    return `<div class="components">${rows}</div>`;
  }

  function captionFor(result) {
    const scene = (result.img_captions || [])[0];
    const meaning = (result.meme_captions || [])[0];
    return scene || meaning || "No annotation available.";
  }

  function renderResults(payload) {
    const grid = document.getElementById("results");
    const empty = document.getElementById("empty-state");
    if (!grid) return;
    if (!payload.results || !payload.results.length) {
      grid.innerHTML = "";
      if (empty) empty.hidden = false;
      return;
    }
    if (empty) empty.hidden = true;
    grid.innerHTML = payload.results
      .map(
        (result) => `
        <article class="card">
          <button class="card-media" type="button" data-id="${result.id}"
                  aria-label="Open details for ${esc(result.image_id)}">
            <img loading="lazy" src="${esc(result.thumbnail || "/api/thumbnail/" + result.id)}"
                 alt="${esc(captionFor(result))}">
            <span class="rank-badge">#${result.rank}</span>
          </button>
          <div class="card-body">
            <div class="card-title">${esc(result.title || result.image_id)}</div>
            <p class="card-caption">${esc(captionFor(result))}</p>
            ${componentBars(result.score_components || {})}
            <div class="card-score">
              <span>${esc(SCORE_LABEL[payload.strategy] || "Score")}</span>
              <strong>${fmt(result.score, 3)}</strong>
            </div>
          </div>
        </article>`
      )
      .join("");
  }

  async function runSearch(query, k) {
    const button = document.getElementById("search-button");
    const trimmed = query.trim();
    if (!trimmed) return;
    lastQuery = trimmed;
    if (button) button.disabled = true;
    status("Searching…");
    try {
      const url = `/api/strategies/${encodeURIComponent(currentStrategy)}/search` +
        `?q=${encodeURIComponent(trimmed)}&k=${encodeURIComponent(k)}`;
      const payload = await api(url);
      renderResults(payload);
      const expanded = (payload.expanded_queries || []).filter((item) => item !== trimmed);
      const extra = expanded.length ? ` · expanded: ${expanded.join(", ")}` : "";
      status(`${payload.count} results · ${fmt(payload.latency_ms, 1)} ms${extra}`);
      setUrlState(trimmed, k);
    } catch (error) {
      status(error.message, true);
      const grid = document.getElementById("results");
      if (grid) grid.innerHTML = "";
    } finally {
      if (button) button.disabled = false;
    }
  }

  function detailMarkup(data) {
    const scenes = (data.img_captions || [])
      .map((text) => `<li>${esc(text)}</li>`)
      .join("");
    const meanings = (data.meme_captions || [])
      .map((text) => `<li>${esc(text)}</li>`)
      .join("");
    const metaphors = (data.metaphors || [])
      .map(
        (item) => `<li><span><strong>${esc(item.metaphor)}</strong></span>
          <span class="arrow">→ ${esc(item.meaning)}</span></li>`
      )
      .join("");
    const vlm = data.vlm
      ? `<section class="annotation-block">
           <h3>VLM description <small>${esc(data.vlm.model || "")}</small></h3>
           <ul>
             ${data.vlm.scene ? `<li>${esc(data.vlm.scene)}</li>` : ""}
             ${data.vlm.ocr_text ? `<li><strong>OCR:</strong> ${esc(data.vlm.ocr_text)}</li>` : ""}
             ${data.vlm.meaning ? `<li><strong>Meaning:</strong> ${esc(data.vlm.meaning)}</li>` : ""}
           </ul>
         </section>`
      : "";
    return `
      <div class="detail-hero"><img src="${esc(data.url)}" alt="${esc(data.title || data.image_id)}"></div>
      <div class="detail-content">
        <div class="detail-title">
          <h2>${esc(data.title || data.image_id)}</h2>
          <span class="image-id">${esc(data.image_id)} · #${data.id}</span>
        </div>
        <div class="detail-meta">
          <span class="pill">${esc(data.split || "unknown split")}</span>
          <span class="pill">${esc(data.category || "meme")}</span>
          ${data.source_url
            ? `<a class="source-link" href="${esc(data.source_url)}" target="_blank" rel="noreferrer">original post ↗</a>`
            : ""}
        </div>
        <div class="detail-columns">
          <section class="annotation-block"><h3>Scene</h3><ul>${scenes || "<li>—</li>"}</ul></section>
          <section class="annotation-block"><h3>Meaning</h3><ul>${meanings || "<li>—</li>"}</ul></section>
        </div>
        ${metaphors ? `<section><h3>Metaphors</h3><ul class="metaphor-list">${metaphors}</ul></section>` : ""}
        ${vlm}
        <section>
          <h3>Similar images <small>via ${esc(strategyLabel(dialogStrategy))}</small></h3>
          <div class="similar-strip" id="similar-strip"><span class="text-link">loading…</span></div>
        </section>
      </div>`;
  }

  async function loadSimilar(id, strategyKey) {
    const strip = document.getElementById("similar-strip");
    if (!strip) return;
    try {
      const payload = await api(
        `/api/similar/${id}?strategy=${encodeURIComponent(strategyKey)}&k=8`
      );
      if (!payload.results || !payload.results.length) {
        strip.innerHTML = '<span class="text-link">No neighbours available.</span>';
        return;
      }
      strip.innerHTML = payload.results
        .slice(0, 8)
        .map(
          (item) => `<button type="button" data-similar-id="${item.id}"
                        title="${esc(item.title || item.image_id)}">
                       <img loading="lazy" src="${esc(item.thumbnail)}" alt="">
                     </button>`
        )
        .join("");
    } catch (error) {
      strip.innerHTML = `<span class="text-link">${esc(error.message)}</span>`;
    }
  }

  async function openDetail(id, strategyKey) {
    const dialog = document.getElementById("detail-dialog");
    if (!dialog) return;
    const body = document.getElementById("detail-body");
    if (!body) return;
    dialogStrategy = strategyKey || currentStrategy;
    body.innerHTML = '<div class="detail-content"><p class="text-link">Loading…</p></div>';
    if (!dialog.open) dialog.showModal();
    try {
      const data = await api(`/api/detail/${id}`);
      body.innerHTML = detailMarkup(data);
      loadSimilar(id, dialogStrategy);
    } catch (error) {
      body.innerHTML =
        `<div class="detail-content"><div class="notice error">${esc(error.message)}</div></div>`;
    }
  }

  window.TSR.openDetail = openDetail;

  function initDialog() {
    const dialog = document.getElementById("detail-dialog");
    if (!dialog) return;
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close();
      const close = event.target.closest("[data-close]");
      if (close) dialog.close();
      const similar = event.target.closest("[data-similar-id]");
      if (similar) openDetail(Number(similar.dataset.similarId), dialogStrategy);
    });
    dialog.addEventListener("close", () => {
      const body = document.getElementById("detail-body");
      if (body) body.innerHTML = "";
    });
  }

  async function initSearch() {
    const config = STRATEGY_PAGE[window.location.pathname] || STRATEGY_PAGE["/clip"];
    currentStrategy = config.key;
    document.getElementById("strategy-kicker").textContent = config.kicker;
    document.getElementById("strategy-title").textContent = config.title;
    document.getElementById("strategy-description").textContent = config.description;
    document.getElementById("footer-strategy").textContent =
      `Meme-Cap · ${strategyLabel(currentStrategy)} · research use only`;
    document.title = `${config.title} — Meme Retrieval`;
    document.querySelectorAll("[data-nav]").forEach((link) => {
      if (link.dataset.nav === currentStrategy) link.setAttribute("aria-current", "page");
    });

    const input = document.getElementById("query");
    const range = document.getElementById("top-k");
    const rangeOut = document.getElementById("top-k-value");
    const form = document.getElementById("search-form");

    range.addEventListener("input", () => {
      rangeOut.textContent = range.value;
    });

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      runSearch(input.value, Number(range.value));
    });

    document.querySelectorAll("[data-example]").forEach((button) => {
      button.addEventListener("click", () => {
        input.value = button.dataset.example;
        runSearch(input.value, Number(range.value));
      });
    });

    document.getElementById("results").addEventListener("click", (event) => {
      const target = event.target.closest("[data-id]");
      if (target) openDetail(Number(target.dataset.id), currentStrategy);
    });

    const state = queryFromUrl();
    if (state.k) {
      range.value = String(Math.min(Number(range.max), Math.max(Number(range.min), state.k)));
      rangeOut.textContent = range.value;
    }
    if (state.q) {
      input.value = state.q;
      runSearch(state.q, Number(range.value));
    }

    try {
      const statuses = await api("/api/strategies");
      const item = statuses.find((entry) => entry.key === currentStrategy);
      const pill = document.getElementById("strategy-readiness");
      if (pill) {
        pill.classList.remove("ready", "unavailable");
        pill.classList.add(item && item.ready ? "ready" : "unavailable");
        pill.textContent = item && item.ready ? "ready" : "unavailable";
        if (item && item.error) pill.title = item.error;
      }
      if (item && !item.ready) {
        status(item.error || "This strategy is not built yet.", true);
        document.getElementById("search-button").disabled = true;
      }
    } catch (error) {
      status(error.message, true);
    }
  }

  // Every page with a detail dialog needs its handlers; home has none and returns early.
  initDialog();
  if (PAGE === "home") initHome();
  if (PAGE === "search") initSearch();
})();
