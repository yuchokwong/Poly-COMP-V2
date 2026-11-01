/* Shared helpers for the Meme Retrieval UI. */
(function () {
  "use strict";

  const STRATEGY_LABEL = {
    clip: "CLIP only",
    clip_nlp: "CLIP + NLP",
    vlm: "VLM descriptions",
  };

  const COMPONENT_LABEL = {
    clip: "CLIP",
    doc: "Dense",
    bm25: "BM25",
    rerank: "Rerank",
    fused: "Fused",
  };

  async function api(url, options) {
    const response = await fetch(url, {
      headers: { Accept: "application/json" },
      ...options,
    });
    const contentType = response.headers.get("content-type") || "";
    const body = contentType.includes("application/json")
      ? await response.json()
      : await response.text();
    if (!response.ok) {
      const detail = body && body.detail ? body.detail : body || response.statusText;
      const error = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      error.status = response.status;
      throw error;
    }
    return body;
  }

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function fmt(value, digits) {
    if (value == null || Number.isNaN(Number(value))) return "—";
    const number = Number(value);
    if (Number.isInteger(number) && digits == null) return String(number);
    return number.toFixed(digits == null ? 3 : digits);
  }

  function pct(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return 0;
    return Math.max(0, Math.min(1, number)) * 100;
  }

  function strategyLabel(key) {
    return STRATEGY_LABEL[key] || key;
  }

  function componentLabel(key) {
    return COMPONENT_LABEL[key] || key;
  }

  function queryFromUrl() {
    const params = new URLSearchParams(window.location.search);
    return { q: params.get("q") || "", k: Number(params.get("k")) || null };
  }

  function setUrlState(q, k) {
    const params = new URLSearchParams();
    if (q) params.set("q", q);
    if (k) params.set("k", k);
    const next = `${window.location.pathname}${params.toString() ? "?" + params : ""}`;
    window.history.replaceState(null, "", next);
  }

  function debounce(fn, wait) {
    let timer = null;
    return function (...args) {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => fn.apply(this, args), wait);
    };
  }

  window.TSR = {
    api,
    esc,
    fmt,
    pct,
    strategyLabel,
    componentLabel,
    queryFromUrl,
    setUrlState,
    debounce,
    STRATEGY_LABEL,
    COMPONENT_LABEL,
  };
})();
