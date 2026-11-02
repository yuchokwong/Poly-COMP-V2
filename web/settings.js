/* Runtime settings editor: GET /api/settings, PUT /api/settings. */
(function () {
  "use strict";

  const TSR = window.TSR;
  const { api, esc } = TSR;

  const form = document.getElementById("settings-form");
  const groups = document.getElementById("settings-groups");
  const status = document.getElementById("settings-status");
  const saveState = document.getElementById("save-state");
  const saveButton = document.getElementById("save-button");
  const resetButton = document.getElementById("reset-button");

  let fields = [];
  let initial = {};

  function say(message, tone) {
    status.textContent = message || "";
    status.dataset.tone = tone || "";
  }

  function currentValue(field) {
    const input = document.getElementById(`field-${field.key}`);
    if (!input) return field.value;
    if (field.type === "bool") return input.checked;
    if (field.type === "int") return Number.parseInt(input.value, 10);
    if (field.type === "float") return Number.parseFloat(input.value);
    return input.value;
  }

  function control(field) {
    const id = `field-${field.key}`;
    const disabled = field.editable === false ? " disabled" : "";
    if (field.type === "bool") {
      return `<label class="setting-row">
        <span class="setting-text"><b>${esc(field.label)}</b>
          <small>${esc(field.help || field.key)}</small></span>
        <input id="${id}" type="checkbox"${field.value ? " checked" : ""}${disabled}>
      </label>`;
    }
    const numeric = field.type === "int" || field.type === "float";
    const type = numeric ? "number" : "text";
    const step = numeric && field.step != null ? ` step="${field.step}"` : "";
    const min = numeric && field.min != null ? ` min="${field.min}"` : "";
    const max = numeric && field.max != null ? ` max="${field.max}"` : "";
    return `<label class="setting-row">
      <span class="setting-text"><b>${esc(field.label)}</b>
        <small>${esc(field.help || field.key)}</small></span>
      <input id="${id}" type="${type}" value="${esc(field.value)}"${step}${min}${max}${disabled}>
    </label>`;
  }

  function render() {
    const byGroup = new Map();
    for (const field of fields) {
      if (!byGroup.has(field.group)) byGroup.set(field.group, []);
      byGroup.get(field.group).push(field);
    }
    groups.innerHTML = [...byGroup.entries()]
      .map(
        ([group, items]) => `<section class="settings-group">
          <h2>${esc(group)}</h2>
          ${items.map(control).join("")}
        </section>`,
      )
      .join("");
    initial = Object.fromEntries(
      fields.filter((f) => f.editable !== false).map((f) => [f.key, f.value]),
    );
  }

  function changes() {
    const updates = {};
    for (const field of fields) {
      if (field.editable === false) continue;
      const value = currentValue(field);
      if (value !== initial[field.key]) updates[field.key] = value;
    }
    return updates;
  }

  async function load() {
    try {
      const payload = await api("/api/settings");
      fields = payload.fields || [];
      render();
      saveState.textContent = "loaded";
      saveState.dataset.tone = "ok";
      say(`${fields.filter((f) => f.editable !== false).length} editable settings.`, "");
    } catch (error) {
      say(`Could not load settings: ${error.message}`, "error");
      saveState.textContent = "offline";
      saveState.dataset.tone = "error";
    }
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const updates = changes();
    if (!Object.keys(updates).length) {
      say("Nothing changed.", "");
      return;
    }
    saveButton.disabled = true;
    try {
      const payload = await api("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updates),
      });
      fields = payload.fields || fields;
      render();
      saveState.textContent = "saved";
      saveState.dataset.tone = "ok";
      say(`Saved ${Object.keys(updates).length} setting(s). They apply from the next search.`, "ok");
    } catch (error) {
      say(`Save failed: ${error.message}`, "error");
      saveState.textContent = "unsaved";
      saveState.dataset.tone = "error";
    } finally {
      saveButton.disabled = false;
    }
  });

  resetButton.addEventListener("click", () => {
    for (const field of fields) {
      if (field.editable === false) continue;
      const input = document.getElementById(`field-${field.key}`);
      if (!input) continue;
      if (field.type === "bool") input.checked = Boolean(field.value);
      else input.value = field.value;
    }
    say("Reverted unsaved changes.", "");
    saveState.textContent = "loaded";
    saveState.dataset.tone = "ok";
  });

  load();
})();
