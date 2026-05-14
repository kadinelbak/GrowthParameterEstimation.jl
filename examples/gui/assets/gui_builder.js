(function () {
  const staticSnippetMap = {
    "builder-key-N": "N",
    "builder-key-S": "S",
    "builder-key-R": "R",
    "builder-key-D": "D",
    "builder-key-A": "A",
    "builder-key-E": "E",
    "builder-key-t": "t",
    "builder-key-plus": " + ",
    "builder-key-minus": " - ",
    "builder-key-times": "*",
    "builder-key-div": "/",
    "builder-key-pow": "^",
    "builder-key-lpar": "(",
    "builder-key-rpar": ")",
    "builder-key-log": "log()",
    "builder-key-exp": "exp()",
  };

  function parseCSVInput(id, fallback) {
    const el = document.getElementById(id);
    if (!el || typeof el.value !== "string") {
      return fallback.slice();
    }
    const parts = el.value
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    return parts.length > 0 ? parts : fallback.slice();
  }

  function parseConstantNames() {
    const el = document.getElementById("builder-constants");
    if (!el || typeof el.value !== "string") {
      return [];
    }
    return el.value
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.includes("="))
      .map((s) => s.split("=")[0].trim())
      .filter((s) => s.length > 0);
  }

  function ensureConstants(assignments) {
    const el = document.getElementById("builder-constants");
    if (!el || typeof el.value !== "string") {
      return;
    }
    const existing = el.value
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    const existingNames = new Set(
      existing.filter((s) => s.includes("=")).map((s) => s.split("=")[0].trim())
    );
    let changed = false;
    for (const assignment of assignments) {
      const name = assignment.split("=")[0].trim();
      if (!existingNames.has(name)) {
        existing.push(assignment);
        existingNames.add(name);
        changed = true;
      }
    }
    if (!changed) {
      return;
    }
    const nextValue = existing.join(", ");
    const proto = Object.getPrototypeOf(el);
    const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
    if (setter) {
      setter.call(el, nextValue);
    } else {
      el.value = nextValue;
    }
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function dynamicSnippet(buttonId) {
    if (staticSnippetMap[buttonId]) {
      return staticSnippetMap[buttonId];
    }

    const states = parseCSVInput("builder-state-names", ["N"]);
    const params = parseCSVInput("builder-param-names", ["r", "K"]);
    const constants = parseConstantNames();
    const symbols = params.concat(constants);

    const x = states[0] || "N";
    const y = states[1] || "R";
    const r = params[0] || "r";
    const k = params.find((p) => /^k$/i.test(p)) || params[1] || "K";
    const alpha = params.find((p) => /alpha/i.test(p)) || "alpha";
    const conv = params.find((p) => /^k/i.test(p)) || "kSR";

    if (buttonId === "builder-key-logistic") {
      return `${r}*${x}*(1 - ${x}/${k})`;
    }
    if (buttonId === "builder-key-hill") {
      let hill = symbols.find((p) => /hill/i.test(p)) || "hill";
      let ic50 = symbols.find((p) => /(ic50|ec50)/i.test(p)) || "ic50";
      if (!symbols.some((p) => p === hill) || !symbols.some((p) => p === ic50)) {
        ensureConstants([`${hill}=1.0`, `${ic50}=1.0`]);
      }
      return `(1 - E^${hill}/(${ic50}^${hill} + E^${hill}))`;
    }
    if (buttonId === "builder-key-competition") {
      return `${r}*${x}*(1 - (${x} + ${alpha}*${y})/${k})`;
    }
    if (buttonId === "builder-key-conversion") {
      return `${conv}*${x}`;
    }

    return null;
  }

  function getEquationTextarea() {
    return document.getElementById("builder-equations");
  }

  function setReactTextareaValue(textarea, newValue) {
    const prototype = Object.getPrototypeOf(textarea);
    const valueSetter = Object.getOwnPropertyDescriptor(prototype, "value")?.set;
    if (valueSetter) {
      valueSetter.call(textarea, newValue);
    } else {
      textarea.value = newValue;
    }
  }

  function insertIntoBuilder(snippet) {
    const target = getEquationTextarea();
    if (!target || typeof target.value !== "string") {
      return;
    }

    const start = typeof target.selectionStart === "number" ? target.selectionStart : target.value.length;
    const end = typeof target.selectionEnd === "number" ? target.selectionEnd : target.value.length;
    const value = target.value || "";
    const nextValue = value.slice(0, start) + snippet + value.slice(end);

    setReactTextareaValue(target, nextValue);

    const nextPos = start + snippet.length;
    if (typeof target.setSelectionRange === "function") {
      target.setSelectionRange(nextPos, nextPos);
    }
    target.focus();
    target.dispatchEvent(new Event("input", { bubbles: true }));
    target.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function getSnippetFromButton(button) {
    if (!button) {
      return null;
    }

    if (button.id) {
      const byId = dynamicSnippet(button.id);
      if (byId) {
        return byId;
      }
    }

    const label = (button.textContent || "").trim().toLowerCase();
    if (label === "growth block") return dynamicSnippet("builder-key-logistic");
    if (label === "hill block") return dynamicSnippet("builder-key-hill");
    if (label === "competition") return dynamicSnippet("builder-key-competition");
    if (label === "conversion") return dynamicSnippet("builder-key-conversion");
    return null;
  }

  document.addEventListener("click", function (event) {
    const button = event.target.closest("button");
    if (!button) {
      return;
    }

    const snippet = getSnippetFromButton(button);
    if (!snippet) {
      return;
    }

    event.preventDefault();
    insertIntoBuilder(snippet);
  });
})();
