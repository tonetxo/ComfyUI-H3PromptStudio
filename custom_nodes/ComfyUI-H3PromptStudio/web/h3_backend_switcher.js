import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const OLLAMA_URL = "http://127.0.0.1:11434";
const LLAMACPP_URL = "http://127.0.0.1:8080";

const NODE_NAMES = [
  "H3_PromptStudio_Unified",
  "H3_VisionAnalyzer",
  "H3_MotionDirector",
  "H3_PromptWriter",
  "H3_DirectorMode",
];

function findWidget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function setWidgetVisibility(widget, hidden) {
  if (!widget) return;
  widget.advanced = hidden;
  widget._h3ps_hidden = hidden;
  if (hidden) {
    if (!widget._origComputeSize) widget._origComputeSize = widget.computeSize;
    widget.computeSize = () => [0, -4];
  } else if (widget._origComputeSize) {
    widget.computeSize = widget._origComputeSize;
  }
  if (widget.callback) widget.callback(widget.value);
}

function installBackendSwitcher(nodeType) {
  const origOnNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const result = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
    const backendWidget = findWidget(this, "backend");
    const urlWidget = findWidget(this, "ollama_url");
    if (!backendWidget) return result;

    // Collect model pairs: foo / foo_llamacpp
    const pairs = [];
    const seen = new Set();
    for (const w of this.widgets || []) {
      if (w.name.endsWith("_llamacpp")) {
        const baseName = w.name.slice(0, -"_llamacpp".length);
        const base = findWidget(this, baseName);
        pairs.push({ baseName, base, cpp: w });
        seen.add(w.name);
        if (base) seen.add(baseName);
      }
    }

    const sync = () => {
      const backend = backendWidget.value;
      const url = backend === "llamacpp" ? LLAMACPP_URL : OLLAMA_URL;
      if (urlWidget && urlWidget.value !== url) {
        urlWidget.value = url;
        if (urlWidget.callback) urlWidget.callback(url);
      }
      for (const { base, cpp } of pairs) {
        if (!base || !cpp) continue;
        if (backend === "llamacpp") {
          setWidgetVisibility(base, true);
          setWidgetVisibility(cpp, false);
        } else {
          setWidgetVisibility(base, false);
          setWidgetVisibility(cpp, true);
        }
      }
      this.setDirtyCanvas?.(true, true);
      app.canvas?.setDirty?.(true, true);
      app.graph?.setDirtyCanvas?.(true, true);
    };

    setTimeout(sync, 0);

    const origConfigure = this.configure;
    this.configure = function (info) {
      const out = origConfigure ? origConfigure.apply(this, arguments) : undefined;
      setTimeout(sync, 0);
      return out;
    };

    const origCallback = backendWidget.callback;
    backendWidget.callback = (value) => {
      if (value !== undefined) backendWidget.value = value;
      sync();
      if (origCallback) origCallback(value);
    };

    return result;
  };
}

// Empty custom_* fields fall back to the preset text on the Python side. To make that
// text visible (placeholders are not rendered by the Vue node renderer), fields are
// filled with it and follow preset changes as long as the user has not edited them:
// a field still empty or still holding the previous preset's exact text is replaced,
// anything else is a user override and is kept.
const PRESET_FIELDS = {
  custom_style: "style",
  custom_camera: "camera",
  custom_motion: "motion",
  custom_audio: "audio",
  custom_negative: "negative",
};

// Fallbacks used by build_writer_prompt that do not depend on the preset.
const STATIC_FIELDS = {
  custom_dialogue: "(none unless explicitly requested)",
  custom_constraints: "(preserve identity, wardrobe, props, geometry and lighting logic)",
};

let presetsPromise = null;

function loadPresets() {
  if (!presetsPromise) {
    presetsPromise = api
      .fetchApi("/h3promptstudio/presets")
      .then((resp) => (resp.ok ? resp.json() : Promise.reject(new Error(`HTTP ${resp.status}`))))
      .catch((e) => {
        console.warn("[H3PromptStudio] could not load presets:", e);
        presetsPromise = null;
        return {};
      });
  }
  return presetsPromise;
}

function setWidgetValue(widget, value) {
  if (!widget || widget.value === value) return;
  widget.value = value;
  if (widget.callback) widget.callback(value);
}

function installPresetFill(nodeType) {
  const origOnNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const result = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
    const presetWidget = findWidget(this, "preset");
    if (!presetWidget) return result;

    // Moves untouched fields from the oldPreset text to the newPreset text.
    const apply = async (oldPreset, newPreset) => {
      const presets = await loadPresets();
      const oldData = presets[oldPreset] || {};
      const newData = presets[newPreset];
      if (!newData) return;
      for (const [field, key] of Object.entries(PRESET_FIELDS)) {
        const widget = findWidget(this, field);
        if (!widget) continue;
        const current = (widget.value || "").trim();
        if (current === "" || current === (oldData[key] || "").trim()) {
          setWidgetValue(widget, newData[key] || "");
        }
      }
      for (const [field, text] of Object.entries(STATIC_FIELDS)) {
        const widget = findWidget(this, field);
        if (widget && !(widget.value || "").trim()) setWidgetValue(widget, text);
      }
      this.setDirtyCanvas?.(true, true);
    };

    // Runs after creation and after a saved workflow is loaded; only empty fields change.
    const init = () => {
      this._h3psPreset = presetWidget.value;
      apply(null, presetWidget.value);
    };
    setTimeout(init, 0);

    const origConfigure = this.configure;
    this.configure = function (info) {
      const out = origConfigure ? origConfigure.apply(this, arguments) : undefined;
      setTimeout(init, 0);
      return out;
    };

    const origCallback = presetWidget.callback;
    const self = this;
    presetWidget.callback = function (value) {
      const out = origCallback ? origCallback.apply(this, arguments) : undefined;
      const oldPreset = self._h3psPreset;
      const newPreset = value !== undefined ? value : presetWidget.value;
      self._h3psPreset = newPreset;
      if (oldPreset !== newPreset) apply(oldPreset, newPreset);
      return out;
    };

    return result;
  };
}

async function rescanLlamacppModels() {
  try {
    const resp = await fetch("/h3promptstudio/rescan_llamacpp", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    const data = await resp.json();
    if (data.ok) {
      window.alert?.("llama.cpp rescan OK:\n" + data.message);
    } else {
      window.alert?.("llama.cpp rescan failed:\n" + (data.message || "unknown error"));
    }
  } catch (e) {
    window.alert?.("llama.cpp rescan error: " + e.message);
  }
}

function installRescanButton(nodeType) {
  const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
  nodeType.prototype.getExtraMenuOptions = function (_, options) {
    const existing = origGetExtraMenuOptions ? origGetExtraMenuOptions.apply(this, arguments) : undefined;
    options.push({
      content: "Rescan llama.cpp models",
      callback: rescanLlamacppModels,
    });
    return existing;
  };
}

app.registerExtension({
  name: "H3PromptStudio.H3Nodes.BackendSwitcher.v1",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (NODE_NAMES.includes(nodeData?.name)) {
      installBackendSwitcher(nodeType);
      installRescanButton(nodeType);
      installPresetFill(nodeType);
    }
  },
});
