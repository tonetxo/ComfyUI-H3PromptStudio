import { app } from "../../scripts/app.js";

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
    }
  },
});
