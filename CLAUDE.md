# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Prompt-engineering tooling for MiniMax H3 / LTX-Video / Wan 2.x video generation. It chains local LLM roles — **Vision** (image → JSON analysis) → **Motion Director** (JSON motion plan) → **Writer** (final prompt) — plus a multi-shot **Director Mode** (planner → per-shot prompts sharing a "continuity bible").

The same pipeline exists in three separately maintained copies that have drifted apart:

| File | Role | Notes |
|---|---|---|
| `h3_gui.py` (~2600 lines) | Standalone Gradio app — the most feature-complete copy | Ollama **and** llama.cpp backends, MiniMax enhancer path, R2V references, history, persisted UI state |
| `custom_nodes/ComfyUI-H3PromptStudio/` | ComfyUI node pack (newest node version) | Has llama.cpp backend, `web/h3_backend_switcher.js`, and a `/h3promptstudio/rescan_llamacpp` route in `__init__.py` |
| `nodes.py` + `__init__.py` (repo root) | Older ComfyUI node pack (what `README.md` documents) | Ollama only; no llama.cpp support |

`custom_nodes/ComfyUI-Qwen-Prompt-Rewriter/` is an untracked, unrelated node. System prompts (`VISION_SYSTEM`, `MOTION_SYSTEM`, `WRITER_SYSTEM`, `DIRECTOR_SYSTEM`, `DIRECTOR_PROMPT_SYSTEM`) and `PRESETS_DATA` are duplicated in each copy — when changing pipeline behaviour, decide explicitly which copies to update rather than assuming one change propagates.

## Run

```bash
./launch_gui.sh                              # creates .venv, installs requirements-gui.txt, launches h3_gui.py
.venv/bin/python h3_gui.py --host 127.0.0.1 --port 7860
```

- `--auto-port` is always on (default True, no way to disable from CLI): if the port is busy the next free one is used. Trust the printed URL.
- `requirements.txt` (`requests`, `Pillow`) is for the ComfyUI pack; `requirements-gui.txt` includes it and adds `gradio>=6` for the GUI — keep Gradio out of `requirements.txt` so it is not installed into ComfyUI. `nodes.py` additionally relies on ComfyUI's `torch`/`numpy`.
- No tests, linter or type checker. Quick sanity check after edits: `.venv/bin/python -m py_compile h3_gui.py nodes.py custom_nodes/ComfyUI-H3PromptStudio/nodes.py`.

## LLM backends (h3_gui.py)

`config.json` → `backend` selects `"ollama"` or `"llamacpp"`; the URL is always stored in `ollama_url` even for llama.cpp (currently `http://127.0.0.1:8080`).

- **ollama**: models from `GET /api/tags`; calls via `/api/chat` / `/api/generate`; `keep_alive` and `num_ctx` apply; explicit unload helpers (`ollama_unload_model`, `ollama_unload_all`).
- **llamacpp**: OpenAI-compatible router at `/v1/models` and `/v1/chat/completions` (`llamacpp_chat_completion`), with thinking disabled via `chat_template_kwargs`. `keep_alive` does not apply (server sleeps idle models). Model list = router models + a hardcoded fallback list + `.gguf` files scanned from `/home/tonetxo/gguf-models`. `_resolve_llamacpp_model` maps aliases.
- "Rescan" runs `~/.config/h3ps/rescan_llamacpp_models.py` (regenerates `~/.config/h3ps/llamacpp-models.ini`) and then `systemctl --user restart h3ps-llamacpp.service`. These machine-specific paths are hardcoded in both `h3_gui.py` and the custom node's `__init__.py`.

`ollama_chat()` is the single entry point used by the pipeline; it dispatches to llama.cpp when `backend == "llamacpp"`.

## Writer paths

In `do_generate`, if the writer model name matches `is_minimax_enhancer()` (`minimax`+`enhancer` or `video-prompt-enhancer`), the writer step uses `generate_enhanced_prompt()` with per-mode structured system prompts (`ENHANCER_SYSTEM_*`: T2VA / I2VA / FL2VA / L2VA / REF for R2V). If `is_enhancer_output_valid()` rejects the result, it falls back to the generic `build_writer_prompt()` + `generate_prompt()` path. LLM JSON output goes through `extract_json` / `_try_parse_json` / `_repair_json`, which tolerate malformed model output.

Workflow modes: `T2VA`, `I2VA`, `FLF2VA`, `L2VA`, `R2V`. In R2V, reference tags like `<Image_1>`, `<Video_1>`, `<Audio_1>`, `<Voice_1>` must survive verbatim into the final prompt. In Director Mode the GUI's Motion model selector doubles as the Director/Planner model (see `DIRECTOR_MODE.md`).

## State the GUI writes (don't hand-edit while it runs)

- `config.json` — rewritten by `save_config()` on most actions. Its values (`keep_alive: "0m"`, llama.cpp backend, etc.) differ from `DEFAULT_CONFIG` in `h3_gui.py`.
- `ui_state.json` — persisted form fields for main and Director tabs; uploaded references are copied into `outputs/ui_refs/`.
- `prompt_history.json` — version history, capped at 100 entries (can be large).
- `outputs/` — `h3_prompt.txt`, `vision_analysis.json`, `motion_plan.json`, `reference_library.json`, timestamped copies in `outputs/archive/`, and Director exports in `outputs/director_shots/SHOT_NN.txt` + `outputs/director_sequence.json`.

## Git quirks

`.gitignore` lists `h3_gui.py`, `custom_nodes/`, `AGENTS.md`, `*.json`, etc. as "local only", but `h3_gui.py` and `custom_nodes/ComfyUI-H3PromptStudio/*` are already tracked, so edits to them still show up in `git status`. New files under ignored paths (e.g. new JSON workflows in `workflows/`) will not be picked up without `git add -f`. User-facing error strings in the code are partly in Spanish; keep the surrounding language when editing them.

## Constraints

Keep dependencies minimal (`gradio`, `requests`, `Pillow`) — don't add a framework. `AGENTS.md` mirrors this file for other agents; keep both in sync.
