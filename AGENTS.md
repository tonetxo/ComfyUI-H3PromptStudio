# AGENTS.md

Single-file Gradio app (`h3_gui.py`, ~660 lines) that builds MiniMax H3 video prompts by chaining three Ollama LLM roles (vision → motion → writer). No tests, no lint/typecheck config, not a git repo.

## Run

```bash
./launch_gui.sh            # creates .venv, pip installs requirements.txt, launches
# or directly:
python h3_gui.py --host 127.0.0.1 --port 7860
```

`--auto-port` defaults to **True**: if 7860 is busy the app silently picks the next free port. The printed URL (`http://host:PORT`) is authoritative — don't assume 7860.

## Hard prerequisites

- Ollama reachable at `config.json` → `ollama_url` (default `http://127.0.0.1:11434`). Without it the GUI loads but every action fails.
- No models are hardcoded. The GUI fetches installed models from `GET /api/tags`. Three roles are independent: `vision_model`, `motion_model`, `writer_model` (any can reuse the same model). All three must be set before generation.

## State the GUI mutates (do not hand-edit while running)

- `config.json` — rewritten by `save_config()` on essentially every action; also holds `temperature_*`, `num_ctx`, `keep_alive`, `preset`. Note `keep_alive` in the committed file is `"0m"` (models unload immediately after each call) — different from `DEFAULT_CONFIG`'s `"20m"`.
- `prompt_history.json` — append-only version history, capped at 100 entries.
- `outputs/` — exports: `h3_prompt.txt`, `vision_analysis.json`, `motion_plan.json`, and `outputs/director_shots/SHOT_NN.txt` + `outputs/director_sequence.json` from Director Mode.

## Architecture notes worth knowing

- Pipeline: reference image → Vision model (JSON) → Motion model (JSON plan) → Writer model (final prompt). System prompts for each role are constants near the top of `h3_gui.py` (`VISION_SYSTEM`, `MOTION_SYSTEM`, …).
- In **Director Mode**, the `Motion model` selector is reused as the Director/Planner model. See `DIRECTOR_MODE.md`.
- Workflow modes: `T2VA`, `I2VA`, `FLF2VA`, `L2VA`, `R2V` (`MODES` in `h3_gui.py:29`). Presets: `H3 Cinematic`, `H3 Dialogue`, `H3 Horror`, `H3 Action`, `Wan 2.x`, `LTX Video` (`PRESETS_DATA` holds their filled fields).
- `outputs/` and `outputs/director_shots/` are created lazily via `mkdir(exist_ok=True)`.

## Editing the app

Everything lives in `h3_gui.py`: config load/save, Ollama client, presets, Director Mode, and `build_ui()` → `gr.Blocks`. Entry point is `main()` at the bottom. Dependencies are intentionally minimal (`gradio`, `requests`) — do not add a framework.

## Reference docs in repo

- `README_GUI.md` — full feature/architecture overview.
- `DIRECTOR_MODE.md` — Director Mode flow and output layout.