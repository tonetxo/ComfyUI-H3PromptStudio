#!/usr/bin/env python3
import argparse
import base64
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import gradio as gr
import requests

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
HISTORY_PATH = ROOT / "prompt_history.json"
DEFAULT_CONFIG = {
    "ollama_url": "http://127.0.0.1:11434",
    "vision_model": "",
    "writer_model": "",
    "motion_model": "",
    "temperature_vision": 0.15,
    "temperature_writer": 0.35,
    "temperature_motion": 0.25,
    "keep_alive": "20m",
    "num_ctx": 32768,
    "preset": "H3 Cinematic",
}

MODES = ["T2VA", "I2VA", "FLF2VA", "L2VA", "R2V"]
PRESETS = ["H3 Cinematic", "H3 Dialogue", "H3 Horror", "H3 Action", "Wan 2.x", "LTX Video"]

PRESETS_DATA = {
    "H3 Cinematic": {
        "style": "photorealistic cinematic, natural motion, physically grounded, high-detail production photography",
        "camera": "coherent cinematic camera movement that remains physically plausible and continuous; preserve spatial relationships",
        "motion": "Prioritize continuous visible motion: subtle subject movement, secondary environmental motion, atmospheric motion, and physically reactive props.",
        "audio": "Diegetic stereo environment only; room tone, machinery, footsteps, cloth, wind, water and other sounds caused by visible events. No music unless requested.",
        "negative": "Frozen environment; static particles; rubbery motion; teleportation; unexplained cuts; camera jumps; duplicated people; identity drift; distorted hands; random text; unwanted music/dialogue.",
    },
    "H3 Dialogue": {
        "style": "photorealistic cinematic drama, natural performance, grounded lighting",
        "camera": "controlled cinematic coverage with motivated push-ins or subtle handheld motion; maintain eyelines and spatial continuity",
        "motion": "Small natural gestures, breathing, posture shifts, eye movement, cloth motion and reactive background activity. Keep dialogue physically synchronized with visible speakers.",
        "audio": "Clean diegetic dialogue with clearly separated speakers, natural room tone and subtle environmental sounds. No music unless requested.",
        "negative": "Mismatched speaker identity; lip-sync drift; random dialogue; frozen actors; frozen background; abrupt camera changes; duplicated people; unwanted music.",
    },
    "H3 Horror": {
        "style": "photorealistic cinematic horror, oppressive atmosphere, restrained practical lighting, tactile textures",
        "camera": "slow creeping push-in or restrained handheld movement with deliberate framing; no gratuitous camera shake",
        "motion": "Emphasize ominous secondary motion: drifting smoke, trembling fixtures, flickering practicals, dust, condensation, fabric, distant shadows and reactive objects. Motion should build tension rather than become chaotic.",
        "audio": "Claustrophobic diegetic ambience, low machinery hum, distant impacts, breath, cloth, creaks, steam, subtle environmental reverb. No score.",
        "negative": "Static scene; excessive shake; random jump scares; supernatural objects unless requested; rubbery motion; lighting popping; unwanted music; unwanted dialogue.",
    },
    "H3 Action": {
        "style": "photorealistic high-intensity cinematic action, physically coherent impacts, grounded materials and inertia",
        "camera": "motivated tracking, handheld or dolly movement that follows the action without teleporting; preserve subject readability",
        "motion": "Strong, continuous primary and secondary motion: body momentum, debris, dust, smoke, water, fabric, recoil, vibration and environmental reactions. Every impact should have visible consequences.",
        "audio": "Diegetic action audio tightly tied to visible events: impacts, machinery, footsteps, debris, alarms, pressure, wind, water. No music unless requested.",
        "negative": "Weightless movement; instant position changes; impossible inertia; frozen debris; frozen atmosphere; camera teleportation; duplicated bodies; random explosions; unwanted music/dialogue.",
    },
    "Wan 2.x": {
        "style": "photorealistic generative video, strong temporal consistency, clear subject silhouettes, robust motion readability",
        "camera": "simple controlled camera movement, avoid unnecessary complexity; preserve framing and geometry",
        "motion": "Prioritize a few major motions plus simple secondary motion. Keep motion continuous and avoid introducing many simultaneous transformations.",
        "audio": "Use concise diegetic sound cues matching visible events; keep the prompt unambiguous.",
        "negative": "Too many simultaneous actions; camera teleportation; sudden scene changes; frozen background; identity drift; duplicated subjects; inconsistent scale.",
    },
    "LTX Video": {
        "style": "cinematic photorealistic video with explicit temporal progression and natural physical motion",
        "camera": "state the shot type first, then the exact camera movement and its direction; keep the movement continuous",
        "motion": "Write motion as observable events over time, emphasizing fluid secondary motion such as smoke, particles, water, cloth and foliage.",
        "audio": "Describe only audible diegetic events that correspond to visible causes.",
        "negative": "Static objects; vague 'dynamic' wording; abrupt cuts; impossible physics; random subject changes; unwanted audio.",
    },
}

VISION_SYSTEM = """You are a forensic visual analyst for video prompting. Analyze the supplied reference image conservatively. Describe only what is visually supported. Do not invent story events, identities, hidden objects, future motion, or off-screen information. Return valid JSON only with keys: scene, subjects, composition, camera_in_image, environment, lighting, materials, wardrobe, props, atmosphere, visible_text, continuity_anchors, motion_candidates, ambiguities."""

MOTION_SYSTEM = """You are a Motion Director for generative video. Using the reference-image analysis and user intent, convert static visual elements into concrete, observable physical motion. Do not invent major subjects or locations. Avoid vague words such as 'dynamic', 'cinematic', or 'realistic' as motion instructions. Return valid JSON only with: primary_actions, subject_motion, environmental_motion, particle_motion, lighting_motion, camera_motion, physics_reactions, timing_beats, audio_events, motion_constraints. Keep motions physically plausible and temporally coherent."""

WRITER_SYSTEM = """You are an expert multimodal video prompt engineer and cinematic director. Write one production-ready English prompt for the selected workflow, with MiniMax H3 as the primary target when selected. Preserve reference identity/composition where relevant. Make physical motion explicit, assign actions to specific subjects, use coherent temporal progression, describe camera movement, and keep audio diegetic unless requested. Do not invent major objects or characters absent from the request/reference analysis. Avoid vague filler. Output only the final prompt."""


DIRECTOR_SYSTEM = """You are a continuity-focused film director and storyboard planner for generative video. Take a scene description, optional reference-image forensic analysis, visual style, and constraints and design a sequence of consecutive video shots. The total requested duration is divided into individual shots suitable for short video generation. Every shot must be independently usable as a video-generation prompt, but all shots must preserve the continuity bible: character identity, wardrobe, props, location geometry, lighting direction/color, time of day, atmosphere, weather and visual style. Do not invent major characters, locations or objects not supported by the scene/reference. Use motivated shot changes: establish geography before action, maintain screen direction/eyelines, and only change camera position when narratively useful. Each shot should have one clear primary action plus secondary physical motion. Avoid packing unrelated actions into the same short shot. Return valid JSON only with: project_title, continuity_bible, global_style, global_audio, shots. continuity_bible must contain: characters, wardrobe, location, props, lighting, atmosphere, camera_language, continuity_rules. Each shot must contain: shot_id, start_time, end_time, duration, purpose, framing, camera, subject_action, secondary_motion, environment_reaction, lighting, audio, dialogue, transition_note, prompt_notes."""

DIRECTOR_PROMPT_SYSTEM = """You are a cinematic video prompt writer. Convert one storyboard shot plus the continuity bible into ONE production-ready English prompt for MiniMax H3. The prompt must restate the critical continuity anchors needed for this shot, then describe framing, camera movement, explicit subject action, secondary physical motion, environmental reactions, lighting and diegetic audio. Use temporal progression only when helpful and keep each shot's action coherent. Do not invent changes to wardrobe, location, character appearance, lighting or props. Do not add music or dialogue unless explicitly specified. This prompt will be generated independently from neighboring shots, so it must be self-contained while remaining consistent with the continuity bible. Output ONLY the prompt text."""


def load_config() -> Dict[str, Any]:
    data = {}
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {**DEFAULT_CONFIG, **data}


def save_config(cfg: Dict[str, Any]) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def get_ollama_models(url: str) -> list[str]:
    r = requests.get(url.rstrip("/") + "/api/tags", timeout=5)
    r.raise_for_status()
    models = [m.get("name", "").strip() for m in r.json().get("models", [])]
    return sorted([m for m in models if m])


def check_ollama(url: str) -> str:
    try:
        models = get_ollama_models(url)
        return "Ollama OK\n" + ("Models: " + ", ".join(models) if models else "No local models found")
    except Exception as e:
        return f"Ollama unavailable: {e}"


def refresh_ollama_models(url: str, current_vision: str = "", current_writer: str = "", current_motion: str = ""):
    try:
        models = get_ollama_models(url)
        if not models:
            fallback = list(dict.fromkeys([m for m in (current_vision, current_writer, current_motion) if m]))
            return tuple(gr.Dropdown(choices=fallback, value=v or None) for v in (current_vision, current_writer, current_motion)) + ("Ollama connected, but no local models are installed.",)
        vals = []
        for current in (current_vision, current_writer, current_motion):
            vals.append(current if current in models else models[0])
        status = f"Ollama OK — {len(models)} local model(s) found."
        return (gr.Dropdown(choices=models, value=vals[0], allow_custom_value=True),
                gr.Dropdown(choices=models, value=vals[1], allow_custom_value=True),
                gr.Dropdown(choices=models, value=vals[2], allow_custom_value=True), status)
    except Exception as e:
        fallback = list(dict.fromkeys([m for m in (current_vision, current_writer, current_motion) if m]))
        return (gr.Dropdown(choices=fallback, value=current_vision or None, allow_custom_value=True),
                gr.Dropdown(choices=fallback, value=current_writer or None, allow_custom_value=True),
                gr.Dropdown(choices=fallback, value=current_motion or None, allow_custom_value=True),
                f"Ollama unavailable: {e}")


def image_to_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def ollama_model_info(url: str, model: str) -> Dict[str, Any]:
    """Return /api/show information when available. Never fail generation just because metadata is unavailable."""
    if not model:
        return {}
    try:
        r = requests.post(url.rstrip("/") + "/api/show", json={"name": model}, timeout=15)
        if r.ok:
            return r.json()
    except requests.RequestException:
        pass
    return {}


def model_supports_vision(url: str, model: str) -> Optional[bool]:
    info = ollama_model_info(url, model)
    caps = info.get("capabilities")
    if isinstance(caps, list):
        # Ollama exposes capabilities such as vision, completion, tools, etc.
        return "vision" in {str(c).lower() for c in caps}
    # Some Ollama/model combinations don't expose capabilities consistently.
    # Return None rather than falsely rejecting a model known to be multimodal.
    return None


def format_ollama_http_error(response: requests.Response, model: str) -> str:
    try:
        data = response.json()
        detail = data.get("error") if isinstance(data, dict) else None
    except ValueError:
        detail = None
    if not detail:
        detail = response.text.strip() or response.reason
    return f"Ollama HTTP {response.status_code} for model '{model}': {detail}"


def ollama_chat(cfg: Dict[str, Any], model: str, messages: list, temperature: float) -> str:
    if not model:
        raise ValueError("No Ollama model selected.")
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": float(temperature), "num_ctx": int(cfg["num_ctx"])},
        "keep_alive": cfg["keep_alive"],
    }
    try:
        r = requests.post(cfg["ollama_url"].rstrip("/") + "/api/chat", json=payload, timeout=900)
    except requests.RequestException as e:
        raise RuntimeError(f"No se pudo conectar con Ollama ({cfg['ollama_url']}): {e}") from e
    if not r.ok:
        raise RuntimeError(format_ollama_http_error(r, model))
    try:
        data = r.json()
        return data["message"]["content"]
    except (ValueError, KeyError, TypeError) as e:
        raise RuntimeError(f"Respuesta inesperada de Ollama para '{model}': {r.text[:1000]}") from e


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise


def analyze_image(cfg: Dict[str, Any], image_path: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": VISION_SYSTEM},
        {"role": "user", "content": "Analyze this reference image. Return JSON only.", "images": [image_to_b64(image_path)]},
    ]
    return extract_json(ollama_chat(cfg, cfg["vision_model"], messages, float(cfg["temperature_vision"])))



def parse_reference_labels(text: str, paths: list[str]) -> list[dict]:
    """Parse optional lines: filename | ROLE | Label. Unspecified files become GENERAL."""
    entries = []
    mapping = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [x.strip() for x in line.split("|", 2)]
        if len(parts) == 1:
            mapping[Path(parts[0]).name.lower()] = ("GENERAL", parts[0])
        elif len(parts) == 2:
            mapping[Path(parts[0]).name.lower()] = (parts[1].upper() or "GENERAL", parts[0])
        else:
            mapping[Path(parts[0]).name.lower()] = (parts[1].upper() or "GENERAL", parts[2] or parts[0])
    for i, path in enumerate(paths, 1):
        name = Path(path).name
        role, label = mapping.get(name.lower(), ("GENERAL", Path(path).stem.replace("_", " ").replace("-", " ")))
        entries.append({"reference_id": f"REF_{i:02d}", "filename": name, "role": role, "label": label, "path": path})
    return entries


def analyze_reference_library(cfg: Dict[str, Any], paths: list[str], labels: str, temperature: float) -> list[dict]:
    refs = parse_reference_labels(labels, paths)
    if not refs:
        return []
    model = cfg.get("vision_model", "")
    if not model:
        raise ValueError("Has cargado referencias: selecciona un Vision model de Ollama.")
    vision_cap = model_supports_vision(cfg["ollama_url"], model)
    if vision_cap is False:
        raise ValueError(f"El modelo '{model}' no declara capacidad de Vision en Ollama.")
    out = []
    for ref in refs:
        prompt = f"Analyze reference {ref['reference_id']} ({ref['role']} — {ref['label']}) for continuity in a multi-shot video project. Return valid JSON only. Focus on identity, appearance, geometry, materials, lighting and persistent visual anchors; do not invent off-screen facts."
        raw = ollama_chat(cfg, model, [
            {"role": "system", "content": VISION_SYSTEM},
            {"role": "user", "content": prompt, "images": [image_to_b64(ref["path"])]},
        ], temperature)
        analysis = extract_json(raw)
        out.append({"reference_id": ref["reference_id"], "filename": ref["filename"], "role": ref["role"], "label": ref["label"], "analysis": analysis})
    return out

def motion_director(cfg: Dict[str, Any], analysis: Dict[str, Any], scene: str, mode: str, duration: str, camera: str) -> Dict[str, Any]:
    text = f"""REFERENCE ANALYSIS:\n{json.dumps(analysis, indent=2, ensure_ascii=False)}\n\nUSER INTENT:\n{scene or '(none)'}\n\nWORKFLOW: {mode}\nDURATION: {duration}\nCAMERA REQUEST: {camera or '(choose based on reference)'}\n\nTranslate the still image into visible, physically plausible motion. Include a compact temporal progression and secondary motion."""
    return extract_json(ollama_chat(cfg, cfg["motion_model"], [{"role": "system", "content": MOTION_SYSTEM}, {"role": "user", "content": text}], float(cfg["temperature_motion"])))


def build_writer_prompt(scene: str, analysis: Optional[Dict[str, Any]], motion_plan: Optional[Dict[str, Any]], mode: str, duration: str,
                        aspect: str, style: str, camera: str, motion: str, audio: str, dialogue: str,
                        constraints: str, negative: str, preset: str) -> str:
    preset_data = PRESETS_DATA[preset]
    analysis_text = json.dumps(analysis, indent=2, ensure_ascii=False) if analysis else "NO REFERENCE IMAGE"
    motion_text = json.dumps(motion_plan, indent=2, ensure_ascii=False) if motion_plan else "NO MOTION DIRECTOR PLAN"
    return f"""Create one final production-ready prompt. PRIMARY TARGET: MiniMax H3. WORKFLOW: {mode}. DURATION: {duration}. ASPECT RATIO: {aspect}. PRESET: {preset}.\n\nUSER SCENE / INTENT:\n{scene or '(none)'}\n\nVISUAL REFERENCE ANALYSIS:\n{analysis_text}\n\nMOTION DIRECTOR PLAN:\n{motion_text}\n\nSTYLE:\n{style or preset_data['style']}\n\nCAMERA:\n{camera or preset_data['camera']}\n\nMOTION PRIORITIES:\n{motion or preset_data['motion']}\n\nAUDIO:\n{audio or preset_data['audio']}\n\nDIALOGUE:\n{dialogue or '(none unless explicitly requested)'}\n\nCONTINUITY / CONSTRAINTS:\n{constraints or '(preserve identity, wardrobe, props, geometry and lighting logic)'}\n\nAVOID:\n{negative or preset_data['negative']}\n\nWrite a coherent temporal sequence rather than a keyword list. Make important movements observable and causally connected. Keep the scene spatially consistent. Output only the final English prompt, with no markdown fences and no explanation."""


def generate_prompt(cfg: Dict[str, Any], writer_prompt: str) -> str:
    return ollama_chat(cfg, cfg["writer_model"], [{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": writer_prompt}], float(cfg["temperature_writer"]))


def load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_history(items: list[dict]) -> None:
    HISTORY_PATH.write_text(json.dumps(items[-100:], indent=2, ensure_ascii=False), encoding="utf-8")


def history_choices(items: list[dict]) -> list[str]:
    return [f"{i.get('id','')} — {i.get('title','Untitled')}" for i in reversed(items)]


def add_history(title: str, payload: dict) -> tuple[list[dict], str]:
    items = load_history()
    item = {"id": time.strftime("%Y%m%d-%H%M%S"), "title": title.strip() or "Untitled", "created": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
    items.append(item)
    save_history(items)
    return items, item["id"]


def get_history_item(choice: str) -> Optional[dict]:
    if not choice:
        return None
    hid = choice.split(" — ", 1)[0]
    return next((i for i in load_history() if i.get("id") == hid), None)


def do_generate(image, scene, mode, duration, aspect, preset, style, camera, motion, audio, dialogue,
                constraints, negative, motion_enabled, ollama_url, vision_model, writer_model, motion_model,
                temperature_writer, temperature_vision, temperature_motion, num_ctx, keep_alive, title):
    cfg = load_config()
    cfg.update({"ollama_url": ollama_url.strip() or cfg["ollama_url"], "vision_model": vision_model.strip(),
                "writer_model": writer_model.strip(), "motion_model": motion_model.strip(),
                "temperature_writer": float(temperature_writer), "temperature_vision": float(temperature_vision),
                "temperature_motion": float(temperature_motion), "num_ctx": int(num_ctx), "keep_alive": keep_alive,
                "preset": preset})
    save_config(cfg)
    if not cfg["writer_model"]:
        raise gr.Error("Selecciona un Writer model de Ollama.")
    analysis = analyze_image(cfg, image) if image else None
    motion_plan = None
    if motion_enabled and analysis:
        if not cfg["motion_model"]:
            raise gr.Error("Motion Director está activado: selecciona un Motion model de Ollama.")
        motion_plan = motion_director(cfg, analysis, scene, mode, duration, camera)
    wp = build_writer_prompt(scene, analysis, motion_plan, mode, duration, aspect, style, camera, motion, audio, dialogue, constraints, negative, preset)
    final = generate_prompt(cfg, wp)
    record = {"prompt": final, "scene": scene, "analysis": analysis, "motion_plan": motion_plan, "mode": mode,
              "duration": duration, "aspect": aspect, "preset": preset, "style": style, "camera": camera,
              "motion": motion, "audio": audio, "dialogue": dialogue, "constraints": constraints, "negative": negative,
              "vision_model": vision_model, "writer_model": writer_model, "motion_model": motion_model}
    items, hid = add_history(title, record)
    return (json.dumps(analysis, indent=2, ensure_ascii=False) if analysis else "No reference image.",
            json.dumps(motion_plan, indent=2, ensure_ascii=False) if motion_plan else "Motion Director disabled or no reference image.",
            final, check_ollama(cfg["ollama_url"]), gr.update(choices=history_choices(items), value=f"{hid} — {title.strip() or 'Untitled'}"))


def restore_history(choice):
    item = get_history_item(choice)
    if not item:
        return ("", "", "", "I2VA", "10s", "16:9", "H3 Cinematic", "", "", "", "", "", "", False, "", "")
    return (item.get("scene", ""), item.get("prompt", ""), json.dumps(item.get("analysis"), indent=2, ensure_ascii=False) if item.get("analysis") else "No reference image.",
            item.get("mode", "I2VA"), item.get("duration", "10s"), item.get("aspect", "16:9"), item.get("preset", "H3 Cinematic"),
            item.get("style", ""), item.get("camera", ""), item.get("motion", ""), item.get("audio", ""), item.get("dialogue", ""),
            item.get("constraints", ""), False, item.get("motion_plan") and json.dumps(item.get("motion_plan"), indent=2, ensure_ascii=False) or "", item.get("title", ""))


def compare_history(a, b):
    ia, ib = get_history_item(a), get_history_item(b)
    if not ia or not ib:
        return "Selecciona dos versiones para comparar."
    fields = ["preset", "mode", "duration", "aspect", "scene", "camera", "motion", "audio", "dialogue", "prompt"]
    chunks = []
    for f in fields:
        av, bv = str(ia.get(f, "")), str(ib.get(f, ""))
        if av != bv:
            chunks.append(f"### {f}\n**A:**\n```text\n{av}\n```\n**B:**\n```text\n{bv}\n```")
    return "\n\n".join(chunks) if chunks else "Las dos versiones son idénticas en los campos comparados."


def save_text(kind: str, text: str) -> str:
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    path = out / ("h3_prompt.txt" if kind == "prompt" else "vision_analysis.json" if kind == "analysis" else "motion_plan.json")
    if kind in {"analysis", "motion"}:
        try:
            text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except Exception:
            pass
    path.write_text(text or "", encoding="utf-8")
    return str(path)


def apply_preset(preset):
    p = PRESETS_DATA[preset]
    return p["style"], p["camera"], p["motion"], p["audio"], p["negative"]


def extract_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end+1])
        raise


def director_generate(scene, reference_files, reference_labels, total_duration, shot_duration, preset, style, camera, motion, audio, dialogue, constraints,
                       ollama_url, vision_model, director_model, writer_model, tv, td, tw, num_ctx, keep_alive):
    cfg = load_config()
    # Director Mode must use exactly the models selected in the GUI.
    # In particular, reference analysis must never fall back to an old/default Vision model.
    cfg.update({
        "ollama_url": ollama_url.strip() or cfg["ollama_url"],
        "vision_model": (vision_model or "").strip(),
        "motion_model": (director_model or "").strip(),
        "writer_model": (writer_model or "").strip(),
        "temperature_vision": float(tv),
        "temperature_motion": float(td),
        "temperature_writer": float(tw),
        "num_ctx": int(num_ctx),
        "keep_alive": keep_alive,
        "preset": preset,
    })
    save_config(cfg)
    if not scene.strip():
        raise ValueError("Introduce una escena para Director Mode.")
    if reference_files and not vision_model:
        raise ValueError("Has cargado imágenes de referencia: selecciona un Vision model de Ollama.")
    if not director_model or not writer_model:
        raise ValueError("Selecciona Director/Planner y Writer models de Ollama.")
    total = int(total_duration); target = int(shot_duration)
    if not 20 <= total <= 60:
        raise ValueError("La duración total debe estar entre 20 y 60 segundos.")
    count = max(2, round(total / target))
    while count > 2 and total // count < 5:
        count -= 1
    base, rem = divmod(total, count)
    durations = [base + (1 if i < rem else 0) for i in range(count)]
    reference_library = analyze_reference_library(cfg, reference_files or [], reference_labels, tv)
    planner_input = {
        "scene": scene, "reference_library": reference_library, "preset": preset,
        "global_style": style, "camera_preferences": camera, "motion_preferences": motion,
        "global_audio": audio, "dialogue": dialogue, "constraints": constraints,
        "total_duration_seconds": total, "target_shot_duration_seconds": target,
        "shot_count": count, "shot_durations_seconds": durations
    }
    raw_plan = ollama_chat(cfg, director_model, [
        {"role":"system","content":DIRECTOR_SYSTEM},
        {"role":"user","content":json.dumps(planner_input, ensure_ascii=False, indent=2)}
    ], td)
    plan = extract_json(raw_plan)
    shots = plan.get("shots", [])
    if not shots:
        raise ValueError("Director model no devolvió ningún plano.")
    bible = plan.get("continuity_bible", {})
    timeline = 0
    prompts = []
    for i, shot in enumerate(shots, 1):
        dur = int(shot.get("duration") or durations[min(i-1, len(durations)-1)])
        dur = max(5, min(15, dur))
        shot_id = shot.get("shot_id") or f"SHOT_{i:02d}"
        shot["shot_id"] = shot_id; shot["start_time"] = timeline; shot["end_time"] = timeline + dur; shot["duration"] = dur
        payload = {"workflow":"MiniMax H3", "shot":shot, "continuity_bible":bible, "global_style":plan.get("global_style") or style,
                   "global_audio":plan.get("global_audio") or audio, "reference_library":reference_library, "user_constraints":constraints}
        prompt = ollama_chat(cfg, writer_model, [
            {"role":"system","content":DIRECTOR_PROMPT_SYSTEM},
            {"role":"user","content":json.dumps(payload, ensure_ascii=False, indent=2)}
        ], tw).strip()
        prompts.append({"shot_id":shot_id,"start_time":timeline,"end_time":timeline+dur,"duration":dur,"prompt":prompt})
        timeline += dur
    result = {"project_title":plan.get("project_title") or "Director Mode sequence", "total_duration":timeline,
              "shot_count":len(prompts), "continuity_bible":bible, "global_style":plan.get("global_style") or style,
              "global_audio":plan.get("global_audio") or audio, "storyboard":shots, "prompts":prompts}
    result["reference_library"] = reference_library
    return reference_library, result


def director_prompt_text(result):
    if not result: return ""
    lines=[f"PROJECT: {result.get('project_title','Director Mode sequence')}", f"TOTAL: {result.get('total_duration',0)}s · {result.get('shot_count',0)} shots"]
    for p in result.get("prompts",[]):
        lines.append(f"\n===== {p['shot_id']} · {p['start_time']}–{p['end_time']}s · {p['duration']}s =====\n{p['prompt']}")
    return "\n".join(lines)


def save_director(result):
    out=ROOT/"outputs"; out.mkdir(exist_ok=True); shots=out/"director_shots"; shots.mkdir(exist_ok=True)
    package=out/"director_sequence.json"; package.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    ref_manifest=out/"reference_library.json"; ref_manifest.write_text(json.dumps(result.get("reference_library",[]),indent=2,ensure_ascii=False),encoding="utf-8")
    paths=[str(package), str(ref_manifest)]
    for p in result.get("prompts",[]):
        f=shots/f"{p['shot_id']}.txt"; f.write_text(p["prompt"],encoding="utf-8"); paths.append(str(f))
    return "Saved:\n"+"\n".join(paths)


def build_ui():
    cfg = load_config()
    hist = load_history()
    with gr.Blocks(title="MiniMax H3 Prompt Studio") as app:
        gr.Markdown("# MiniMax H3 Prompt Studio\nLocal GUI · Ollama Vision → Motion Director → H3 Prompt Engineer")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="filepath", label="Reference image", sources=["upload", "clipboard"])
                scene = gr.Textbox(label="Scene / intent", lines=5, placeholder="Describe what should happen in the shot…")
                with gr.Row():
                    mode = gr.Dropdown(MODES, value="I2VA", label="Workflow")
                    duration = gr.Dropdown(["5s", "6s", "8s", "10s", "12s", "15s"], value="10s", label="Duration")
                    aspect = gr.Dropdown(["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"], value="16:9", label="Aspect")
                preset = gr.Dropdown(PRESETS, value=cfg.get("preset", "H3 Cinematic"), label="Cinematic preset")
                initial_p = PRESETS_DATA.get(cfg.get("preset", "H3 Cinematic"), PRESETS_DATA["H3 Cinematic"])
                style = gr.Textbox(label="Style", lines=2, value=initial_p["style"])
                camera = gr.Textbox(label="Camera", lines=2, value=initial_p["camera"])
                motion = gr.Textbox(label="Motion priorities", lines=3, value=initial_p["motion"])
                audio = gr.Textbox(label="Audio", lines=2, value=initial_p["audio"])
                dialogue = gr.Textbox(label="Dialogue", lines=2, placeholder="Optional. Identify every speaker explicitly.")
                constraints = gr.Textbox(label="Continuity / constraints", lines=3,
                    value="Preserve the reference composition, character identity, wardrobe, props, lighting and spatial geometry unless the requested action requires change.")
                negative = gr.Textbox(label="Avoid", lines=3, value=initial_p["negative"])
                motion_enabled = gr.Checkbox(value=True, label="Enable Motion Director (requires reference image)")
                title = gr.Textbox(label="History title", value="H3 draft")
                generate = gr.Button("Generate H3 Prompt", variant="primary", size="lg")

                with gr.Accordion("Ollama / advanced", open=True):
                    ollama_url = gr.Textbox(value=cfg["ollama_url"], label="Ollama URL")
                    with gr.Row():
                        vision_model = gr.Dropdown(choices=[], value=cfg.get("vision_model", ""), label="Vision model", allow_custom_value=True)
                        motion_model = gr.Dropdown(choices=[], value=cfg.get("motion_model", ""), label="Motion model", allow_custom_value=True)
                        writer_model = gr.Dropdown(choices=[], value=cfg.get("writer_model", ""), label="Writer model", allow_custom_value=True)
                    with gr.Row():
                        refresh_models = gr.Button("Refresh models")
                        check = gr.Button("Check Ollama")
                    temperature_vision = gr.Slider(0, 1, value=cfg["temperature_vision"], step=0.05, label="Vision temperature")
                    temperature_motion = gr.Slider(0, 1, value=cfg["temperature_motion"], step=0.05, label="Motion temperature")
                    temperature_writer = gr.Slider(0, 1, value=cfg["temperature_writer"], step=0.05, label="Writer temperature")
                    num_ctx = gr.Number(value=cfg["num_ctx"], precision=0, label="Context length")
                    keep_alive = gr.Textbox(value=cfg["keep_alive"], label="Ollama keep_alive")
                    ollama_status = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=1):
                analysis = gr.Code(label="Vision analysis (JSON)", language="json", lines=18)
                motion_plan = gr.Code(label="Motion Director plan (JSON)", language="json", lines=18)
                prompt = gr.Textbox(label="Final prompt", lines=20)
                with gr.Row():
                    save_p = gr.Button("Save prompt")
                    save_a = gr.Button("Save analysis")
                    save_m = gr.Button("Save motion plan")
                saved = gr.Textbox(label="Saved file", interactive=False)

                with gr.Accordion("Prompt history / versions", open=False):
                    history = gr.Dropdown(choices=history_choices(hist), label="Version", allow_custom_value=False)
                    with gr.Row():
                        restore = gr.Button("Restore version")
                        delete_history = gr.Button("Clear history")
                    with gr.Row():
                        compare_a = gr.Dropdown(choices=history_choices(hist), label="Compare A")
                        compare_b = gr.Dropdown(choices=history_choices(hist), label="Compare B")
                    compare = gr.Button("Compare versions")
                    comparison = gr.Markdown()

        with gr.Tab("Director Mode"):
            gr.Markdown("## Director Mode\nPlan a 20–60 second sequence as consecutive short H3 shots with a shared continuity bible.")
            with gr.Row():
                with gr.Column(scale=1):
                    d_scene = gr.Textbox(label="Scene / sequence brief", lines=7)
                    d_refs = gr.File(label="Reference images (optional, multiple)", file_count="multiple", file_types=["image"], type="filepath")
                    d_ref_labels = gr.Textbox(label="Reference roles / labels (optional)", lines=4, placeholder="One per line: captain.png | CHARACTER | Captain\nengine_room.jpg | LOCATION | U-29 engine room\nuniform.png | WARDROBE | Captain uniform")
                    with gr.Row():
                        d_total = gr.Dropdown(["20","24","30","36","40","45","48","50","54","60"], value="30", label="Total duration (s)")
                        d_shot = gr.Dropdown(["5","6","8","10","12","15"], value="8", label="Target shot duration (s)")
                    d_preset = gr.Dropdown(PRESETS, value="H3 Cinematic", label="Cinematic preset")
                    d_style = gr.Textbox(label="Global style", lines=2, value=PRESETS_DATA["H3 Cinematic"]["style"])
                    d_camera = gr.Textbox(label="Camera language", lines=2, value=PRESETS_DATA["H3 Cinematic"]["camera"])
                    d_motion = gr.Textbox(label="Global motion", lines=3, value=PRESETS_DATA["H3 Cinematic"]["motion"])
                    d_audio = gr.Textbox(label="Global audio", lines=2, value=PRESETS_DATA["H3 Cinematic"]["audio"])
                    d_dialogue = gr.Textbox(label="Dialogue / spoken lines", lines=3)
                    d_constraints = gr.Textbox(label="Continuity rules", lines=4, value="Keep character identity, wardrobe, props, location geometry, lighting direction and time of day identical between shots. Maintain screen direction and eyelines unless a motivated transition changes them.")
                    d_generate = gr.Button("Build sequence + prompts", variant="primary", size="lg")
                    d_export = gr.Button("Save Director package")
                    d_saved = gr.Textbox(label="Saved files", interactive=False)
                with gr.Column(scale=1):
                    d_bible = gr.Code(label="Continuity Bible + Reference Library", language="json", lines=22)
                    d_storyboard = gr.Code(label="Storyboard / shot plan", language="json", lines=22)
                    d_prompts = gr.Textbox(label="All H3 shot prompts", lines=24)
                    d_status = gr.Textbox(label="Director status", interactive=False)
            d_shot_selector = gr.Dropdown(choices=[], value=None, label="Select shot", allow_custom_value=False, interactive=False)
            d_selected_prompt = gr.Textbox(label="Selected H3 prompt", lines=14)
            d_selected_meta = gr.Code(label="Selected shot metadata", language="json", lines=8)
            director_state = gr.State({})

        preset.change(apply_preset, inputs=preset, outputs=[style, camera, motion, audio, negative])
        generate.click(do_generate,
            inputs=[image, scene, mode, duration, aspect, preset, style, camera, motion, audio, dialogue, constraints, negative,
                    motion_enabled, ollama_url, vision_model, writer_model, motion_model, temperature_writer, temperature_vision,
                    temperature_motion, num_ctx, keep_alive, title],
            outputs=[analysis, motion_plan, prompt, ollama_status, history])
        def sync_compare_choices(items):
            choices = items if isinstance(items, list) else []
            return gr.update(choices=choices), gr.update(choices=choices)
        generate.click(lambda _unused: sync_compare_choices(history_choices(load_history())), inputs=prompt, outputs=[compare_a, compare_b])
        refresh_models.click(refresh_ollama_models, inputs=[ollama_url, vision_model, writer_model, motion_model], outputs=[vision_model, writer_model, motion_model, ollama_status])
        check.click(check_ollama, inputs=ollama_url, outputs=ollama_status)
        app.load(refresh_ollama_models, inputs=[ollama_url, vision_model, writer_model, motion_model], outputs=[vision_model, writer_model, motion_model, ollama_status])
        save_p.click(lambda x: save_text("prompt", x), inputs=prompt, outputs=saved)
        save_a.click(lambda x: save_text("analysis", x), inputs=analysis, outputs=saved)
        save_m.click(lambda x: save_text("motion", x), inputs=motion_plan, outputs=saved)
        restore.click(restore_history, inputs=history,
            outputs=[scene, prompt, analysis, mode, duration, aspect, preset, style, camera, motion, audio, dialogue, constraints, motion_enabled, motion_plan, title])
        compare.click(compare_history, inputs=[compare_a, compare_b], outputs=comparison)
        def clear_hist():
            save_history([])
            return gr.update(choices=[], value=None), gr.update(choices=[], value=None), gr.update(choices=[], value=None)
        delete_history.click(clear_hist, outputs=[history, compare_a, compare_b])
        d_preset.change(lambda p: apply_preset(p)[:4], inputs=d_preset, outputs=[d_style, d_camera, d_motion, d_audio])
        def run_director(*args):
            ref, result = director_generate(*args)
            choices=[p["shot_id"] for p in result.get("prompts",[])]
            first=result.get("prompts",[])[0] if result.get("prompts") else {}
            meta={k:first.get(k) for k in ["shot_id","start_time","end_time","duration"]} if first else {}
            selector_update = gr.update(
                choices=choices,
                value=choices[0] if choices else None,
                interactive=bool(choices),
            )
            return (json.dumps({"reference_library": ref},indent=2,ensure_ascii=False) if ref else "No reference images.",
                    json.dumps(result.get("storyboard",[]),indent=2,ensure_ascii=False),director_prompt_text(result),
                    f"Director OK — {result.get('shot_count',0)} shots / {result.get('total_duration',0)}s",selector_update,
                    first.get("prompt","") if first else "",json.dumps(meta,indent=2,ensure_ascii=False),result)
        d_generate.click(run_director,
            inputs=[d_scene,d_refs,d_ref_labels,d_total,d_shot,d_preset,d_style,d_camera,d_motion,d_audio,d_dialogue,d_constraints,
                    ollama_url,vision_model,motion_model,writer_model,temperature_vision,temperature_motion,temperature_writer,num_ctx,keep_alive],
            outputs=[d_bible,d_storyboard,d_prompts,d_status,d_shot_selector,d_selected_prompt,d_selected_meta,director_state])
        def show_director_shot(choice,result):
            for p in (result or {}).get("prompts",[]):
                if p.get("shot_id")==choice:
                    meta={k:p.get(k) for k in ["shot_id","start_time","end_time","duration"]}
                    return p.get("prompt","") , json.dumps(meta,indent=2,ensure_ascii=False)
            return "", "{}"
        d_shot_selector.change(show_director_shot,inputs=[d_shot_selector,director_state],outputs=[d_selected_prompt,d_selected_meta])
        d_export.click(save_director,inputs=director_state,outputs=d_saved)

    return app


def find_free_port(host: str, start: int, attempts: int = 20) -> int:
    import socket
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    raise OSError(f"No free port found from {start} to {start + attempts - 1}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--auto-port", action="store_true", default=True, help="Find the next free port when the requested port is busy.")
    args = parser.parse_args()
    app = build_ui()
    port = find_free_port(args.host, args.port) if args.auto_port else args.port
    print(f"MiniMax H3 Prompt Studio · http://{args.host}:{port}")
    app.launch(server_name=args.host, server_port=port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
