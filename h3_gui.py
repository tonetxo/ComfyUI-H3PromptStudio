#!/usr/bin/env python3
import argparse
import base64
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import gradio as gr
import requests

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
HISTORY_PATH = ROOT / "prompt_history.json"
UI_STATE_PATH = ROOT / "ui_state.json"
UI_REFS_DIR = ROOT / "outputs" / "ui_refs"
DEFAULT_CONFIG = {
    "backend": "ollama",
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
PRESETS = ["H3 Cinematic", "H3 Dialogue", "H3 Horror", "H3 Action", "Wan 2.x", "LTX Video", "LTX Video FLF"]

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
    "LTX Video FLF": {
        "style": "photorealistic LTX Video first-last frame transition, crisp temporal progression, photorealistic lighting continuity",
        "camera": "specify initial framing from First Frame, then continuous motivated camera move (dolly, pan, tilt, push-in) transitioning toward Last Frame composition",
        "motion": "First-to-Last Frame trajectory: Describe step-by-step physical motion starting from the exact pose/composition of Image 1 (FIRST_FRAME) and smoothly evolving into the pose/composition of Image 2 (LAST_FRAME). Include secondary physical reaction, fabric dynamics, and environmental lighting shifting over time.",
        "audio": "Synchronized diegetic audio matching the physical transition from start to finish.",
        "negative": "Abrupt morphing; teleportation; frozen intermediate frames; identity drift; lighting popping; camera jumps; unwanted music; distorted limbs.",
    },
}

VISION_SYSTEM = """You are a forensic visual analyst for video prompting. Analyze the supplied reference image conservatively. Describe only what is visually supported. Do not invent story events, identities, hidden objects, future motion, or off-screen information. Return valid JSON only with keys: scene, subjects, composition, camera_in_image, environment, lighting, materials, wardrobe, props, atmosphere, visible_text, continuity_anchors, motion_candidates, ambiguities."""

MOTION_SYSTEM = """You are a Motion Director for generative video. Using the reference-image analysis and user intent, convert static visual elements into concrete, observable physical motion. Do not invent major subjects or locations. Avoid vague words such as 'dynamic', 'cinematic', or 'realistic' as motion instructions. Return valid JSON only with: primary_actions, subject_motion, environmental_motion, particle_motion, lighting_motion, camera_motion, physics_reactions, timing_beats, audio_events, motion_constraints. Keep motions physically plausible and temporally coherent."""

WRITER_SYSTEM = """You are an expert multimodal video prompt engineer and cinematic director. Write one production-ready English prompt for the selected workflow, with MiniMax H3 or LTX Video as the primary target when selected. Preserve reference identity/composition where relevant. When reference tags such as <Image_1>, <Image_2>, <Video_1>, <Audio_1>, <Audio_0>, or <Voice_1> are used, strictly preserve them as functional anchors. Make physical motion explicit, assign actions to specific subjects, use coherent temporal progression, describe camera movement, and keep audio diegetic unless requested. Do not invent major objects or characters absent from the request/reference analysis. Avoid vague filler. Output only the final prompt."""


DIRECTOR_SYSTEM = """You are a continuity-focused film director and storyboard planner for generative video. Take a scene description, optional reference-image forensic analysis, visual style, and constraints and design a sequence of consecutive video shots. The total requested duration is divided into individual shots suitable for short video generation. Every shot must be independently usable as a video-generation prompt, but all shots must preserve the continuity bible: character identity, wardrobe, props, location geometry, lighting direction/color, time of day, atmosphere, weather and visual style. Do not invent major characters, locations or objects not supported by the scene/reference. Use motivated shot changes: establish geography before action, maintain screen direction/eyelines, and only change camera position when narratively useful. Each shot should have one clear primary action plus secondary physical motion. Avoid packing unrelated actions into the same short shot. Return valid JSON only with: project_title, continuity_bible, global_style, global_audio, shots. continuity_bible must contain: characters, wardrobe, location, props, lighting, atmosphere, camera_language, continuity_rules. Each shot must contain: shot_id, start_time, end_time, duration, purpose, framing, camera, subject_action, secondary_motion, environment_reaction, lighting, audio, dialogue, transition_note, prompt_notes."""

DIRECTOR_PROMPT_SYSTEM = """You are a cinematic video prompt writer. Convert one storyboard shot plus the continuity bible into ONE production-ready English prompt for MiniMax H3 or LTX Video. The prompt must restate the critical continuity anchors needed for this shot, then describe framing, camera movement, explicit subject action, secondary physical motion, environmental reactions, lighting and diegetic audio. If the workflow is First-Last Frame (FLF2VA / LTX Video FLF), explicitly detail the visible physical trajectory from the initial frame (FIRST_FRAME) to the ending frame (LAST_FRAME). Use temporal progression only when helpful and keep each shot's action coherent. Do not invent changes to wardrobe, location, character appearance, lighting or props. Do not add music or dialogue unless explicitly specified. This prompt will be generated independently from neighboring shots, so it must be self-contained while remaining consistent with the continuity bible. Output ONLY the prompt text."""

# TODO: Add MiniMax Prompt Enhancer contract support to Director Mode. The enhancer expects a single
# full-reference user envelope; Director Mode currently rewrites per-shot prompts via DIRECTOR_PROMPT_SYSTEM.
# A future implementation could build one enhancer envelope per shot from the shot + continuity bible.

# ---------------------------------------------------------------------------
# MiniMax Video Prompt Enhancer 2.6B contract
# System prompts and envelope builder below are derived from the model card at
# https://huggingface.co/geocine/minimax-video-prompt-enhancer-2.6b-gguf
# They are used ONLY when the selected writer model is detected as the enhancer.
# ---------------------------------------------------------------------------

ENHANCER_R2V_TASKS = [
    "reference_generation",
    "video_editing",
    "video_continuation",
    "keyframe_completion",
    "video_editing+audio_reuse",
    "video_continuation+audio_reference",
    "reference_generation+audio_reference",
]

EVOLVE_VOCAB = [
    "cinematic", "dramatic", "atmospheric", "moody", "ethereal", "surreal",
    "hyperrealistic", "photorealistic", "volumetric", "noir", "neon", "golden",
    "misty", "stormy", "serene", "tense", "epic", "intimate", "melancholic",
    "euphoric", "ominous", "dreamlike", "futuristic", "rustic", "decayed",
    "luxurious", "desolate", "lush", "intricate", "minimalist", "dynamic",
    "static", "fluid", "fragmented", "seamless", "chaotic", "ordered", "warm",
    "cold", "vibrant", "muted", "wide shot", "close up", "extreme close up",
    "medium shot", "overhead", "low angle", "dutch angle", "tracking", "handheld",
    "static tripod", "golden hour", "blue hour", "midday", "night", "dusk", "dawn",
    "backlit", "rim light", "soft light", "hard light", "film grain", "lens flare",
    "bokeh", "motion blur", "sharp focus", "shallow depth of field", "deep focus",
    "anamorphic", "35mm", "16mm", "IMAX", "digital", "vintage", "celluloid",
    "orchestral", "electronic", "ambient", "silence", "distant", "nearby",
    "echoing", "muffled", "crisp", "slow motion", "time lapse", "real time",
    "long take", "quick cut", "montage",
]

EVOLVE_SYNONYMS = {
    "big": ["massive", "enormous", "colossal", "immense", "towering"],
    "small": ["tiny", "minuscule", "petite", "compact", "diminutive"],
    "fast": ["rapid", "swift", "quick", "accelerated", "hurried"],
    "slow": ["leisurely", "gradual", "deliberate", "unhurried", "languid"],
    "happy": ["joyful", "elated", "euphoric", "content", "radiant"],
    "sad": ["melancholic", "somber", "mournful", "forlorn", "sorrowful"],
    "angry": ["furious", "irate", "livid", "incensed", "wrathful"],
    "scared": ["terrified", "petrified", "horrified", "alarmed", "panicked"],
    "beautiful": ["gorgeous", "stunning", "breathtaking", "exquisite", "radiant"],
    "ugly": ["grotesque", "unsightly", "repulsive", "hideous", "monstrous"],
    "dark": ["dim", "shadowy", "murky", "tenebrous", "obscure"],
    "light": ["luminous", "radiant", "brilliant", "gleaming", "ethereal"],
    "old": ["ancient", "weathered", "aged", "antique", "timeworn"],
    "new": ["pristine", "modern", "novel", "recent", "fresh"],
    "loud": ["deafening", "thunderous", "cacophonous", "boisterous", "clamorous"],
    "quiet": ["silent", "hushed", "muffled", "subdued", "tranquil"],
    "hot": ["scorching", "blazing", "searing", "sweltering", "torrid"],
    "cold": ["frigid", "freezing", "icy", "glacial", "wintry"],
    "good": ["excellent", "superb", "magnificent", "stellar", "remarkable"],
    "bad": ["dreadful", "abysmal", "atrocious", "deplorable", "lamentable"],
    "run": ["sprint", "dash", "race", "bolt", "charge"],
    "walk": ["stride", "stroll", "saunter", "march", "amble"],
    "look": ["gaze", "stare", "glance", "peer", "behold"],
    "say": ["whisper", "shout", "declare", "mutter", "proclaim"],
    "make": ["craft", "forge", "construct", "assemble", "create"],
    "break": ["shatter", "fracture", "splinter", "rupture", "demolish"],
    "give": ["bestow", "grant", "present", "hand", "deliver"],
    "take": ["seize", "grab", "snatch", "claim", "capture"],
    "find": ["discover", "locate", "uncover", "detect", "unearth"],
    "lose": ["misplace", "forfeit", "surrender", "relinquish", "abandon"],
    "begin": ["commence", "initiate", "launch", "embark", "inaugurate"],
    "end": ["conclude", "terminate", "cease", "finalize", "culminate"],
    "come": ["arrive", "approach", "enter", "emerge", "appear"],
    "go": ["depart", "leave", "exit", "vanish", "disappear"],
    "know": ["understand", "comprehend", "grasp", "recognize", "perceive"],
    "think": ["ponder", "contemplate", "reflect", "deliberate", "meditate"],
    "want": ["desire", "crave", "yearn", "covet", "long for"],
    "need": ["require", "demand", "necessitate", "warrant", "call for"],
    "feel": ["sense", "perceive", "experience", "detect", "intuit"],
    "see": ["observe", "witness", "behold", "discern", "sight"],
    "hear": ["perceive", "detect", "listen", "catch", "make out"],
    "love": ["adore", "cherish", "treasure", "revere", "idolize"],
    "hate": ["despise", "loathe", "abhor", "detest", "execrate"],
}

_WORD_RE = re.compile(r"[a-zA-Z]+")


def _evolve_replace(token: str, strength: int, repl: str) -> str:
    if not _WORD_RE.fullmatch(token) or random.random() * 100 >= strength:
        return token
    if token[0].isupper():
        return repl.capitalize()
    return repl


def evolve_words(prompt: str, strength: int) -> str:
    parts = re.split(r"(\b)", prompt)
    return "".join(
        _evolve_replace(p, strength, random.choice(EVOLVE_VOCAB))
        if _WORD_RE.fullmatch(p) else p
        for p in parts
    )


def evolve_internal(prompt: str, strength: int) -> str:
    raw_words = _WORD_RE.findall(prompt)
    if len(raw_words) < 2:
        return prompt
    parts = re.split(r"(\b)", prompt)
    return "".join(
        _evolve_replace(p, strength, random.choice(raw_words))
        if _WORD_RE.fullmatch(p) else p
        for p in parts
    )


def evolve_synonyms(prompt: str, strength: int) -> str:
    parts = re.split(r"(\b)", prompt)
    out = []
    for p in parts:
        lower = p.lower()
        syns = EVOLVE_SYNONYMS.get(lower)
        if not syns or random.random() * 100 >= strength:
            out.append(p)
        else:
            out.append(_evolve_replace(p, strength, random.choice(syns)))
    return "".join(out)


def generate_evolved(prompt: str, mode: str, strength: int, count: int, seed: int) -> str:
    if not prompt.strip():
        return ""
    base_seed = hash((int(seed), mode, prompt.strip()))
    variants = []
    for i in range(int(count)):
        random.seed(base_seed + i)
        if mode == "words":
            v = evolve_words(prompt, strength)
        elif mode == "internal":
            v = evolve_internal(prompt, strength)
        else:
            v = evolve_synonyms(prompt, strength)
        variants.append(" ".join(v.split()))
    return "\n\n".join(f"--- Variant {i + 1} ---\n{v}" for i, v in enumerate(variants))


def extract_first_variant(evolved_text: str) -> str:
    blocks = re.split(r"--- Variant \d+ ---", evolved_text.strip())
    if len(blocks) > 1:
        return blocks[1].strip()
    return evolved_text.strip()

ENHANCER_SYSTEM_T2VA = """You enhance rough video prompts into structured audiovisual rewrite prompts for T2VA (text-only, no reference pictures).

Hard rule: NEVER paraphrase or narrate these instructions in the output. Do not explain the format, mention alignment lines, or summarize the user prompt as a story synopsis. There is no image reference for T2VA. Write only concrete audiovisual scene content.

Output rules:
1) T2VA has no instruction line. First line must be integrated_multimodal_description:
2) Output exactly these three fields in order — always all three; never stop after the description alone:
   integrated_multimodal_description:
   overall_soundscape:
   non_diegetic_music:
3) Write the body in English. Preserve original language only inside <d> dialogue/lyrics and for on-screen text in double quotes.
4) [Shot 1] has no timestamp. Later shots use: [Shot N] At MM:SS.mmm, ...
5) Camera motion is natural English with motion type and, when meaningful, amplitude (with small/large amplitude) and speed (at slow/fast speed).
6) Speakers use stable IDs (S1), (S2). Dialogue format: <d>[Language] exact words</d>. Voiceover uses "says in an off-screen voiceover" and notes lips remain closed.
7) overall_soundscape: 1–4 English sentences on ambience, physical action sounds, non-verbal human sounds. No dialogue/singing/diegetic music. Use N/A only for total silence.
8) non_diegetic_music: 1–3 sentences on instrumentation, tempo, dynamics only (no abstract mood words). Use N/A when absent.
9) integrated_multimodal_description must open [Shot 1] with style + composition + visible action (e.g. "Live-action, cinematic, a medium-wide shot frames…"). Do not summarize the user prompt as a story synopsis.
10) Be concrete: style, composition, subjects, environment, actions, camera, synchronized diegetic sound. Do not invent timestamps outside the given duration."""

ENHANCER_SYSTEM_I2VA = """You enhance rough video prompts into structured audiovisual rewrite prompts for I2VA (first-frame image → video).

Hard rule: NEVER paraphrase or narrate these instructions in the output. Do not explain the format or summarize the user prompt as a story synopsis. Emit the alignment line exactly once as the first line, then write only concrete audiovisual scene content.

Output rules:
1) First line must be exactly:
   For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
   Then one blank line.
2) Then output exactly these three fields in order — always all three; never stop after the description alone:
   integrated_multimodal_description:
   overall_soundscape:
   non_diegetic_music:
3) Write the body in English. Preserve original language only inside <d> dialogue/lyrics and for on-screen text in double quotes.
4) [Shot 1] has no timestamp. Later shots use: [Shot N] At MM:SS.mmm, ...
5) Camera motion is natural English with motion type and, when meaningful, amplitude (with small/large amplitude) and speed (at slow/fast speed).
6) Speakers use stable IDs (S1), (S2). Dialogue format: <d>[Language] exact words</d>. Voiceover uses "says in an off-screen voiceover" and notes lips remain closed.
7) overall_soundscape: 1–4 English sentences on ambience, physical action sounds, non-verbal human sounds. No dialogue/singing/diegetic music. Use N/A only for total silence.
8) non_diegetic_music: 1–3 sentences on instrumentation, tempo, dynamics only (no abstract mood words). Use N/A when absent.
9) Picture 1 is the first frame of Shot 1; develop forward from it. Open [Shot 1] with style + composition locked to <Picture 1>, then action.
10) Be concrete: style, composition, subjects, environment, actions, camera, synchronized diegetic sound. Do not invent timestamps outside the given duration."""

ENHANCER_SYSTEM_FL2VA = """You enhance rough video prompts into structured audiovisual rewrite prompts for FL2VA (first + last frame → video).

Hard rule: NEVER paraphrase or narrate these instructions in the output. Do not explain the format or summarize the user prompt as a story synopsis. Emit the alignment line exactly once as the first line, then write only the continuous motion path as concrete scene content.

Output rules:
1) First line must be exactly (N = final shot number, S.SS = duration to two decimals):
   How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.
   Then one blank line.
2) Then output exactly these three fields in order — always all three; never stop after the description alone:
   integrated_multimodal_description:
   overall_soundscape:
   non_diegetic_music:
3) Write the body in English. Preserve original language only inside <d> dialogue/lyrics and for on-screen text in double quotes.
4) [Shot 1] has no timestamp. Later shots use: [Shot N] At MM:SS.mmm, ...
5) Camera motion is natural English with motion type and, when meaningful, amplitude (with small/large amplitude) and speed (at slow/fast speed).
6) Speakers use stable IDs (S1), (S2). Dialogue format: <d>[Language] exact words</d>. Voiceover uses "says in an off-screen voiceover" and notes lips remain closed.
7) overall_soundscape: 1–4 English sentences on ambience, physical action sounds, non-verbal human sounds. No dialogue/singing/diegetic music. Use N/A only for total silence.
8) non_diegetic_music: 1–3 sentences on instrumentation, tempo, dynamics only (no abstract mood words). Use N/A when absent.
9) Picture 1 is the opening; Picture 2 is the ending. Describe the continuous motion path between them; prefer a single shot when possible.
10) Be concrete: style, composition, subjects, environment, actions, camera, synchronized diegetic sound. Do not invent timestamps outside the given duration."""

ENHANCER_SYSTEM_L2VA = """You enhance rough video prompts into structured audiovisual rewrite prompts for L2VA (last-frame image → video).

Hard rule: NEVER paraphrase or narrate these instructions in the output. Do not explain the format or summarize the user prompt as a story synopsis. Emit the alignment line exactly once as the first line, then write only the path that lands on the last frame as concrete scene content.

Output rules:
1) First line must be exactly (N = final shot number, S.SS = duration to two decimals):
   How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.
   Then one blank line.
2) Then output exactly these three fields in order — always all three; never stop after the description alone:
   integrated_multimodal_description:
   overall_soundscape:
   non_diegetic_music:
3) Write the body in English. Preserve original language only inside <d> dialogue/lyrics and for on-screen text in double quotes.
4) [Shot 1] has no timestamp. Later shots use: [Shot N] At MM:SS.mmm, ...
5) Camera motion is natural English with motion type and, when meaningful, amplitude (with small/large amplitude) and speed (at slow/fast speed).
6) Speakers use stable IDs (S1), (S2). Dialogue format: <d>[Language] exact words</d>. Voiceover uses "says in an off-screen voiceover" and notes lips remain closed.
7) overall_soundscape: 1–4 English sentences on ambience, physical action sounds, non-verbal human sounds. No dialogue/singing/diegetic music. Use N/A only for total silence.
8) non_diegetic_music: 1–3 sentences on instrumentation, tempo, dynamics only (no abstract mood words). Use N/A when absent.
9) Picture 1 is the last frame of the final shot. Infer a plausible opening, then converge onto <Picture 1> by the end.
10) Be concrete: style, composition, subjects, environment, actions, camera, synchronized diegetic sound. Do not invent timestamps outside the given duration."""

ENHANCER_SYSTEM_REF = """You rewrite rough video prompts into full-reference mode structured outputs.

Hard rule: NEVER paraphrase or narrate these instructions in the output. Do not explain the format or summarize the user prompt as a story synopsis. Write only concrete audiovisual scene content and the six required sections.

Write all six sections in English, in this exact order:
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:

Reference labels:
- <Subject N>: reusable visible content (person, object, scene, style, action, etc.)
- <Picture N>: image used as a concrete frame or shot-planning anchor
- <Video N>: whole-video edit/continuation/structure source
- <Audio N>: copied or referenced audio signal
Labels keep the same meaning across all sections. Do not invent free labels (e.g. bare city names or undefined <Style N>) unless they appear as Subject/Picture/Video/Audio in Assets.

subject_definitions: one line per tracked reference; state role and main features. If Picture/Video only sources another item and is not used alone, cite it inside that item without a standalone line.

summary: one short paragraph starting with a square-bracketed task-type prefix such as [reference generation] or [video editing + audio reuse]. Use only previously defined labels. Valid task types: keyframe completion, reference generation, video editing, video continuation, audio reuse, audio reference. Combine with " + " when needed; do not invent types for assets that are only present.

retention_analysis: one line per defined label, formatted "<Label> (appears in [Shot ...]): marker - explanation".
Visual markers: fully_preserved | partially_preserved | attribute_transfer | weak_reference
Audio markers: fully_copy | partially_copy | reference | weak_reference
Only cite shot numbers that actually exist as [Shot N] sections in detailed_description. Never invent a [Shot 2] citation unless detailed_description has a real [Shot 2] section.

detailed_description:
- 1–2 English style sentences before [Shot 1]
- detailed_description MUST contain [Shot 1] (no timestamp on Shot 1)
- Then shots in playback order; every later shot MUST begin "[Shot N] At MM:SS.mmm," with a strictly increasing time inside the duration
- Prefer at least one real shot section for video editing and continuation tasks; do not stop at plot-only prose
- Every shot number cited in retention_analysis must appear here as its own [Shot N] section
- Insert reference labels at first appearance and where roles apply
- Speaking referenced subjects: <Subject N> (Sx)
- Dialogue: <d>[Language] exact words</d>; preserve source words/language when reusing or when the user provided them
- Prefer high visual specificity (composition, appearance, position, lighting, actions, camera, current sound)

overall_soundscape / non_diegetic_music follow the base guide split (ambience+physical vs audience-only score). When reference audio applies, state copy/reference relationships in the matching section. Always include both fields (use N/A when absent).

Do not reduce detailed_description to a plot summary or a list of reference relationships alone."""


def is_minimax_enhancer(model_name: str) -> bool:
    """Return True if the selected writer model is the MiniMax Prompt Enhancer."""
    if not model_name:
        return False
    name = model_name.lower()
    return ("minimax" in name and "enhancer" in name) or "video-prompt-enhancer" in name


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


RESCAN_SCRIPT = Path("/home/tonetxo/.config/h3ps/rescan_llamacpp_models.py")
GGUF_MODELS_DIR = Path("/home/tonetxo/gguf-models")


def _scan_gguf_models() -> list[str]:
    """Fallback scan of /home/tonetxo/gguf-models for .gguf files."""
    found = []
    if not GGUF_MODELS_DIR.exists():
        return found
    for subdir in sorted(GGUF_MODELS_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        for p in sorted(subdir.glob("*.gguf")):
            if "mmproj" not in p.name.lower():
                found.append(p.stem)
    return sorted(found)


def get_ollama_models(url: str, backend: str = "ollama") -> list[str]:
    if backend == "llamacpp":
        try:
            r = requests.get(url.rstrip("/") + "/v1/models", timeout=5)
            r.raise_for_status()
            models = [m.get("id", "").strip() for m in r.json().get("data", [])]
        except Exception:
            models = []
        # Ensure known fallbacks and newly-downloaded GGUFs are selectable.
        fallback = ["gemma4-e4b-obliterated", "minimax-enhancer", "qwen2.1-pe-t2i", "qwen2.1-pe-i2i", "qwen3.5-9b", "qwen3.8-27b"]
        for m in fallback:
            if m not in models:
                models.append(m)
        for m in _scan_gguf_models():
            if m not in models:
                models.append(m)
        return sorted([m for m in models if m])
    r = requests.get(url.rstrip("/") + "/api/tags", timeout=5)
    r.raise_for_status()
    models = [m.get("name", "").strip() for m in r.json().get("models", [])]
    return sorted([m for m in models if m])


def rescan_llamacpp_models() -> str:
    """Regenerate router preset, restart router, and wait for it to come up."""
    try:
        if RESCAN_SCRIPT.exists():
            import subprocess
            subprocess.run([sys.executable, str(RESCAN_SCRIPT)], check=True, timeout=60)
        subprocess.run(["systemctl", "--user", "restart", "h3ps-llamacpp.service"], check=True, timeout=180)
    except subprocess.CalledProcessError as e:
        return f"Rescan failed: {e}"
    except Exception as e:
        return f"Rescan error: {e}"

    # Wait for the router to become ready
    url = "http://127.0.0.1:8080/v1/models"
    for _ in range(40):
        try:
            r = requests.get(url, timeout=2)
            if r.ok:
                ids = [m.get("id", "").strip() for m in r.json().get("data", []) if m.get("id", "").strip()]
                return f"llama.cpp OK — {len(ids)} model(s): {', '.join(ids)}"
        except Exception:
            pass
        time.sleep(0.5)
    return "llama.cpp router restart timed out"


def check_ollama(url: str, backend: str = "ollama") -> str:
    try:
        models = get_ollama_models(url, backend)
        kind = "llama.cpp" if backend == "llamacpp" else "Ollama"
        return f"{kind} OK\n" + ("Models: " + ", ".join(models) if models else "No local models found")
    except Exception as e:
        return f"{('llama.cpp' if backend == 'llamacpp' else 'Ollama')} unavailable: {e}"


def refresh_ollama_models(url: str, current_vision: str = "", current_writer: str = "", current_motion: str = "", backend: str = "ollama"):
    try:
        models = get_ollama_models(url, backend)
        if not models:
            fallback = list(dict.fromkeys([m for m in (current_vision, current_writer, current_motion) if m]))
            return tuple(gr.Dropdown(choices=fallback, value=v or None) for v in (current_vision, current_writer, current_motion)) + ("Connected, but no local models are installed.",)
        vals = []
        for current in (current_vision, current_writer, current_motion):
            vals.append(current if current in models else models[0])
        status = f"{'llama.cpp' if backend == 'llamacpp' else 'Ollama'} OK — {len(models)} local model(s) found."
        return (gr.Dropdown(choices=models, value=vals[0], allow_custom_value=True),
                gr.Dropdown(choices=models, value=vals[1], allow_custom_value=True),
                gr.Dropdown(choices=models, value=vals[2], allow_custom_value=True), status)
    except Exception as e:
        fallback = list(dict.fromkeys([m for m in (current_vision, current_writer, current_motion) if m]))
        return (gr.Dropdown(choices=fallback, value=current_vision or None, allow_custom_value=True),
                gr.Dropdown(choices=fallback, value=current_writer or None, allow_custom_value=True),
                gr.Dropdown(choices=fallback, value=current_motion or None, allow_custom_value=True),
                f"{'llama.cpp' if backend == 'llamacpp' else 'Ollama'} unavailable: {e}")


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


def model_supports_vision(url: str, model: str, backend: str = "ollama") -> Optional[bool]:
    if backend == "llamacpp":
        # No capability endpoint; assume vision-capable (mmproj-attached models).
        return None
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


def _openai_messages(messages: list) -> list:
    """Convert Ollama-style messages (images: [b64]) to OpenAI content parts."""
    out = []
    for m in messages:
        role = m.get("role", "user")
        imgs = m.get("images") or []
        if imgs:
            parts = [{"type": "text", "text": m.get("content", "")}]
            for b64 in imgs:
                parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            out.append({"role": role, "content": parts})
        else:
            out.append({"role": role, "content": m.get("content", "")})
    return out


def _resolve_llamacpp_model(model: str) -> str:
    """Normalize model aliases for llama.cpp router."""
    if not model:
        return model
    n = model.strip().lower()
    if n in ("qwen2.1-pe-i2i", "qwen2.1-pe-i2i-official"):
        return "qwen2.1-pe-i2i"
    return model


def llamacpp_chat_completion(cfg: Dict[str, Any], model: str, messages: list, temperature: float, track_stats: Optional[list] = None, stop: Optional[list] = None, max_tokens: Optional[int] = None) -> str:
    """llama.cpp router via OpenAI-compatible /v1/chat/completions.

    Thinking is disabled with chat_template_kwargs; VRAM release is the server's
    own sleep-idle, so keep_alive does not apply.
    """
    model = _resolve_llamacpp_model(model)
    if not model:
        raise ValueError("No model selected.")
    if max_tokens is None:
        max_tokens = int(cfg.get("max_tokens") or 4096)
    payload: Dict[str, Any] = {
        "model": model,
        "messages": _openai_messages(messages),
        "temperature": float(temperature),
        "top_p": 0.90,
        "top_k": 40,
        "min_p": 0.0,
        "stream": False,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repeat_penalty": 1.0,
        "repeat_last_n": 512,
    }
    if stop:
        payload["stop"] = [s for s in stop if s != "</s>"]
    t0 = time.perf_counter()
    try:
        r = requests.post(cfg["ollama_url"].rstrip("/") + "/v1/chat/completions", json=payload, timeout=900)
    except requests.RequestException as e:
        raise RuntimeError(f"No se pudo conectar con llama.cpp ({cfg['ollama_url']}): {e}") from e
    elapsed = time.perf_counter() - t0
    if not r.ok:
        try:
            err = r.json().get("error", {}).get("message", r.text)
        except Exception:
            err = r.text
        raise RuntimeError(f"llama.cpp HTTP {r.status_code} para '{model}': {str(err)[:400]}")
    try:
        data = r.json()
        if track_stats is not None and isinstance(track_stats, list):
            usage = data.get("usage", {})
            eval_cnt = usage.get("completion_tokens", 0)
            tok_s = round(eval_cnt / elapsed, 1) if elapsed and eval_cnt else 0
            track_stats.append({
                "model": model,
                "elapsed_s": round(elapsed, 2),
                "eval_count": eval_cnt,
                "eval_duration_s": round(elapsed, 2),
                "tokens_per_second": tok_s,
                "total_duration_s": round(elapsed, 2),
                "load_duration_s": 0.0,
                "prompt_eval_count": usage.get("prompt_tokens", 0),
            })
        return data["choices"][0]["message"].get("content", "") or ""
    except (ValueError, KeyError, TypeError) as e:
        raise RuntimeError(f"Respuesta inesperada de llama.cpp para '{model}': {r.text[:1000]}") from e


def ollama_chat(cfg: Dict[str, Any], model: str, messages: list, temperature: float, track_stats: Optional[list] = None, max_tokens: Optional[int] = None) -> str:
    if not model:
        raise ValueError("No model selected.")
    if cfg.get("backend") == "llamacpp":
        return llamacpp_chat_completion(cfg, model, messages, temperature, track_stats, max_tokens=max_tokens)
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": float(temperature), "num_ctx": int(cfg["num_ctx"])},
        "keep_alive": cfg["keep_alive"],
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(cfg["ollama_url"].rstrip("/") + "/api/chat", json=payload, timeout=900)
    except requests.RequestException as e:
        raise RuntimeError(f"No se pudo conectar con Ollama ({cfg['ollama_url']}): {e}") from e
    elapsed = time.perf_counter() - t0
    if not r.ok:
        raise RuntimeError(format_ollama_http_error(r, model))
    try:
        data = r.json()
        if track_stats is not None and isinstance(track_stats, list):
            eval_cnt = data.get("eval_count", 0)
            eval_dur = data.get("eval_duration", 0)
            tok_s = round(eval_cnt / (eval_dur / 1e9), 1) if eval_dur else 0
            track_stats.append({
                "model": model,
                "elapsed_s": round(elapsed, 2),
                "eval_count": eval_cnt,
                "eval_duration_s": round(eval_dur / 1e9, 2),
                "tokens_per_second": tok_s,
                "total_duration_s": round(data.get("total_duration", 0) / 1e9, 2),
                "load_duration_s": round(data.get("load_duration", 0) / 1e9, 2),
                "prompt_eval_count": data.get("prompt_eval_count", 0),
            })
        msg = data.get("message", {})
        content = msg.get("content", "") or ""
        thinking = msg.get("thinking", "") or ""
        if not content.strip() and thinking.strip():
            content = thinking.strip()
        return content
    except (ValueError, KeyError, TypeError) as e:
        raise RuntimeError(f"Respuesta inesperada de Ollama para '{model}': {r.text[:1000]}") from e


def ollama_generate_raw(cfg: Dict[str, Any], model: str, prompt: str, temperature: float, track_stats: Optional[list] = None, stop: Optional[list[str]] = None) -> str:
    """Call Ollama /api/generate with a hand-built prompt, bypassing the chat template.

    This is required for the MiniMax Prompt Enhancer because its GGUF embeds a chat
    template that starts the assistant turn with a thinking block. Using the raw
    generate endpoint with an explicit ChatML string avoids that mismatch.

    On the llamacpp backend the server applies the model's chat template itself and
    supports chat_template_kwargs, so the plain chat endpoint is used instead.
    """
    if not model:
        raise ValueError("No model selected.")
    if cfg.get("backend") == "llamacpp":
        # ChatML hand-built prompt is unnecessary/harmful here; strip to plain text
        # by extracting the user content from the ChatML wrapper.
        parts = prompt.split("<|im_start|>user\n")
        user_text = parts[-1].split("<|im_end|>")[0] if len(parts) > 1 else prompt
        system_text = ""
        sys_parts = prompt.split("<|im_start|>system\n")
        if len(sys_parts) > 1:
            system_text = sys_parts[1].split("<|im_end|>")[0]
        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})
        return llamacpp_chat_completion(cfg, model, messages, temperature, track_stats, stop=stop)
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_ctx": int(cfg["num_ctx"]),
            "top_k": 40,
            "repeat_penalty": 1.0,
            "stop": ["<|im_end|>", "<|endoftext|>", "</s>", "🤔", "Now construct the full response"] + (stop or []),
        },
        "keep_alive": cfg["keep_alive"],
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(cfg["ollama_url"].rstrip("/") + "/api/generate", json=payload, timeout=900)
    except requests.RequestException as e:
        raise RuntimeError(f"No se pudo conectar con Ollama ({cfg['ollama_url']}): {e}") from e
    elapsed = time.perf_counter() - t0
    if not r.ok:
        raise RuntimeError(format_ollama_http_error(r, model))
    try:
        data = r.json()
        if track_stats is not None and isinstance(track_stats, list):
            eval_cnt = data.get("eval_count", 0)
            eval_dur = data.get("eval_duration", 0)
            tok_s = round(eval_cnt / (eval_dur / 1e9), 1) if eval_dur else 0
            track_stats.append({
                "model": model,
                "elapsed_s": round(elapsed, 2),
                "eval_count": eval_cnt,
                "eval_duration_s": round(eval_dur / 1e9, 2),
                "tokens_per_second": tok_s,
                "total_duration_s": round(data.get("total_duration", 0) / 1e9, 2),
                "load_duration_s": round(data.get("load_duration", 0) / 1e9, 2),
                "prompt_eval_count": data.get("prompt_eval_count", 0),
            })
        return data.get("response", "") or ""
    except (ValueError, KeyError, TypeError) as e:
        raise RuntimeError(f"Respuesta inesperada de Ollama para '{model}': {r.text[:1000]}") from e


def _repair_json(text: str) -> Dict[str, Any]:
    """Attempt to repair a truncated JSON object by closing unclosed strings/arrays/objects."""
    text = re.sub(r",\s*([}\]])", r"\1", text)
    out = []
    stack = []
    in_str = False
    esc = False
    for ch in text:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                open_ch = stack.pop()
                if (ch == "}" and open_ch != "{") or (ch == "]" and open_ch != "["):
                    continue
        out.append(ch)
    if in_str:
        out.append('"')
    while stack:
        open_ch = stack.pop()
        out.append("}" if open_ch == "{" else "]")
    repaired = "".join(out)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return json.loads(repaired, strict=False)


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    raw = text.strip()

    # 1. Look for ```json ... ``` blocks
    code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, flags=re.I)
    for block in code_blocks:
        b = block.strip()
        if not b:
            continue
        try:
            val = json.loads(b, strict=False)
            if isinstance(val, dict):
                return val
            if isinstance(val, list):
                return {"items": val}
        except Exception:
            start, end = b.find("{"), b.rfind("}")
            if start >= 0 and end > start:
                try:
                    val = json.loads(b[start:end+1], strict=False)
                    if isinstance(val, dict):
                        return val
                except Exception:
                    try:
                        return _repair_json(b[start:end+1])
                    except Exception:
                        pass

    # 2. Try direct json.loads
    try:
        val = json.loads(raw, strict=False)
        if isinstance(val, dict):
            return val
        if isinstance(val, list):
            return {"items": val}
    except Exception:
        pass

    # 3. Try finding outermost { ... }
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        cand = raw[start:end+1]
        try:
            val = json.loads(cand, strict=False)
            if isinstance(val, dict):
                return val
            if isinstance(val, list):
                return {"items": val}
        except Exception:
            try:
                return _repair_json(cand)
            except Exception:
                pass

    return None


def extract_json(text: str, fallback_key: str = "raw_output") -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {
            fallback_key: "",
            "primary_actions": [],
            "subject_motion": "No motion parsed",
            "camera_trajectory": "",
            "shots": []
        }

    # If thinking tags are present, prioritize content outside <think>...</think>
    candidates = []
    if "<think>" in text and "</think>" in text:
        parts = text.split("</think>", 1)
        after_think = parts[1].strip()
        inside_think = parts[0].replace("<think>", "").strip()
        if after_think:
            candidates.append(after_think)
        if inside_think:
            candidates.append(inside_think)

    candidates.append(text)

    for cand in candidates:
        parsed = _try_parse_json(cand)
        if parsed is not None:
            return parsed

    # Clean text from thinking tags if any for fallback
    clean_text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I).strip() or text

    # Final fallback so it never crashes the application
    return {
        fallback_key: clean_text,
        "subject_motion": clean_text,
        "primary_actions": [],
        "camera_trajectory": "",
        "continuity_bible": {},
        "shots": []
    }


def _analysis_to_text(analysis: Dict[str, Any]) -> str:
    """Flatten a forensic vision analysis dict into a concise English description."""
    if not isinstance(analysis, dict):
        return str(analysis or "").strip()
    parts = []
    scene = analysis.get("scene") or analysis.get("description")
    if scene:
        parts.append(str(scene))
    subjects = analysis.get("subjects")
    if isinstance(subjects, list):
        for s in subjects:
            if isinstance(s, dict):
                desc = s.get("description") or s.get("type")
                if desc:
                    parts.append(desc)
            else:
                parts.append(str(s))
    elif subjects:
        parts.append(str(subjects))
    for key in ("composition", "environment", "lighting", "materials", "wardrobe", "props", "atmosphere"):
        val = analysis.get(key)
        if val:
            parts.append(str(val))
    return " ".join(parts).strip() or "No visual description available."


def analysis_to_enhancer_assets(analysis: Optional[Dict[str, Any]], mode: str) -> list[str]:
    """Convert the GUI's vision analysis into enhancer Asset lines."""
    if not analysis or not isinstance(analysis, dict):
        return ["(none)"]

    # T2VA never has images in this UI.
    if mode == "T2VA":
        return ["(none)"]

    # I2VA / L2VA: single analysis dict or wrapped in single-frame structure.
    if mode in ("I2VA", "L2VA"):
        single = analysis.get("analysis") if isinstance(analysis, dict) else analysis
        if isinstance(single, dict):
            text = _analysis_to_text(single)
        else:
            text = _analysis_to_text(analysis)
        if mode == "I2VA":
            return [f"Picture 1: first frame — {text}"]
        return [f"Picture 1: last frame — {text}"]

    # FLF2VA: first_frame / last_frame structure.
    if mode == "FLF2VA":
        assets = []
        first = analysis.get("first_frame", {}).get("analysis") if isinstance(analysis.get("first_frame"), dict) else None
        last = analysis.get("last_frame", {}).get("analysis") if isinstance(analysis.get("last_frame"), dict) else None
        if first:
            assets.append(f"Picture 1: first frame — {_analysis_to_text(first)}")
        if last:
            assets.append(f"Picture 2: last frame — {_analysis_to_text(last)}")
        return assets or ["(none)"]

    # R2V: reference_images list.
    if mode == "R2V":
        refs = analysis.get("reference_images", [])
        if not refs:
            return ["(none)"]
        assets = []
        for item in refs:
            if not isinstance(item, dict):
                continue
            idx = len(assets) + 1
            tag = item.get("tag") or f"<Image_{idx}>"
            text = _analysis_to_text(item.get("analysis"))
            # The enhancer contract wants "Picture N"; the user will manually map to H3 tags later.
            assets.append(f"Picture {idx}: reference image {tag} — {text}")
        return assets or ["(none)"]

    return ["(none)"]


def _motion_plan_to_text(motion_plan: Optional[Dict[str, Any]]) -> str:
    """Convert Motion Director JSON into a short prose context block."""
    if not motion_plan or not isinstance(motion_plan, dict):
        return ""
    parts = []
    if motion_plan.get("primary_actions"):
        parts.append("Primary actions: " + " ".join(str(a) for a in motion_plan["primary_actions"] if a))
    if motion_plan.get("camera_motion"):
        cm = motion_plan["camera_motion"]
        if isinstance(cm, dict):
            parts.append("Camera: " + " ".join(str(v) for v in cm.values() if v))
        else:
            parts.append("Camera: " + str(cm))
    if motion_plan.get("timing_beats"):
        parts.append("Timing: " + " ".join(str(b) for b in motion_plan["timing_beats"] if b))
    if motion_plan.get("motion_constraints"):
        parts.append("Constraints: " + " ".join(str(c) for c in motion_plan["motion_constraints"] if c))
    if motion_plan.get("audio_events"):
        parts.append("Audio events: " + " ".join(str(e) for e in motion_plan["audio_events"] if e))
    return "\n".join(parts).strip()


def build_enhancer_user_prompt(
    mode: str,
    duration: str,
    scene: str,
    analysis: Optional[Dict[str, Any]],
    motion_plan: Optional[Dict[str, Any]],
    audio_notes: str,
    r2v_task: str = "reference_generation",
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) following the official enhancer contract."""
    # Choose system prompt and task label.
    if mode == "T2VA":
        system = ENHANCER_SYSTEM_T2VA
        task = "T2VA"
    elif mode == "I2VA":
        system = ENHANCER_SYSTEM_I2VA
        task = "I2VA"
    elif mode == "FLF2VA":
        system = ENHANCER_SYSTEM_FL2VA
        task = "FL2VA"
    elif mode == "L2VA":
        system = ENHANCER_SYSTEM_L2VA
        task = "L2VA"
    elif mode == "R2V":
        system = ENHANCER_SYSTEM_REF
        task = f"{r2v_task} (full-reference rewrite)"
    else:
        system = ENHANCER_SYSTEM_T2VA
        task = "T2VA"

    # Normalize duration to two-decimal seconds string.
    dur_str = (duration or "10s").strip()
    digits = re.sub(r"[^0-9.]", "", dur_str)
    try:
        dur_val = float(digits) if digits else 10.0
    except ValueError:
        dur_val = 10.0
    duration_line = f"{dur_val:.2f}s"

    assets = analysis_to_enhancer_assets(analysis, mode)
    assets_block = "\n".join(f"- {a}" for a in assets)

    user_parts = [f"Task: {task}", f"Duration: {duration_line}", "Assets:", assets_block]

    # Build user prompt body.
    prompt_body = (scene or "(none)").strip()
    motion_text = _motion_plan_to_text(motion_plan)
    if motion_text:
        prompt_body += f"\n\nMotion Director context (use as reference only):\n{motion_text}"
    if audio_notes and audio_notes.strip():
        prompt_body += f"\n\nAudio notes: {audio_notes.strip()}"

    user_parts.append("")
    user_parts.append("User prompt:")
    user_parts.append(prompt_body)

    return system, "\n".join(user_parts)


def generate_enhanced_prompt(
    cfg: Dict[str, Any],
    mode: str,
    duration: str,
    scene: str,
    analysis: Optional[Dict[str, Any]],
    motion_plan: Optional[Dict[str, Any]],
    audio_notes: str,
    r2v_task: str,
    track_stats: Optional[list] = None,
) -> str:
    """Call the MiniMax Prompt Enhancer using its official contract via raw generate.

    We bypass Ollama's chat template because the enhancer's GGUF embeds a template that
    forces a thinking block. A hand-built ChatML string sent to /api/generate gives the
    direct structured output the model was fine-tuned to produce.
    """
    system, user_prompt = build_enhancer_user_prompt(
        mode, duration, scene, analysis, motion_plan, audio_notes, r2v_task
    )
    chatml_prompt = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    raw = ollama_generate_raw(
        cfg,
        cfg["writer_model"],
        chatml_prompt,
        float(cfg["temperature_writer"]),
        track_stats=track_stats,
    )
    return clean_enhancer_output(raw)


def clean_enhancer_output(text: str) -> str:
    """Remove reasoning/thinking noise and return only the structured output.

    The MiniMax Prompt Enhancer sometimes emits an internal planning monologue,
    then a rough draft, then a final output. We keep only the final output by
    cutting to the LAST valid section header and stripping any transition marker
    that appears immediately before it.
    """
    if not text:
        return ""

    # Strip known thinking wrappers.
    cleaned = re.sub(r"<thinking>[\s\S]*?</thinking>", "", text, flags=re.I)
    cleaned = re.sub(r"</?think(?:ing)?>", "", cleaned, flags=re.I)
    # Strip Unicode start/end-of-thought markers used by some templates.
    cleaned = re.sub(r"[\u0003\u0004]", "", cleaned)

    # Valid enhancer output starts with one of these section headers.
    # If the model emitted a rough draft first, the real output begins at the LAST
    # occurrence of the correct header.
    valid_headers = (
        "subject_definitions:",
        "integrated_multimodal_description:",
        "For the target video, at 0.00 seconds",
        "How the reference pictures align",
    )
    last_idx = -1
    chosen_header = ""
    for header in valid_headers:
        idx = cleaned.lower().rfind(header.lower())
        if idx > last_idx:
            last_idx = idx
            chosen_header = header
    if last_idx == -1:
        # No valid header found; the whole thing is likely reasoning.
        return ""

    cleaned = cleaned[last_idx:]

    # If a transition marker sits just before the header (on the previous line or
    # appended without newline), strip the whole line up to the header.
    first_newline = cleaned.find("\n")
    if first_newline != -1:
        first_line = cleaned[:first_newline]
        transition_markers = (
            "Let me write the complete response now",
            "Now construct the full response",
            "Here is the full response",
            "Here is the final response",
            "Here is the output",
            "Here is the complete response",
            "Let me construct the final response",
        )
        for marker in transition_markers:
            marker_idx = first_line.lower().find(marker.lower())
            if marker_idx != -1:
                cleaned = cleaned[first_newline + 1:]
                break

    # Trim leading markdown/list noise.
    cleaned = re.sub(r"^[\s:\->*_`]+", "", cleaned)
    return cleaned.strip()


def is_enhancer_output_valid(mode: str, text: str) -> bool:
    """Check whether the cleaned enhancer output starts with the expected section."""
    if not text or not text.strip():
        return False
    t = text.strip().lower()
    if mode == "R2V":
        return t.startswith("subject_definitions:")
    return t.startswith("integrated_multimodal_description:") or t.startswith("for the target video") or t.startswith("how the reference pictures align")


def ollama_unload_model(url: str, model: str) -> None:
    if not model:
        return
    try:
        clean_url = url.rstrip("/") if url else "http://127.0.0.1:11434"
        requests.post(f"{clean_url}/api/generate", json={"model": model, "keep_alive": 0}, timeout=5)
    except Exception:
        pass


def ollama_unload_all(url: str) -> None:
    try:
        clean_url = url.rstrip("/") if url else "http://127.0.0.1:11434"
        r = requests.get(f"{clean_url}/api/ps", timeout=5)
        if r.ok:
            for m in r.json().get("models", []):
                name = m.get("name") or m.get("model")
                if name:
                    ollama_unload_model(clean_url, name)
    except Exception:
        pass


def analyze_image(cfg: Dict[str, Any], image_path: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": VISION_SYSTEM},
        {"role": "user", "content": "Analyze this reference image. Return JSON only.", "images": [image_to_b64(image_path)]},
    ]
    return extract_json(ollama_chat(cfg, cfg["vision_model"], messages, float(cfg["temperature_vision"])))


def analyze_visual_inputs(cfg: Dict[str, Any], images_dict: Optional[Dict[str, Any]], track_stats: Optional[list] = None) -> Optional[Dict[str, Any]]:
    """Forensic visual analysis supporting single image, dual-frame FLF, or multi-reference R2V (up to 6 images)."""
    if not images_dict:
        return None

    v_model = cfg.get("vision_model", "")
    if not v_model:
        raise gr.Error("Has cargado imágenes de referencia: selecciona un Vision model de Ollama.")

    t_v = float(cfg.get("temperature_vision", 0.15))

    try:
        # 1. Dual-frame FLF mode (First Frame + Last Frame)
        if "first_frame" in images_dict or "last_frame" in images_dict:
            first_p = images_dict.get("first_frame")
            last_p = images_dict.get("last_frame")

            if first_p and last_p:
                msg_first = [
                    {"role": "system", "content": VISION_SYSTEM},
                    {"role": "user", "content": "Analyze FIRST_FRAME (Initial State) for video interpolation. Return JSON only.", "images": [image_to_b64(first_p)]},
                ]
                first_raw = ollama_chat(cfg, v_model, msg_first, t_v, track_stats=track_stats)

                msg_last = [
                    {"role": "system", "content": VISION_SYSTEM},
                    {"role": "user", "content": "Analyze LAST_FRAME (Target Ending State) for video interpolation. Return JSON only.", "images": [image_to_b64(last_p)]},
                ]
                last_raw = ollama_chat(cfg, v_model, msg_last, t_v, track_stats=track_stats)

                return {
                    "workflow_type": "FIRST_LAST_FRAME (FLF)",
                    "first_frame": {"filename": Path(first_p).name, "analysis": extract_json(first_raw)},
                    "last_frame": {"filename": Path(last_p).name, "analysis": extract_json(last_raw)},
                }
            elif first_p:
                msg = [
                    {"role": "system", "content": VISION_SYSTEM},
                    {"role": "user", "content": "Analyze FIRST_FRAME. Return JSON only.", "images": [image_to_b64(first_p)]},
                ]
                return {
                    "workflow_type": "FIRST_LAST_FRAME (Single Frame)",
                    "first_frame": {"filename": Path(first_p).name, "analysis": extract_json(ollama_chat(cfg, v_model, msg, t_v, track_stats=track_stats))},
                }
            else:
                msg = [
                    {"role": "system", "content": VISION_SYSTEM},
                    {"role": "user", "content": "Analyze LAST_FRAME. Return JSON only.", "images": [image_to_b64(last_p)]},
                ]
                return {
                    "workflow_type": "FIRST_LAST_FRAME (Single Frame)",
                    "last_frame": {"filename": Path(last_p).name, "analysis": extract_json(ollama_chat(cfg, v_model, msg, t_v, track_stats=track_stats))},
                }

        # 2. Multi-image R2V mode (up to 6 images)
        elif "r2v_images" in images_dict:
            paths = images_dict["r2v_images"][:6]
            ref_list = []
            for i, p in enumerate(paths, 1):
                tag = f"<Image_{i}>"
                prompt = f"Analyze reference image {tag} ({Path(p).name}) for Reference-to-Video (R2V). Focus on subject identity, key visual features, wardrobe/materials, lighting and spatial anchors. Return valid JSON only."
                msg = [
                    {"role": "system", "content": VISION_SYSTEM},
                    {"role": "user", "content": prompt, "images": [image_to_b64(p)]},
                ]
                raw = ollama_chat(cfg, v_model, msg, t_v, track_stats=track_stats)
                ref_list.append({
                    "tag": tag,
                    "filename": Path(p).name,
                    "analysis": extract_json(raw),
                })
            return {
                "workflow_type": "REFERENCE_TO_VIDEO (R2V)",
                "reference_count": len(ref_list),
                "reference_images": ref_list,
            }

        # 3. Single image mode
        elif "single" in images_dict:
            single_p = images_dict["single"]
            msg = [
                {"role": "system", "content": VISION_SYSTEM},
                {"role": "user", "content": "Analyze this reference image. Return JSON only.", "images": [image_to_b64(single_p)]},
            ]
            raw = ollama_chat(cfg, v_model, msg, t_v, track_stats=track_stats)
            return extract_json(raw)

    finally:
        if cfg.get("keep_alive") == "0m":
            ollama_unload_model(cfg.get("ollama_url", "http://127.0.0.1:11434"), v_model)

    return None



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
    vision_cap = model_supports_vision(cfg["ollama_url"], model, cfg.get("backend", "ollama"))
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

def motion_director(cfg: Dict[str, Any], analysis: Dict[str, Any], scene: str, mode: str, duration: str, camera: str, track_stats: Optional[list] = None) -> Dict[str, Any]:
    text = f"""REFERENCE ANALYSIS:
{json.dumps(analysis, indent=2, ensure_ascii=False)}

USER INTENT:
{scene or '(none)'}

WORKFLOW: {mode}
DURATION: {duration}
CAMERA REQUEST: {camera or '(choose based on reference)'}

Translate the visual elements into visible, physically plausible motion.
If First-Last Frame (FLF) analysis is provided, describe the complete physical transition trajectory from the initial frame to the ending frame.
If multiple reference images (R2V) are provided with tags (<Image_1>, <Image_2>, etc.), coordinate their visible actions while preserving their visual identities and spatial relationships."""
    return extract_json(ollama_chat(cfg, cfg["motion_model"], [{"role": "system", "content": MOTION_SYSTEM}, {"role": "user", "content": text}], float(cfg["temperature_motion"]), track_stats=track_stats))


def build_writer_prompt(scene: str, analysis: Optional[Dict[str, Any]], motion_plan: Optional[Dict[str, Any]], mode: str, duration: str,
                        aspect: str, style: str, camera: str, motion: str, audio: str, dialogue: str,
                        constraints: str, negative: str, preset: str, video_ref: str = "") -> str:
    preset_data = PRESETS_DATA.get(preset, PRESETS_DATA["H3 Cinematic"])
    analysis_text = json.dumps(analysis, indent=2, ensure_ascii=False) if analysis else "NO REFERENCE IMAGE"
    motion_text = json.dumps(motion_plan, indent=2, ensure_ascii=False) if motion_plan else "NO MOTION DIRECTOR PLAN"

    # Audio instruction with automatic R2V tag injection
    if mode == "R2V":
        if audio and ("<Audio" in audio or "<Voice" in audio):
            audio_text = audio
        elif audio:
            audio_text = f"{audio} Explicitly synchronize all motion and diegetic audio pacing to <Audio_1>."
        else:
            audio_text = f"{preset_data['audio']} Explicitly synchronize all motion, rhythm and diegetic audio pacing to <Audio_1>."
    else:
        audio_text = audio or preset_data['audio']

    specific_requirements = []

    # FLF Requirement
    if mode == "FLF2VA" or (analysis and isinstance(analysis, dict) and ("first_frame" in analysis or "first_frame_analysis" in analysis)):
        specific_requirements.append("""
CRITICAL FIRST-LAST FRAME (FLF) REQUIREMENT:
This generation is in First-Last Frame interpolation mode (FLF2VA / LTX Video FLF).
You MUST explicitly detail the continuous, physically grounded visual progression starting from the composition, subjects and pose of FIRST_FRAME and smoothly evolving into the target ending composition, subjects and pose of LAST_FRAME over the requested duration. Detail the intermediate transformations and camera movement.
""")

    # Video reference instruction
    has_video_ref = bool(video_ref and video_ref.strip()) or ("<Video" in (scene or "")) or ("<Video" in (constraints or ""))
    video_anchor_desc = ""
    if has_video_ref:
        v_note = video_ref.strip() if (video_ref and video_ref.strip()) else "Subject motion dynamics, performance, rhythm and camera tracking follow <Video_1>."
        video_anchor_desc = f"- Video anchor: `<Video_1>`. You MUST explicitly embed `<Video_1>` into the prompt (e.g., '{v_note}'). Coordinate subject motion and camera choreography with <Video_1>."

    # R2V Requirement
    if mode == "R2V" or "<Audio" in (audio or "") or "<Audio" in (scene or "") or (analysis and isinstance(analysis, dict) and "reference_images" in analysis) or has_video_ref:
        ref_images = analysis.get("reference_images", []) if (analysis and isinstance(analysis, dict)) else []
        image_tags = [item.get("tag", f"<Image_{i}>") for i, item in enumerate(ref_images, 1)] if ref_images else ["<Image_1>"]
        tags_str = ", ".join(image_tags)
        
        anchor_lines = [
            "- Audio anchor: `<Audio_1>` (or `<Audio_0>` if specified). You MUST embed `<Audio_1>` into the prompt (e.g. 'Action, diegetic sound design and rhythm are precisely synchronized with <Audio_1>').",
            f"- Visual anchors: {tags_str}. You MUST explicitly reference each visual anchor ({tags_str}) in the prompt describing their specific actions, materials, identity and spatial interactions."
        ]
        if has_video_ref:
            anchor_lines.append(video_anchor_desc)

        anchors_block = "\n".join(anchor_lines)
        specific_requirements.append(f"""
CRITICAL R2V REQUIREMENT:
This generation is in R2V (Reference-to-Video) mode.
Available reference anchors:
{anchors_block}
NEVER omit the reference tags in the output.
""")
    elif has_video_ref:
        specific_requirements.append(f"""
CRITICAL VIDEO REFERENCE REQUIREMENT:
A reference video is active in this generation.
{video_anchor_desc}
NEVER omit the `<Video_1>` tag in the final prompt.
""")

    req_text = "\n".join(specific_requirements)

    return f"""Create one final production-ready prompt. PRIMARY TARGET: MiniMax H3. WORKFLOW: {mode}. DURATION: {duration}. ASPECT RATIO: {aspect}. PRESET: {preset}.\n\nUSER SCENE / INTENT:\n{scene or '(none)'}\n\nVISUAL REFERENCE ANALYSIS:\n{analysis_text}\n\nMOTION DIRECTOR PLAN:\n{motion_text}\n\nSTYLE:\n{style or preset_data['style']}\n\nCAMERA:\n{camera or preset_data['camera']}\n\nMOTION PRIORITIES:\n{motion or preset_data['motion']}\n\nAUDIO:\n{audio_text}\n\nDIALOGUE:\n{dialogue or '(none unless explicitly requested)'}\n\nCONTINUITY / CONSTRAINTS:\n{constraints or '(preserve identity, wardrobe, props, geometry and lighting logic)'}\n\nAVOID:\n{negative or preset_data['negative']}\n{req_text}\nWrite a coherent temporal sequence rather than a keyword list. Make important movements observable and causally connected. Keep the scene spatially consistent. Output only the final English prompt, with no markdown fences and no explanation."""


def generate_prompt(cfg: Dict[str, Any], writer_prompt: str, track_stats: Optional[list] = None) -> str:
    return ollama_chat(cfg, cfg["writer_model"], [{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": writer_prompt}], float(cfg["temperature_writer"]), track_stats=track_stats)


def load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_history(items: list[dict]) -> None:
    HISTORY_PATH.write_text(json.dumps(items[-100:], indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# UI state persistence
# ---------------------------------------------------------------------------

UI_STATE_FIELDS = [
    "mode", "duration", "aspect",
    "single_image", "flf_first", "flf_last",
    "r2v_state", "r2v_video_enabled", "r2v_video_file", "r2v_video_notes", "r2v_task",
    "scene", "preset", "style", "camera", "motion", "audio", "dialogue", "constraints", "negative",
    "motion_enabled", "title",
    "backend", "ollama_url", "vision_model", "motion_model", "writer_model",
    "temperature_vision", "temperature_motion", "temperature_writer", "num_ctx", "keep_alive",
    "analysis", "motion_plan", "prompt", "timing_display",
    "evolve_mode", "evolve_strength", "evolve_count", "evolve_seed",
]

DIRECTOR_UI_STATE_FIELDS = [
    "d_scene", "d_ref_state", "d_ref_labels",
    "d_mode", "d_total", "d_shot", "d_preset",
    "d_style", "d_camera", "d_motion", "d_audio", "d_dialogue", "d_constraints",
    "director_state", "d_bible", "d_storyboard", "d_prompts", "d_status",
]


def _copy_ref_to_persistent(path: Optional[str], subdir: str) -> Optional[str]:
    """Copy an uploaded/dropped reference image into outputs/ui_refs/<subdir> so it survives restarts."""
    if not path:
        return None
    src = Path(path)
    if not src.exists():
        return str(path)
    dest_dir = UI_REFS_DIR / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Add a timestamp prefix to avoid collisions and keep original stem.
    ts = time.strftime("%Y%m%d_%H%M%S")
    dest = dest_dir / f"{ts}_{src.name}"
    try:
        import shutil
        shutil.copy2(src, dest)
        return str(dest)
    except Exception:
        return str(path)


def _persist_refs(state: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve transient uploaded image paths to persistent copies before saving state."""
    if not isinstance(state, dict):
        return state
    out = dict(state)
    for key in ("single_image", "flf_first", "flf_last"):
        if out.get(key):
            out[key] = _copy_ref_to_persistent(out[key], key)
    r2v = out.get("r2v_state")
    if isinstance(r2v, list):
        out["r2v_state"] = [_copy_ref_to_persistent(p, "r2v") for p in r2v]
    d_refs = out.get("d_ref_state")
    if isinstance(d_refs, list):
        out["d_ref_state"] = [_copy_ref_to_persistent(p, "director") for p in d_refs]
    return out


def load_ui_state() -> Dict[str, Any]:
    if not UI_STATE_PATH.exists():
        return {}
    try:
        data = json.loads(UI_STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_ui_state(state: Dict[str, Any]) -> None:
    try:
        UI_STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def clear_ui_state() -> None:
    if UI_STATE_PATH.exists():
        try:
            UI_STATE_PATH.unlink()
        except Exception:
            pass


def save_main_ui_state(values: Dict[str, Any]) -> None:
    """Persist the main tab UI state. Transient image paths are copied first."""
    state = load_ui_state()
    state.update(_persist_refs(values))
    save_ui_state(state)


def save_director_ui_state(values: Dict[str, Any]) -> None:
    """Persist the Director Mode tab UI state."""
    state = load_ui_state()
    state.update(_persist_refs(values))
    save_ui_state(state)


def history_choices(items: list[dict]) -> list[str]:
    res = []
    for i in reversed(items):
        t_tot = i.get("timings", {}).get("total_s")
        t_str = f" [{t_tot:.1f}s]" if t_tot is not None else ""
        res.append(f"{i.get('id','')} — {i.get('title','Untitled')}{t_str}")
    return res


def format_timing_badge(timings: Dict[str, Any], enhancer_fallback: bool = False) -> str:
    if not timings:
        return "⏱️ **Tiempos de ejecución:** *(Aún no se ha generado ningún prompt)*"

    tot = timings.get("total_s", 0)
    parts = []

    v_time = timings.get("vision_s", 0)
    v_model = timings.get("vision_model", "")
    if v_time and v_time > 0:
        v_speed = timings.get("vision_tok_s", 0)
        speed_str = f" · *{v_speed:.1f} tok/s*" if v_speed else ""
        parts.append(f"👁️ **Visión:** `{v_model}` **{v_time:.2f}s**{speed_str}")

    m_time = timings.get("motion_s", 0)
    m_model = timings.get("motion_model", "")
    if m_time and m_time > 0:
        m_speed = timings.get("motion_tok_s", 0)
        speed_str = f" · *{m_speed:.1f} tok/s*" if m_speed else ""
        parts.append(f"🎬 **Motion:** `{m_model}` **{m_time:.2f}s**{speed_str}")

    w_time = timings.get("writer_s", 0)
    w_model = timings.get("writer_model", "")
    if w_time and w_time > 0:
        w_speed = timings.get("writer_tok_s", 0)
        speed_str = f" · *{w_speed:.1f} tok/s*" if w_speed else ""
        fallback_note = " · ⚠️ *fallback genérico*" if enhancer_fallback else ""
        parts.append(f"✍️ **Writer:** `{w_model}` **{w_time:.2f}s**{speed_str}{fallback_note}")

    stages = " &nbsp;│&nbsp; ".join(parts) if parts else ""
    return f"⏱️ **Tiempo Total:** `{tot:.2f}s` &nbsp;│&nbsp; {stages}" if stages else f"⏱️ **Tiempo Total:** `{tot:.2f}s`"


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


def do_generate(single_image, flf_first, flf_last, r2v_state, r2v_video_enabled, r2v_video_file, r2v_video_notes,
                scene, mode, duration, aspect, preset, style, camera, motion, audio, dialogue,
                constraints, negative, motion_enabled, backend, ollama_url, vision_model, writer_model, motion_model,
                temperature_writer, temperature_vision, temperature_motion, num_ctx, keep_alive, title,
                r2v_task="reference_generation"):
    cfg = load_config()
    cfg.update({"backend": backend, "ollama_url": ollama_url.strip() or cfg["ollama_url"], "vision_model": vision_model.strip(),
                "writer_model": writer_model.strip(), "motion_model": motion_model.strip(),
                "temperature_writer": float(temperature_writer), "temperature_vision": float(temperature_vision),
                "temperature_motion": float(temperature_motion), "num_ctx": int(num_ctx), "keep_alive": keep_alive,
                "preset": preset})
    save_config(cfg)
    if not cfg["writer_model"]:
        raise gr.Error("Selecciona un Writer model de Ollama.")

    # Resolve active visual inputs based on mode
    images_dict = {}
    if mode == "FLF2VA":
        first_p = flf_first or single_image
        last_p = flf_last
        if first_p:
            images_dict["first_frame"] = first_p
        if last_p:
            images_dict["last_frame"] = last_p
        if not first_p and not last_p:
            images_dict = None
    elif mode == "R2V":
        r2v_paths = list(r2v_state or [])
        if not r2v_paths and single_image:
            r2v_paths = [single_image]
        if r2v_paths:
            images_dict["r2v_images"] = r2v_paths[:6]
        else:
            images_dict = None
    else:
        img_p = single_image or flf_first
        if img_p:
            images_dict["single"] = img_p
        else:
            images_dict = None

    video_ref_str = ""
    if mode == "R2V" and r2v_video_enabled:
        file_hint = f" ({Path(r2v_video_file).name})" if r2v_video_file else ""
        note = r2v_video_notes.strip() if r2v_video_notes else "Subject motion dynamics, pacing, performance and camera tracking follow <Video_1>."
        video_ref_str = f"{note}{file_hint}"
    elif "<Video" in (scene or "") or "<Video" in (constraints or ""):
        video_ref_str = "<Video_1>"

    stats_log = []
    t_total_start = time.perf_counter()

    t_v_start = time.perf_counter()
    analysis = analyze_visual_inputs(cfg, images_dict, track_stats=stats_log)
    t_vision = time.perf_counter() - t_v_start if images_dict else 0.0

    motion_plan = None
    t_motion = 0.0
    if motion_enabled and analysis:
        if not cfg["motion_model"]:
            raise gr.Error("Motion Director está activado: selecciona un Motion model de Ollama.")
        t_m_start = time.perf_counter()
        motion_plan = motion_director(cfg, analysis, scene, mode, duration, camera, track_stats=stats_log)
        t_motion = time.perf_counter() - t_m_start

    t_w_start = time.perf_counter()
    final = ""
    enhancer_used = False
    if is_minimax_enhancer(cfg["writer_model"]):
        enhancer_used = True
        final = generate_enhanced_prompt(
            cfg, mode, duration, scene, analysis, motion_plan, audio, r2v_task, track_stats=stats_log
        )
        if not is_enhancer_output_valid(mode, final):
            # Fallback to the generic writer pipeline when the enhancer emits only reasoning.
            wp = build_writer_prompt(scene, analysis, motion_plan, mode, duration, aspect, style, camera, motion, audio, dialogue, constraints, negative, preset, video_ref=video_ref_str)
            final = generate_prompt(cfg, wp, track_stats=stats_log)
            enhancer_used = False
    else:
        wp = build_writer_prompt(scene, analysis, motion_plan, mode, duration, aspect, style, camera, motion, audio, dialogue, constraints, negative, preset, video_ref=video_ref_str)
        final = generate_prompt(cfg, wp, track_stats=stats_log)
    t_writer = time.perf_counter() - t_w_start

    t_total = time.perf_counter() - t_total_start

    writer_speed = next((s["tokens_per_second"] for s in reversed(stats_log) if s["model"] == writer_model), 0)
    vision_speed = next((s["tokens_per_second"] for s in stats_log if s["model"] == vision_model), 0) if images_dict else 0
    motion_speed = next((s["tokens_per_second"] for s in stats_log if s["model"] == motion_model), 0) if motion_plan else 0

    timings = {
        "total_s": round(t_total, 2),
        "vision_s": round(t_vision, 2),
        "motion_s": round(t_motion, 2),
        "writer_s": round(t_writer, 2),
        "vision_model": vision_model if images_dict else "",
        "motion_model": motion_model if motion_plan else "",
        "writer_model": writer_model,
        "writer_tok_s": writer_speed,
        "vision_tok_s": vision_speed,
        "motion_tok_s": motion_speed,
    }

    record = {"prompt": final, "scene": scene, "analysis": analysis, "motion_plan": motion_plan, "mode": mode,
              "duration": duration, "aspect": aspect, "preset": preset, "style": style, "camera": camera,
              "motion": motion, "audio": audio, "dialogue": dialogue, "constraints": constraints, "negative": negative,
              "vision_model": vision_model, "writer_model": writer_model, "motion_model": motion_model,
              "references": images_dict if images_dict else None, "video_ref": video_ref_str or None,
              "timings": timings}
    items, hid = add_history(title, record)

    if keep_alive == "0m":
        ollama_unload_all(cfg["ollama_url"])

    timing_md = format_timing_badge(timings, enhancer_fallback=(is_minimax_enhancer(writer_model) and not enhancer_used))

    # Persist main UI state so it survives restarts.
    save_main_ui_state({
        "mode": mode, "duration": duration, "aspect": aspect,
        "single_image": single_image, "flf_first": flf_first, "flf_last": flf_last,
        "r2v_state": r2v_state, "r2v_video_enabled": r2v_video_enabled,
        "r2v_video_file": r2v_video_file, "r2v_video_notes": r2v_video_notes, "r2v_task": r2v_task,
        "scene": scene, "preset": preset, "style": style, "camera": camera,
        "motion": motion, "audio": audio, "dialogue": dialogue, "constraints": constraints,
        "negative": negative, "motion_enabled": motion_enabled, "title": title,
        "ollama_url": ollama_url, "vision_model": vision_model, "motion_model": motion_model,
        "writer_model": writer_model,
        "temperature_vision": temperature_vision, "temperature_motion": temperature_motion,
        "temperature_writer": temperature_writer, "num_ctx": num_ctx, "keep_alive": keep_alive,
        "analysis": json.dumps(analysis, indent=2, ensure_ascii=False) if analysis else "No reference image.",
        "motion_plan": json.dumps(motion_plan, indent=2, ensure_ascii=False) if motion_plan else "Motion Director disabled or no reference image.",
        "prompt": final, "timing_display": timing_md,
    })

    return (timing_md,
            json.dumps(analysis, indent=2, ensure_ascii=False) if analysis else "No reference image.",
            json.dumps(motion_plan, indent=2, ensure_ascii=False) if motion_plan else "Motion Director disabled or no reference image.",
            final, check_ollama(cfg["ollama_url"]), gr.update(choices=history_choices(items), value=f"{hid} — {title.strip() or 'Untitled'} [{timings['total_s']:.1f}s]"))


def restore_history(choice):
    item = get_history_item(choice)
    if not item:
        return ("", "", "", "I2VA", "10s", "16:9", "H3 Cinematic", "", "", "", "", "", "", False, "", "", "⏱️ **Tiempos de ejecución:** *(Sin datos)*")
    timing_md = format_timing_badge(item.get("timings", {}))
    return (item.get("scene", ""), item.get("prompt", ""), json.dumps(item.get("analysis"), indent=2, ensure_ascii=False) if item.get("analysis") else "No reference image.",
            item.get("mode", "I2VA"), item.get("duration", "10s"), item.get("aspect", "16:9"), item.get("preset", "H3 Cinematic"),
            item.get("style", ""), item.get("camera", ""), item.get("motion", ""), item.get("audio", ""), item.get("dialogue", ""),
            item.get("constraints", ""), False, item.get("motion_plan") and json.dumps(item.get("motion_plan"), indent=2, ensure_ascii=False) or "", item.get("title", ""), timing_md)


def compare_history(a, b):
    ia, ib = get_history_item(a), get_history_item(b)
    if not ia or not ib:
        return "Selecciona dos versiones para comparar."
    
    chunks = []
    
    # 1. Performance & Model Comparison Table
    ta = ia.get("timings", {})
    tb = ib.get("timings", {})
    
    timing_table = []
    timing_table.append("### ⏱️ Comparativa de Rendimiento y Modelos")
    timing_table.append("| Métrica / Etapa | Versión A | Versión B | Comparación / Diferencia |")
    timing_table.append("| :--- | :--- | :--- | :--- |")
    
    vm_a, vm_b = ia.get("vision_model", "-") or "-", ib.get("vision_model", "-") or "-"
    mm_a, mm_b = ia.get("motion_model", "-") or "-", ib.get("motion_model", "-") or "-"
    wm_a, wm_b = ia.get("writer_model", "-") or "-", ib.get("writer_model", "-") or "-"
    
    timing_table.append(f"| **Vision Model** | `{vm_a}` | `{vm_b}` | {'Idéntico' if vm_a == vm_b else '⚠️ Diferente'} |")
    timing_table.append(f"| **Motion Model** | `{mm_a}` | `{mm_b}` | {'Idéntico' if mm_a == mm_b else '⚠️ Diferente'} |")
    timing_table.append(f"| **Writer Model** | `{wm_a}` | `{wm_b}` | {'Idéntico' if wm_a == wm_b else '⚠️ Diferente'} |")
    
    def fmt_diff(va, vb):
        if va and vb:
            diff = vb - va
            pct = ((vb - va) / va * 100) if va else 0
            sign = "+" if diff > 0 else ""
            if abs(diff) < 0.05:
                return "Equivalente (~0.0s)"
            winner = " 🟢 (A más rápido)" if diff > 0 else " 🟢 (B más rápido)"
            return f"`{sign}{diff:.2f}s` ({sign}{pct:.1f}%){winner}"
        elif va and not vb:
            return "Solo en A"
        elif vb and not va:
            return "Solo en B"
        return "—"

    if ta or tb:
        tot_a, tot_b = ta.get("total_s", 0), tb.get("total_s", 0)
        v_a, v_b = ta.get("vision_s", 0), tb.get("vision_s", 0)
        m_a, m_b = ta.get("motion_s", 0), tb.get("motion_s", 0)
        w_a, w_b = ta.get("writer_s", 0), tb.get("writer_s", 0)
        
        timing_table.append(f"| **Tiempo Visión** | {v_a:.2f}s | {v_b:.2f}s | {fmt_diff(v_a, v_b)} |")
        timing_table.append(f"| **Tiempo Motion** | {m_a:.2f}s | {m_b:.2f}s | {fmt_diff(m_a, m_b)} |")
        timing_table.append(f"| **Tiempo Writer** | {w_a:.2f}s | {w_b:.2f}s | {fmt_diff(w_a, w_b)} |")
        timing_table.append(f"| **⚡ TIEMPO TOTAL** | **{tot_a:.2f}s** | **{tot_b:.2f}s** | **{fmt_diff(tot_a, tot_b)}** |")
        
        spd_w_a, spd_w_b = ta.get("writer_tok_s", 0), tb.get("writer_tok_s", 0)
        if spd_w_a or spd_w_b:
            spd_diff = spd_w_b - spd_w_a
            sign = "+" if spd_diff > 0 else ""
            timing_table.append(f"| **Velocidad Writer** | {spd_w_a:.1f} tok/s | {spd_w_b:.1f} tok/s | `{sign}{spd_diff:.1f} tok/s` |")
    else:
        timing_table.append("| *(Sin datos de timer en estas versiones antiguas)* | — | — | — |")
        
    chunks.append("\n".join(timing_table))
    
    # 2. Field differences
    fields = ["preset", "mode", "duration", "aspect", "scene", "camera", "motion", "audio", "dialogue", "prompt"]
    diff_chunks = []
    for f in fields:
        av, bv = str(ia.get(f, "")), str(ib.get(f, ""))
        if av != bv:
            diff_chunks.append(f"#### {f.capitalize()}\n**A:**\n```text\n{av}\n```\n**B:**\n```text\n{bv}\n```")
    if diff_chunks:
        chunks.append("### 📝 Diferencias de Prompt y Parámetros")
        chunks.extend(diff_chunks)
    else:
        chunks.append("*(Los parámetros y el prompt generado son idénticos)*")
        
    return "\n\n".join(chunks)


def save_text(kind: str, text: str) -> str:
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    archive = out / "archive"
    archive.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = "h3_prompt" if kind == "prompt" else "vision_analysis" if kind == "analysis" else "motion_plan"
    ext = ".txt" if kind == "prompt" else ".json"

    latest_path = out / f"{base_name}{ext}"
    archive_path = archive / f"{base_name}_{timestamp}{ext}"

    if kind in {"analysis", "motion"}:
        try:
            text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except Exception:
            pass

    content = text or ""
    latest_path.write_text(content, encoding="utf-8")
    archive_path.write_text(content, encoding="utf-8")
    return f"Saved: {latest_path.name} | Archived: archive/{archive_path.name}"


def save_director(result):
    if not result:
        return "No director package to save."
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pkg_dir = out / "director_packages" / f"package_{timestamp}"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    shots_dir = out / "director_shots"
    shots_dir.mkdir(exist_ok=True)
    (out / "director_sequence.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "reference_library.json").write_text(json.dumps(result.get("reference_library", []), indent=2, ensure_ascii=False), encoding="utf-8")

    (pkg_dir / "director_sequence.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (pkg_dir / "reference_library.json").write_text(json.dumps(result.get("reference_library", []), indent=2, ensure_ascii=False), encoding="utf-8")

    for p in result.get("prompts", []):
        shot_filename = f"{p['shot_id']}.txt"
        (shots_dir / shot_filename).write_text(p["prompt"], encoding="utf-8")
        (pkg_dir / shot_filename).write_text(p["prompt"], encoding="utf-8")

    return f"Saved in outputs/ and archived in: outputs/director_packages/package_{timestamp}/"


def apply_preset(preset):
    p = PRESETS_DATA[preset]
    return p["style"], p["camera"], p["motion"], p["audio"], p["negative"]


def extract_paths(value: Any) -> list[str]:
    """Normalize gr.Gallery / gr.File values into a list of file paths."""
    if value is None:
        return []

    def _get_path_from_single(item: Any) -> str | None:
        if item is None:
            return None
        if isinstance(item, str):
            return item
        if isinstance(item, tuple):
            return _get_path_from_single(item[0]) if len(item) > 0 else None
        if isinstance(item, dict):
            if "path" in item and item["path"]:
                return item["path"]
            if "image" in item:
                img = item["image"]
                p = img.get("path") if isinstance(img, dict) else img
                return _get_path_from_single(p)
            if "video" in item:
                vid = item["video"]
                p = vid.get("path") if isinstance(vid, dict) else vid
                return _get_path_from_single(p)
        p = getattr(item, "path", None)
        return str(p) if p else None

    if isinstance(value, list):
        out = []
        for item in value:
            p = _get_path_from_single(item)
            if p and p not in out:
                out.append(p)
        return out

    single = _get_path_from_single(value)
    return [single] if single else []



def director_generate(scene, mode, reference_files, reference_labels, total_duration, shot_duration, preset, style, camera, motion, audio, dialogue, constraints,
                       backend, ollama_url, vision_model, director_model, writer_model, tv, td, tw, num_ctx, keep_alive):
    cfg = load_config()
    # Director Mode must use exactly the models selected in the GUI.
    # In particular, reference analysis must never fall back to an old/default Vision model.
    cfg.update({
        "backend": backend,
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
    reference_files = extract_paths(reference_files)
    if reference_files and not vision_model:
        raise ValueError("Has cargado imágenes de referencia: selecciona un Vision model de Ollama.")
    if not director_model or not writer_model:
        raise ValueError("Selecciona Director/Planner y Writer models de Ollama.")
    total = int(total_duration); target = int(shot_duration)
    if not 20 <= total <= 180:
        raise ValueError("La duración total debe estar entre 20 y 180 segundos.")
    count = max(2, round(total / target))
    while count > 2 and total // count < 5:
        count -= 1
    base, rem = divmod(total, count)
    durations = [base + (1 if i < rem else 0) for i in range(count)]
    reference_library = analyze_reference_library(cfg, reference_files or [], reference_labels, tv)
    planner_input = {
        "scene": scene, "target_workflow": mode, "reference_library": reference_library, "preset": preset,
        "global_style": style, "camera_preferences": camera, "motion_preferences": motion,
        "global_audio": audio, "dialogue": dialogue, "constraints": constraints,
        "total_duration_seconds": total, "target_shot_duration_seconds": target,
        "shot_count": count, "shot_durations_seconds": durations
    }
    raw_plan = ollama_chat(cfg, director_model, [
        {"role":"system","content":DIRECTOR_SYSTEM},
        {"role":"user","content":json.dumps(planner_input, ensure_ascii=False, indent=2)}
    ], td, max_tokens=4096)
    plan = extract_json(raw_plan)
    shots = plan.get("shots", [])
    if not shots:
        print(f"[Director] Error: No shots parsed. Raw output (len={len(raw_plan)}):\n{raw_plan[:800]}")
        raise ValueError("Director model no devolvió ningún plano.")
    bible = plan.get("continuity_bible", {})
    timeline = 0
    prompts = []
    for i, shot in enumerate(shots, 1):
        dur = int(shot.get("duration") or durations[min(i-1, len(durations)-1)])
        dur = max(5, min(30, dur))
        shot_id = shot.get("shot_id") or f"SHOT_{i:02d}"
        shot["shot_id"] = shot_id; shot["start_time"] = timeline; shot["end_time"] = timeline + dur; shot["duration"] = dur
        payload = {"target_workflow": mode, "workflow_target": f"MiniMax H3 ({mode})", "shot": shot, "continuity_bible": bible, "global_style": plan.get("global_style") or style,
                   "global_audio": plan.get("global_audio") or audio, "reference_library": reference_library, "user_constraints": constraints}
        prompt = ollama_chat(cfg, writer_model, [
            {"role":"system","content":DIRECTOR_PROMPT_SYSTEM},
            {"role":"user","content":json.dumps(payload, ensure_ascii=False, indent=2)}
        ], tw, max_tokens=4096).strip()
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


def build_ui():
    cfg = load_config()
    hist = load_history()
    ui_state = load_ui_state()

    # Visibility of reference containers follows the persisted workflow mode.
    initial_mode = ui_state.get("mode", "I2VA")
    single_visible = initial_mode not in ("FLF2VA", "R2V")
    flf_visible = initial_mode == "FLF2VA"
    r2v_visible = initial_mode == "R2V"

    # Restore saved preset or fall back to config default.
    initial_preset = ui_state.get("preset") or cfg.get("preset", "H3 Cinematic")
    initial_p = PRESETS_DATA.get(initial_preset, PRESETS_DATA["H3 Cinematic"])

    with gr.Blocks(title="MiniMax H3 Prompt Studio") as app:
        gr.Markdown("# MiniMax H3 Prompt Studio\nLocal GUI · Ollama Vision → Motion Director → H3 Prompt Engineer")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    mode = gr.Dropdown(MODES, value=initial_mode, label="Workflow Mode")
                    duration = gr.Dropdown(["5s", "6s", "8s", "10s", "12s", "15s", "18s", "20s", "25s", "30s"], value=ui_state.get("duration", "10s"), label="Duration")
                    aspect = gr.Dropdown(["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"], value=ui_state.get("aspect", "16:9"), label="Aspect")

                # --- Adaptive Visual References Container ---
                # 1. Single Reference (I2VA, L2VA, T2VA)
                with gr.Row(visible=single_visible) as single_img_row:
                    single_image = gr.Image(value=ui_state.get("single_image"), type="filepath", label="Reference Image (<Image_1>)", sources=["upload", "clipboard"])

                # 2. Dual Reference (FLF2VA / LTX Video FLF)
                with gr.Row(visible=flf_visible) as flf_img_row:
                    flf_first = gr.Image(value=ui_state.get("flf_first"), type="filepath", label="First Frame (<Image_1> - Start / Initial State)", sources=["upload", "clipboard"])
                    flf_last = gr.Image(value=ui_state.get("flf_last"), type="filepath", label="Last Frame (<Image_2> - End / Target State)", sources=["upload", "clipboard"])

                # 3. Multi Reference (R2V - up to 6 images)
                saved_r2v_paths = ui_state.get("r2v_state", [])
                saved_r2v_items = [(p, f"<Image_{i}>: {Path(p).name}") for i, p in enumerate(saved_r2v_paths, 1) if Path(p).exists()]
                r2v_info_md = (
                    f"**{len(saved_r2v_items)}/6 referencias cargadas:** " + ", ".join([f"`<Image_{i}>`" for i in range(1, len(saved_r2v_items) + 1)])
                    if saved_r2v_items
                    else "*(Sube hasta 6 imágenes para R2V. Cada imagen se etiquetará automáticamente como `<Image_1>` a `<Image_6>`)*"
                )
                with gr.Column(visible=r2v_visible) as r2v_img_col:
                    r2v_state = gr.State(saved_r2v_paths)
                    r2v_selected_idx = gr.State(None)
                    r2v_file = gr.File(
                        label="Upload Reference Images for R2V (Up to 6 images: <Image_1> to <Image_6>)",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath",
                    )
                    r2v_gallery = gr.Gallery(
                        value=saved_r2v_items,
                        label="R2V References (<Image_1> ... <Image_6>)",
                        show_label=True,
                        columns=3,
                        height=200,
                        object_fit="cover",
                        type="filepath",
                        interactive=True,
                    )
                    with gr.Row():
                        r2v_remove_btn = gr.Button("🗑️ Eliminar seleccionada", size="sm", interactive=False)
                        r2v_clear_btn = gr.Button("🧹 Borrar todas las referencias", size="sm", variant="secondary")
                    r2v_task = gr.Dropdown(
                        choices=ENHANCER_R2V_TASKS,
                        value=ui_state.get("r2v_task", "reference_generation"),
                        label="R2V enhancer task label",
                        info="Only used when the Writer model is the MiniMax Prompt Enhancer.",
                        visible=r2v_visible,
                    )
                    r2v_tags_info = gr.Markdown(r2v_info_md)
                    r2v_video_enabled = gr.Checkbox(label="🎬 Incluir Referencia de Vídeo (<Video_1>)", value=ui_state.get("r2v_video_enabled", False))
                    r2v_video_box_visible = bool(ui_state.get("r2v_video_enabled", False))
                    with gr.Column(visible=r2v_video_box_visible) as r2v_video_box:
                        r2v_video_file = gr.File(value=ui_state.get("r2v_video_file"), label="Archivo de Vídeo opcional (<Video_1>) — (Referencia de etiqueta, 0 VRAM)", file_count="single", file_types=["video"], type="filepath")
                        r2v_video_notes = gr.Textbox(
                            label="Instrucción de Vídeo (<Video_1>)",
                            placeholder="Opcional: p.ej. 'Subject motion dynamics, pacing, performance and camera tracking follow <Video_1>'",
                            lines=2,
                            value=ui_state.get("r2v_video_notes", "Subject motion dynamics, pacing, performance and camera tracking follow <Video_1>.")
                        )
                # ---------------------------------------------

                scene = gr.Textbox(label="Scene / intent", lines=5, value=ui_state.get("scene", ""), placeholder="Describe what should happen in the shot…")
                preset = gr.Dropdown(PRESETS, value=initial_preset, label="Cinematic preset")
                style = gr.Textbox(label="Style", lines=2, value=ui_state.get("style", initial_p["style"]))
                camera = gr.Textbox(label="Camera", lines=2, value=ui_state.get("camera", initial_p["camera"]))
                motion = gr.Textbox(label="Motion priorities", lines=3, value=ui_state.get("motion", initial_p["motion"]))
                audio = gr.Textbox(label="Audio", lines=2, value=ui_state.get("audio", initial_p["audio"]))
                dialogue = gr.Textbox(label="Dialogue", lines=2, value=ui_state.get("dialogue", ""), placeholder="Optional. Identify every speaker explicitly.")
                constraints = gr.Textbox(label="Continuity / constraints", lines=3,
                    value=ui_state.get("constraints", "Preserve the reference composition, character identity, wardrobe, props, lighting and spatial geometry unless the requested action requires change."))
                negative = gr.Textbox(label="Avoid", lines=3, value=ui_state.get("negative", initial_p["negative"]))
                motion_enabled = gr.Checkbox(value=ui_state.get("motion_enabled", True), label="Enable Motion Director (requires reference image)")
                title = gr.Textbox(label="History title", value=ui_state.get("title", "H3 draft"))
                generate = gr.Button("Generate H3 Prompt", variant="primary", size="lg")

                with gr.Accordion("Ollama / advanced", open=True):
                    backend_sel = gr.Dropdown(choices=["ollama", "llamacpp"], value=ui_state.get("backend", cfg.get("backend", "ollama")), label="Backend", allow_custom_value=False)
                    ollama_url = gr.Textbox(value=ui_state.get("ollama_url", cfg["ollama_url"]), label="Ollama / llama.cpp URL")
                    with gr.Row():
                        vision_model = gr.Dropdown(choices=[], value=ui_state.get("vision_model", cfg.get("vision_model", "")), label="Vision model", allow_custom_value=True)
                        motion_model = gr.Dropdown(choices=[], value=ui_state.get("motion_model", cfg.get("motion_model", "")), label="Motion model", allow_custom_value=True)
                        writer_model = gr.Dropdown(choices=[], value=ui_state.get("writer_model", cfg.get("writer_model", "")), label="Writer model", allow_custom_value=True)
                    with gr.Row():
                        refresh_models = gr.Button("Refresh models")
                        check = gr.Button("Check Ollama")
                        rescan_btn = gr.Button("Rescan llama.cpp models", variant="secondary")
                    temperature_vision = gr.Slider(0, 1, value=ui_state.get("temperature_vision", cfg["temperature_vision"]), step=0.05, label="Vision temperature")
                    temperature_motion = gr.Slider(0, 1, value=ui_state.get("temperature_motion", cfg["temperature_motion"]), step=0.05, label="Motion temperature")
                    temperature_writer = gr.Slider(0, 1, value=ui_state.get("temperature_writer", cfg["temperature_writer"]), step=0.05, label="Writer temperature")
                    num_ctx = gr.Number(value=ui_state.get("num_ctx", cfg["num_ctx"]), precision=0, label="Context length")
                    keep_alive = gr.Textbox(value=ui_state.get("keep_alive", cfg["keep_alive"]), label="Ollama keep_alive")
                    ollama_status = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=1):
                timing_display = gr.Markdown(value=ui_state.get("timing_display", "⏱️ **Tiempos de ejecución:** *(Aún no se ha generado ningún prompt)*"))
                analysis = gr.Code(label="Vision analysis (JSON)", language="json", lines=18, value=ui_state.get("analysis", ""))
                motion_plan = gr.Code(label="Motion Director plan (JSON)", language="json", lines=18, value=ui_state.get("motion_plan", ""))
                prompt = gr.Textbox(label="Final prompt", lines=20, value=ui_state.get("prompt", ""), buttons=["copy"])
                with gr.Row():
                    save_p = gr.Button("Save prompt")
                    save_a = gr.Button("Save analysis")
                    save_m = gr.Button("Save motion plan")

                with gr.Accordion("Evolve / Prompt Transmuter", open=False):
                    evolve_mode = gr.Dropdown(
                        choices=["words", "internal", "synonyms"],
                        value=ui_state.get("evolve_mode", "words"),
                        label="Mode",
                    )
                    evolve_mode_info = gr.Markdown(
                        "**Words**: cinematic vocabulary injection · **Words from prompt**: reuse words already in the prompt · **Cinematic synonyms**: synonym substitution"
                    )
                    evolve_strength = gr.Slider(
                        minimum=0, maximum=100, step=1, value=ui_state.get("evolve_strength", 10),
                        label="Probability (% of tokens changed)",
                    )
                    with gr.Row():
                        evolve_count = gr.Number(
                            value=ui_state.get("evolve_count", 4), minimum=1, maximum=16, precision=0,
                            label="Variants",
                        )
                        evolve_seed = gr.Number(
                            value=ui_state.get("evolve_seed", 42), precision=0, label="Seed",
                        )
                    with gr.Row():
                        btn_evolve = gr.Button("Transmute", variant="primary")
                        btn_evolve_copy = gr.Button("Copy")
                        btn_evolve_send = gr.Button("Send to Scene", variant="secondary")
                    evolve_output = gr.Textbox(
                        label="Variants",
                        lines=10,
                        placeholder="Variants will appear here…",
                        interactive=False,
                    )

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
            gr.Markdown("## Director Mode\nPlan a 20–180 second sequence as consecutive short H3/LTX shots (5–30s per shot) with a shared continuity bible.")
            with gr.Row():
                with gr.Column(scale=1):
                    d_scene = gr.Textbox(label="Scene / sequence brief", lines=7, value=ui_state.get("d_scene", ""))
                    saved_d_refs = ui_state.get("d_ref_state", [])
                    saved_d_items = [p for p in saved_d_refs if Path(p).exists()]
                    d_ref_state = gr.State(saved_d_items)
                    d_selected_ref_idx = gr.State(None)
                    d_file = gr.File(label="Upload / drop reference images (drop one at a time or many — they accumulate below)", file_count="multiple", file_types=["image"], type="filepath")
                    d_refs = gr.Gallery(label="Reference images library (click an image to select and delete)", show_label=True, columns=4, height=200, object_fit="cover", type="filepath", interactive=True, value=saved_d_items)
                    with gr.Row():
                        d_remove_selected = gr.Button("🗑️ Eliminar seleccionada", size="sm", interactive=False)
                        d_clear_refs = gr.Button("🧹 Borrar todas las referencias", size="sm", variant="secondary")
                    d_ref_labels = gr.Textbox(label="Reference roles / labels (optional)", lines=4, value=ui_state.get("d_ref_labels", ""), placeholder="One per line: captain.png | CHARACTER | Captain\nengine_room.jpg | LOCATION | U-29 engine room\nuniform.png | WARDROBE | Captain uniform")
                    with gr.Row():
                        d_mode = gr.Dropdown(MODES, value=ui_state.get("d_mode", "I2VA"), label="Target workflow")
                        d_total = gr.Dropdown(["20","24","30","36","40","45","48","50","54","60","75","90","120","150","180"], value=ui_state.get("d_total", "30"), label="Total duration (s)")
                        d_shot = gr.Dropdown(["5","6","8","10","12","15","18","20","25","30"], value=ui_state.get("d_shot", "8"), label="Target shot duration (s)")
                    d_preset = gr.Dropdown(PRESETS, value=ui_state.get("d_preset", "H3 Cinematic"), label="Cinematic preset")
                    d_style = gr.Textbox(label="Global style", lines=2, value=ui_state.get("d_style", PRESETS_DATA["H3 Cinematic"]["style"]))
                    d_camera = gr.Textbox(label="Camera language", lines=2, value=ui_state.get("d_camera", PRESETS_DATA["H3 Cinematic"]["camera"]))
                    d_motion = gr.Textbox(label="Global motion", lines=3, value=ui_state.get("d_motion", PRESETS_DATA["H3 Cinematic"]["motion"]))
                    d_audio = gr.Textbox(label="Global audio", lines=2, value=ui_state.get("d_audio", PRESETS_DATA["H3 Cinematic"]["audio"]))
                    d_dialogue = gr.Textbox(label="Dialogue / spoken lines", lines=3, value=ui_state.get("d_dialogue", ""))
                    d_constraints = gr.Textbox(label="Continuity rules", lines=4, value=ui_state.get("d_constraints", "Keep character identity, wardrobe, props, location geometry, lighting direction and time of day identical between shots. Maintain screen direction and eyelines unless a motivated transition changes them."))
                    d_generate = gr.Button("Build sequence + prompts", variant="primary", size="lg")
                    d_export = gr.Button("Save Director package")
                    d_saved = gr.Textbox(label="Saved files", interactive=False)
                with gr.Column(scale=1):
                    d_bible = gr.Code(label="Continuity Bible + Reference Library", language="json", lines=22, value=ui_state.get("d_bible", ""))
                    d_storyboard = gr.Code(label="Storyboard / shot plan", language="json", lines=22, value=ui_state.get("d_storyboard", ""))
                    d_prompts = gr.Textbox(label="All H3 shot prompts", lines=24, value=ui_state.get("d_prompts", ""), buttons=["copy"])
                    d_status = gr.Textbox(label="Director status", value=ui_state.get("d_status", ""), interactive=False)
            d_shot_selector = gr.Dropdown(choices=[], value=None, label="Select shot", allow_custom_value=False, interactive=False)
            d_selected_prompt = gr.Textbox(label="Selected H3 prompt", lines=14, buttons=["copy"])
            d_selected_meta = gr.Code(label="Selected shot metadata", language="json", lines=8)
            director_state = gr.State(ui_state.get("director_state", {}))

        def on_mode_change(m):
            if m == "FLF2VA":
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            elif m == "R2V":
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
            else:
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

        mode.change(on_mode_change, inputs=mode, outputs=[single_img_row, flf_img_row, r2v_img_col, r2v_task])

        def on_preset_change(p):
            p_data = apply_preset(p)
            target_mode = "FLF2VA" if p == "LTX Video FLF" else gr.update()
            return p_data[0], p_data[1], p_data[2], p_data[3], p_data[4], target_mode

        preset.change(on_preset_change, inputs=preset, outputs=[style, camera, motion, audio, negative, mode])

        def append_r2v_refs(new_files, current_list):
            new_paths = extract_paths(new_files)
            updated = list(current_list or [])
            for p in new_paths:
                if p and p not in updated and len(updated) < 6:
                    updated.append(p)
            updated = updated[:6]
            gallery_items = [(path, f"<Image_{i}>: {Path(path).name}") for i, path in enumerate(updated, 1)]
            info_md = f"**{len(updated)}/6 referencias cargadas:** " + ", ".join([f"`<Image_{i}>`" for i in range(1, len(updated) + 1)]) if updated else "*(Sube hasta 6 imágenes para R2V. Cada imagen se etiquetará automáticamente como `<Image_1>` a `<Image_6>`)*"
            return updated, gallery_items, None, info_md, None, gr.update(interactive=False, value="🗑️ Eliminar seleccionada")

        def on_r2v_selected(current_list, evt: gr.SelectData):
            idx = evt.index
            if isinstance(idx, (int, float)) and 0 <= int(idx) < len(current_list or []):
                tag = f"<Image_{int(idx)+1}>"
                fname = Path(current_list[int(idx)]).name
                return int(idx), gr.update(interactive=True, value=f"🗑️ Eliminar {tag} ({fname})")
            return None, gr.update(interactive=False, value="🗑️ Eliminar seleccionada")

        def remove_selected_r2v(selected_idx, current_list):
            updated = list(current_list or [])
            if selected_idx is not None and isinstance(selected_idx, (int, float)) and 0 <= int(selected_idx) < len(updated):
                updated.pop(int(selected_idx))
            gallery_items = [(path, f"<Image_{i}>: {Path(path).name}") for i, path in enumerate(updated, 1)]
            info_md = f"**{len(updated)}/6 referencias cargadas:** " + ", ".join([f"`<Image_{i}>`" for i in range(1, len(updated) + 1)]) if updated else "*(Sube hasta 6 imágenes para R2V. Cada imagen se etiquetará automáticamente como `<Image_1>` a `<Image_6>`)*"
            return updated, gallery_items, None, info_md, None, gr.update(interactive=False, value="🗑️ Eliminar seleccionada")

        def clear_all_r2v():
            return [], [], None, "*(Sube hasta 6 imágenes para R2V. Cada imagen se etiquetará automáticamente como `<Image_1>` a `<Image_6>`)*", None, gr.update(interactive=False, value="🗑️ Eliminar seleccionada")

        r2v_file.change(append_r2v_refs, inputs=[r2v_file, r2v_state], outputs=[r2v_state, r2v_gallery, r2v_file, r2v_tags_info, r2v_selected_idx, r2v_remove_btn])
        r2v_gallery.select(on_r2v_selected, inputs=[r2v_state], outputs=[r2v_selected_idx, r2v_remove_btn])
        r2v_remove_btn.click(remove_selected_r2v, inputs=[r2v_selected_idx, r2v_state], outputs=[r2v_state, r2v_gallery, r2v_file, r2v_tags_info, r2v_selected_idx, r2v_remove_btn])
        r2v_clear_btn.click(clear_all_r2v, outputs=[r2v_state, r2v_gallery, r2v_file, r2v_tags_info, r2v_selected_idx, r2v_remove_btn])
        r2v_video_enabled.change(lambda v: gr.update(visible=v), inputs=r2v_video_enabled, outputs=r2v_video_box)

        def on_r2v_video_uploaded(file_value):
            path = extract_paths(file_value)[0] if file_value else None
            return path, True, gr.update(visible=True)
        r2v_video_file.upload(on_r2v_video_uploaded, inputs=r2v_video_file, outputs=[r2v_video_file, r2v_video_enabled, r2v_video_box])

        generate.click(do_generate,
            inputs=[single_image, flf_first, flf_last, r2v_state, r2v_video_enabled, r2v_video_file, r2v_video_notes, scene, mode, duration, aspect, preset, style, camera, motion, audio, dialogue, constraints, negative,
                    motion_enabled, backend_sel, ollama_url, vision_model, writer_model, motion_model, temperature_writer, temperature_vision,
                    temperature_motion, num_ctx, keep_alive, title, r2v_task],
            outputs=[timing_display, analysis, motion_plan, prompt, ollama_status, history])
        def sync_compare_choices(items):
            choices = items if isinstance(items, list) else []
            return gr.update(choices=choices), gr.update(choices=choices)
        generate.click(lambda _unused: sync_compare_choices(history_choices(load_history())), inputs=prompt, outputs=[compare_a, compare_b])
        refresh_models.click(refresh_ollama_models, inputs=[ollama_url, vision_model, writer_model, motion_model, backend_sel], outputs=[vision_model, writer_model, motion_model, ollama_status])
        check.click(check_ollama, inputs=[ollama_url, backend_sel], outputs=ollama_status)
        rescan_btn.click(rescan_llamacpp_models, outputs=[ollama_status]).then(
            refresh_ollama_models,
            inputs=[ollama_url, vision_model, writer_model, motion_model, backend_sel],
            outputs=[vision_model, writer_model, motion_model, ollama_status]
        )
        app.load(refresh_ollama_models, inputs=[ollama_url, vision_model, writer_model, motion_model, backend_sel], outputs=[vision_model, writer_model, motion_model, ollama_status])
        save_p.click(lambda x: save_text("prompt", x), inputs=prompt, outputs=saved)
        save_a.click(lambda x: save_text("analysis", x), inputs=analysis, outputs=saved)
        save_m.click(lambda x: save_text("motion", x), inputs=motion_plan, outputs=saved)

        btn_evolve.click(
            generate_evolved,
            inputs=[scene, evolve_mode, evolve_strength, evolve_count, evolve_seed],
            outputs=evolve_output,
        )
        btn_evolve_send.click(
            extract_first_variant,
            inputs=evolve_output,
            outputs=scene,
        )
        evolve_copy_js = """
        (text) => {
            if (!text) return "";
            navigator.clipboard.writeText(text).then(() => {}, () => {});
            return text;
        }
        """
        btn_evolve_copy.click(None, inputs=evolve_output, outputs=evolve_output, js=evolve_copy_js)

        restore.click(restore_history, inputs=history,
            outputs=[scene, prompt, analysis, mode, duration, aspect, preset, style, camera, motion, audio, dialogue, constraints, motion_enabled, motion_plan, title, timing_display])
        compare.click(compare_history, inputs=[compare_a, compare_b], outputs=comparison)
        def clear_hist():
            save_history([])
            return gr.update(choices=[], value=None), gr.update(choices=[], value=None), gr.update(choices=[], value=None)
        delete_history.click(clear_hist, outputs=[history, compare_a, compare_b])
        d_preset.change(lambda p: apply_preset(p)[:4], inputs=d_preset, outputs=[d_style, d_camera, d_motion, d_audio])

        def generate_ref_labels_text(paths: list[str], current_labels_text: str = "") -> str:
            existing_lines_by_filename = {}
            for line in (current_labels_text or "").splitlines():
                line_str = line.strip()
                if not line_str:
                    continue
                parts = [x.strip() for x in line_str.split("|", 2)]
                if parts:
                    filename = Path(parts[0]).name.lower()
                    existing_lines_by_filename[filename] = line_str

            new_lines = []
            is_two_file_initial = (len(paths) == 2 and not existing_lines_by_filename)

            for idx, p in enumerate(paths):
                fname = Path(p).name
                fname_lower = fname.lower()
                if fname_lower in existing_lines_by_filename:
                    new_lines.append(existing_lines_by_filename[fname_lower])
                else:
                    stem = Path(p).stem.replace("_", " ").replace("-", " ")
                    stem_lower = stem.lower()
                    if is_two_file_initial:
                        role = "FIRST_FRAME" if idx == 0 else "LAST_FRAME"
                    elif any(k in stem_lower for k in ["char", "person", "man", "woman", "captain", "actor", "hero", "guy", "girl"]):
                        role = "CHARACTER"
                    elif any(k in stem_lower for k in ["loc", "room", "env", "place", "bg", "background", "stage", "city", "house", "interior", "exterior"]):
                        role = "LOCATION"
                    elif any(k in stem_lower for k in ["prop", "item", "object", "car", "weapon", "gun", "sword"]):
                        role = "PROP"
                    elif any(k in stem_lower for k in ["start", "first", "frame1", "begin"]):
                        role = "FIRST_FRAME"
                    elif any(k in stem_lower for k in ["end", "last", "frame2", "final"]):
                        role = "LAST_FRAME"
                    else:
                        role = "GENERAL"
                    new_lines.append(f"{fname} | {role} | {stem.capitalize()}")

            return "\n".join(new_lines)

        def append_d_refs(new_files, current_list, current_labels):
            new_paths = extract_paths(new_files)
            updated = list(current_list or [])
            for p in new_paths:
                if p and p not in updated:
                    updated.append(p)
            labels_text = generate_ref_labels_text(updated, current_labels)
            return updated, updated, None, labels_text, None, gr.update(interactive=False, value="🗑️ Eliminar seleccionada")

        def on_ref_selected(current_list, evt: gr.SelectData):
            idx = evt.index
            if isinstance(idx, (int, float)) and 0 <= int(idx) < len(current_list or []):
                fname = Path(current_list[int(idx)]).name
                return int(idx), gr.update(interactive=True, value=f"🗑️ Eliminar '{fname}'")
            return None, gr.update(interactive=False, value="🗑️ Eliminar seleccionada")

        def remove_selected_ref(selected_idx, current_list, current_labels):
            updated = list(current_list or [])
            if selected_idx is not None and isinstance(selected_idx, (int, float)) and 0 <= int(selected_idx) < len(updated):
                updated.pop(int(selected_idx))
            labels_text = generate_ref_labels_text(updated, current_labels)
            return updated, updated, None, labels_text, None, gr.update(interactive=False, value="🗑️ Eliminar seleccionada")

        def clear_all_refs():
            return [], [], None, "", None, gr.update(interactive=False, value="🗑️ Eliminar seleccionada")

        d_file.upload(append_d_refs, inputs=[d_file, d_ref_state, d_ref_labels], outputs=[d_refs, d_ref_state, d_file, d_ref_labels, d_selected_ref_idx, d_remove_selected])
        d_refs.upload(append_d_refs, inputs=[d_refs, d_ref_state, d_ref_labels], outputs=[d_refs, d_ref_state, d_file, d_ref_labels, d_selected_ref_idx, d_remove_selected])
        d_refs.select(on_ref_selected, inputs=[d_ref_state], outputs=[d_selected_ref_idx, d_remove_selected])
        d_remove_selected.click(remove_selected_ref, inputs=[d_selected_ref_idx, d_ref_state, d_ref_labels], outputs=[d_refs, d_ref_state, d_file, d_ref_labels, d_selected_ref_idx, d_remove_selected])
        d_clear_refs.click(clear_all_refs, outputs=[d_refs, d_ref_state, d_file, d_ref_labels, d_selected_ref_idx, d_remove_selected])

        def run_director(*args):
            t_dir_start = time.perf_counter()
            ref, result = director_generate(*args)
            t_dir = time.perf_counter() - t_dir_start
            choices=[p["shot_id"] for p in result.get("prompts",[])]
            first=result.get("prompts",[])[0] if result.get("prompts") else {}
            meta={k:first.get(k) for k in ["shot_id","start_time","end_time","duration"]} if first else {}
            selector_update = gr.update(
                choices=choices,
                value=choices[0] if choices else None,
                interactive=bool(choices),
            )
            # Persist director UI state.
            save_director_ui_state({
                "d_scene": args[0], "d_mode": args[1], "d_ref_state": args[2], "d_ref_labels": args[3],
                "d_total": args[4], "d_shot": args[5], "d_preset": args[6],
                "d_style": args[7], "d_camera": args[8], "d_motion": args[9],
                "d_audio": args[10], "d_dialogue": args[11], "d_constraints": args[12],
                "director_state": result,
                "d_bible": json.dumps({"reference_library": ref}, indent=2, ensure_ascii=False) if ref else "No reference images.",
                "d_storyboard": json.dumps(result.get("storyboard", []), indent=2, ensure_ascii=False),
                "d_prompts": director_prompt_text(result),
                "d_status": f"Director OK — {result.get('shot_count',0)} shots / {result.get('total_duration',0)}s (⏱️ {t_dir:.1f}s)",
            })
            return (json.dumps({"reference_library": ref},indent=2,ensure_ascii=False) if ref else "No reference images.",
                    json.dumps(result.get("storyboard",[]),indent=2,ensure_ascii=False),director_prompt_text(result),
                    f"Director OK — {result.get('shot_count',0)} shots / {result.get('total_duration',0)}s (⏱️ {t_dir:.1f}s)",selector_update,
                    first.get("prompt","") if first else "",json.dumps(meta,indent=2,ensure_ascii=False),result)
        d_generate.click(run_director,
            inputs=[d_scene,d_mode,d_ref_state,d_ref_labels,d_total,d_shot,d_preset,d_style,d_camera,d_motion,d_audio,d_dialogue,d_constraints,
                    backend_sel,ollama_url,vision_model,motion_model,writer_model,temperature_vision,temperature_motion,temperature_writer,num_ctx,keep_alive],
            outputs=[d_bible,d_storyboard,d_prompts,d_status,d_shot_selector,d_selected_prompt,d_selected_meta,director_state])
        def show_director_shot(choice,result):
            for p in (result or {}).get("prompts",[]):
                if p.get("shot_id")==choice:
                    meta={k:p.get(k) for k in ["shot_id","start_time","end_time","duration"]}
                    return p.get("prompt","") , json.dumps(meta,indent=2,ensure_ascii=False)
            return "", "{}"
        d_shot_selector.change(show_director_shot,inputs=[d_shot_selector,director_state],outputs=[d_selected_prompt,d_selected_meta])
        d_export.click(save_director,inputs=director_state,outputs=d_saved)

        # If a director sequence was restored, populate the shot selector so the user can browse shots immediately.
        if ui_state.get("director_state"):
            d_shots = ui_state["director_state"].get("prompts", [])
            if d_shots:
                d_choices = [p["shot_id"] for p in d_shots]
                d_shot_selector.choices = d_choices
                d_shot_selector.value = d_choices[0]
                d_shot_selector.interactive = True

        # ---- Save UI state reactively on any meaningful input change ----
        # We define one snapshot function that collects every main-tab value.
        main_inputs_for_save = [
            single_image, flf_first, flf_last, r2v_state, r2v_video_enabled, r2v_video_file, r2v_video_notes,
            scene, mode, duration, aspect, preset, style, camera, motion, audio, dialogue, constraints,
            negative, motion_enabled, title, backend_sel, ollama_url, vision_model, writer_model, motion_model,
            temperature_vision, temperature_motion, temperature_writer, num_ctx, keep_alive, r2v_task,
            evolve_mode, evolve_strength, evolve_count, evolve_seed,
        ]
        def _make_main_state_dict(*args):
            return {
                "single_image": args[0], "flf_first": args[1], "flf_last": args[2], "r2v_state": args[3],
                "r2v_video_enabled": args[4], "r2v_video_file": args[5], "r2v_video_notes": args[6],
                "scene": args[7], "mode": args[8], "duration": args[9], "aspect": args[10], "preset": args[11],
                "style": args[12], "camera": args[13], "motion": args[14], "audio": args[15], "dialogue": args[16],
                "constraints": args[17], "negative": args[18], "motion_enabled": args[19], "title": args[20],
                "backend": args[21], "ollama_url": args[22], "vision_model": args[23], "writer_model": args[24],
                "motion_model": args[25],
                "temperature_vision": args[26], "temperature_motion": args[27], "temperature_writer": args[28],
                "num_ctx": args[29], "keep_alive": args[30], "r2v_task": args[31],
                "evolve_mode": args[32], "evolve_strength": args[33], "evolve_count": args[34], "evolve_seed": args[35],
            }
        def _save_main_from_inputs(*args):
            save_main_ui_state(_make_main_state_dict(*args))
            return None

        # Image uploads must also be persisted explicitly because gr.Image change
        # events do not always propagate the new path into the global snapshot in time.
        single_image.upload(_save_main_from_inputs, inputs=main_inputs_for_save)
        flf_first.upload(_save_main_from_inputs, inputs=main_inputs_for_save)
        flf_last.upload(_save_main_from_inputs, inputs=main_inputs_for_save)
        single_image.change(_save_main_from_inputs, inputs=main_inputs_for_save)
        flf_first.change(_save_main_from_inputs, inputs=main_inputs_for_save)
        flf_last.change(_save_main_from_inputs, inputs=main_inputs_for_save)

        # Persist whenever R2V references are added or removed.
        def append_r2v_refs_and_save(single_img, ff, fl, old_r2v_state, new_files, current_list, *rest):
            updated, gallery_items, _, info_md, _, btn_update = append_r2v_refs(new_files, current_list)
            args = (single_img, ff, fl, updated) + rest
            _save_main_from_inputs(*args)
            return updated, gallery_items, None, info_md, None, btn_update
        r2v_file.change(append_r2v_refs_and_save, inputs=main_inputs_for_save[:3] + [r2v_state, r2v_file, r2v_state] + main_inputs_for_save[4:], outputs=[r2v_state, r2v_gallery, r2v_file, r2v_tags_info, r2v_selected_idx, r2v_remove_btn])

        def remove_selected_r2v_and_save(single_img, ff, fl, old_r2v_state, selected_idx, current_list, *rest):
            updated, gallery_items, _, info_md, _, btn_update = remove_selected_r2v(selected_idx, current_list)
            args = (single_img, ff, fl, updated) + rest
            _save_main_from_inputs(*args)
            return updated, gallery_items, None, info_md, None, btn_update
        r2v_remove_btn.click(remove_selected_r2v_and_save, inputs=main_inputs_for_save[:3] + [r2v_state, r2v_selected_idx, r2v_state] + main_inputs_for_save[4:], outputs=[r2v_state, r2v_gallery, r2v_file, r2v_tags_info, r2v_selected_idx, r2v_remove_btn])

        def clear_all_r2v_and_save(single_img, ff, fl, old_r2v_state, *rest):
            updated, gallery_items, _, info_md, _, btn_update = clear_all_r2v()
            args = (single_img, ff, fl, updated) + rest
            _save_main_from_inputs(*args)
            return updated, gallery_items, None, info_md, None, btn_update
        r2v_clear_btn.click(clear_all_r2v_and_save, inputs=main_inputs_for_save[:3] + [r2v_state] + main_inputs_for_save[4:], outputs=[r2v_state, r2v_gallery, r2v_file, r2v_tags_info, r2v_selected_idx, r2v_remove_btn])

        # Video reference persistence.
        r2v_video_enabled.change(_save_main_from_inputs, inputs=main_inputs_for_save)
        r2v_video_file.upload(_save_main_from_inputs, inputs=main_inputs_for_save)
        r2v_video_file.change(_save_main_from_inputs, inputs=main_inputs_for_save)

        # Text / dropdown / slider listeners.
        for comp in (scene, mode, duration, aspect, preset, style, camera, motion, audio, dialogue,
                     constraints, negative, motion_enabled, title, ollama_url, backend_sel,
                     temperature_vision, temperature_motion, temperature_writer, num_ctx, keep_alive, r2v_task,
                     evolve_mode, evolve_strength, evolve_count, evolve_seed):
            try:
                comp.change(_save_main_from_inputs, inputs=main_inputs_for_save)
            except Exception:
                pass
        for comp in (vision_model, motion_model, writer_model):
            try:
                comp.change(_save_main_from_inputs, inputs=main_inputs_for_save)
            except Exception:
                pass

        def on_backend_change(new_backend):
            default_url = "http://127.0.0.1:8080" if new_backend == "llamacpp" else "http://127.0.0.1:11434"
            return (gr.update(value=default_url),
                    gr.Dropdown(choices=[], value=None),
                    gr.Dropdown(choices=[], value=None),
                    gr.Dropdown(choices=[], value=None),
                    "")

        backend_sel.change(on_backend_change, inputs=backend_sel,
                           outputs=[ollama_url, vision_model, writer_model, motion_model, ollama_status])
        backend_sel.change(refresh_ollama_models, inputs=[ollama_url, vision_model, writer_model, motion_model, backend_sel],
                           outputs=[vision_model, writer_model, motion_model, ollama_status])

        # ---- Director Mode state persistence ----
        director_inputs_for_save = [
            d_scene, d_ref_state, d_ref_labels, d_mode, d_total, d_shot, d_preset, d_style, d_camera,
            d_motion, d_audio, d_dialogue, d_constraints,
        ]
        def _make_director_state_dict(*args):
            return {
                "d_scene": args[0], "d_ref_state": args[1], "d_ref_labels": args[2], "d_mode": args[3],
                "d_total": args[4], "d_shot": args[5], "d_preset": args[6], "d_style": args[7],
                "d_camera": args[8], "d_motion": args[9], "d_audio": args[10], "d_dialogue": args[11],
                "d_constraints": args[12],
            }
        def _save_director_from_inputs(*args):
            save_director_ui_state(_make_director_state_dict(*args))
            return None

        # Persist director reference images when uploaded or removed.
        def append_d_refs_and_save(new_files, current_list, current_labels, *rest):
            updated, gallery, _, labels, _, btn = append_d_refs(new_files, current_list, current_labels)
            args = (updated,) + rest
            _save_director_from_inputs(*args)
            return updated, gallery, None, labels, None, btn
        d_file.upload(append_d_refs_and_save, inputs=[d_file, d_ref_state, d_ref_labels] + director_inputs_for_save[3:], outputs=[d_refs, d_ref_state, d_file, d_ref_labels, d_selected_ref_idx, d_remove_selected])
        d_refs.upload(append_d_refs_and_save, inputs=[d_refs, d_ref_state, d_ref_labels] + director_inputs_for_save[3:], outputs=[d_refs, d_ref_state, d_file, d_ref_labels, d_selected_ref_idx, d_remove_selected])

        def remove_selected_ref_and_save(selected_idx, current_list, current_labels, *rest):
            updated, gallery, _, labels, _, btn = remove_selected_ref(selected_idx, current_list, current_labels)
            args = (updated,) + rest
            _save_director_from_inputs(*args)
            return updated, gallery, None, labels, None, btn
        d_remove_selected.click(remove_selected_ref_and_save, inputs=[d_selected_ref_idx, d_ref_state, d_ref_labels] + director_inputs_for_save[3:], outputs=[d_refs, d_ref_state, d_file, d_ref_labels, d_selected_ref_idx, d_remove_selected])

        def clear_all_refs_and_save(*rest):
            updated, gallery, _, labels, _, btn = clear_all_refs()
            args = (updated,) + rest
            _save_director_from_inputs(*args)
            return updated, gallery, None, labels, None, btn
        d_clear_refs.click(clear_all_refs_and_save, inputs=director_inputs_for_save[3:], outputs=[d_refs, d_ref_state, d_file, d_ref_labels, d_selected_ref_idx, d_remove_selected])

        # Text / dropdown listeners.
        for comp in (d_scene, d_ref_labels, d_mode, d_total, d_shot, d_preset, d_style, d_camera,
                     d_motion, d_audio, d_dialogue, d_constraints):
            try:
                comp.change(_save_director_from_inputs, inputs=director_inputs_for_save)
            except Exception:
                pass

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
