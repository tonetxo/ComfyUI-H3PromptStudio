import base64
import io
import json
import re
import requests
try:
    import torch
    import numpy as np
except ImportError:
    torch = None
    np = None
from PIL import Image
from typing import Any, Dict, Optional, Tuple, List

# Presets and System Prompts ported faithfully from H3 Prompt Studio
MODES = ["I2VA", "FLF2VA", "T2VA", "L2VA", "R2V"]
DURATIONS = ["5s", "6s", "8s", "10s", "12s", "15s", "18s", "20s", "25s", "30s"]
ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "2.39:1"]
PRESETS = [
    "H3 Cinematic",
    "H3 Dialogue",
    "H3 Horror",
    "H3 Action",
    "Wan 2.x",
    "LTX Video",
    "LTX Video FLF",
]

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

WRITER_SYSTEM = """You are an expert multimodal video prompt engineer and cinematic director. Write one production-ready English prompt for the selected workflow, with MiniMax H3 or LTX Video as the primary target when selected. Preserve reference identity/composition where relevant. When reference tags such as <Image_1>, <Image_2>, <Audio_1>, <Audio_0>, or <Voice_1> are used, strictly preserve them as functional anchors. Make physical motion explicit, assign actions to specific subjects, use coherent temporal progression, describe camera movement, and keep audio diegetic unless requested. Do not invent major objects or characters absent from the request/reference analysis. Avoid vague filler. Output only the final prompt."""

DIRECTOR_SYSTEM = """You are a continuity-focused film director and storyboard planner for generative video. Take a scene description, optional reference-image forensic analysis, visual style, and constraints and design a sequence of consecutive video shots. The total requested duration is divided into individual shots suitable for short video generation. Every shot must be independently usable as a video-generation prompt, but all shots must preserve the continuity bible: character identity, wardrobe, props, location geometry, lighting direction/color, time of day, atmosphere, weather and visual style. Do not invent major characters, locations or objects not supported by the scene/reference. Use motivated shot changes: establish geography before action, maintain screen direction/eyelines, and only change camera position when narratively useful. Each shot should have one clear primary action plus secondary physical motion. Avoid packing unrelated actions into the same short shot. Return valid JSON only with: project_title, continuity_bible, global_style, global_audio, shots. continuity_bible must contain: characters, wardrobe, location, props, lighting, atmosphere, camera_language, continuity_rules. Each shot must contain: shot_id, start_time, end_time, duration, purpose, framing, camera, subject_action, secondary_motion, environment_reaction, lighting, audio, dialogue, transition_note, prompt_notes."""

DIRECTOR_PROMPT_SYSTEM = """You are a cinematic video prompt writer. Convert one storyboard shot plus the continuity bible into ONE production-ready English prompt for MiniMax H3 or LTX Video. The prompt must restate the critical continuity anchors needed for this shot, then describe framing, camera movement, explicit subject action, secondary physical motion, environmental reactions, lighting and diegetic audio. If the workflow is First-Last Frame (FLF2VA / LTX Video FLF), explicitly detail the visible physical trajectory from the initial frame (FIRST_FRAME) to the ending frame (LAST_FRAME). Use temporal progression only when helpful and keep each shot's action coherent. Do not invent changes to wardrobe, location, character appearance, lighting or props. Do not add music or dialogue unless explicitly specified. This prompt will be generated independently from neighboring shots, so it must be self-contained while remaining consistent with the continuity bible. Output ONLY the prompt text."""


def get_ollama_models(url: str = "http://127.0.0.1:11434") -> List[str]:
    """Fetch all installed models from local Ollama instance dynamically."""
    try:
        r = requests.get(url.rstrip("/") + "/api/tags", timeout=3)
        if r.ok:
            models = [m.get("name", "").strip() for m in r.json().get("models", [])]
            models = sorted([m for m in models if m])
            if models:
                return models
    except Exception:
        pass
    return [
        "llama3.2-vision:latest",
        "qwen2.5:latest",
        "mistral:latest",
        "llama3.2:latest",
        "llava:latest",
    ]


def pick_default_models(models: List[str]) -> Tuple[str, str, str]:
    """Pick sensible defaults for vision, motion, and writer roles."""
    if not models:
        return ("", "", "")
    # Default vision: search for vl, vision, or llava
    vision_default = models[0]
    for m in models:
        m_lower = m.lower()
        if "vision" in m_lower or "-vl" in m_lower or "llava" in m_lower:
            vision_default = m
            break

    # Default text / writer / motion
    text_default = models[0]
    for m in models:
        m_lower = m.lower()
        if "qwen" in m_lower or "gemma" in m_lower or "ministral" in m_lower or "deepseek" in m_lower or "llama" in m_lower:
            if "vision" not in m_lower and "-vl" not in m_lower:
                text_default = m
                break

    return vision_default, text_default, text_default


def tensor_to_base64(img_tensor) -> Optional[str]:
    """Convert ComfyUI Image Tensor [B, H, W, C] to Base64 JPEG string."""
    if img_tensor is None:
        return None
    try:
        if torch is not None and isinstance(img_tensor, torch.Tensor):
            if img_tensor.ndim == 4:
                img_tensor = img_tensor[0]
            i = 255.0 * img_tensor.cpu().numpy()
            img_np = np.clip(i, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np)
        elif isinstance(img_tensor, Image.Image):
            img = img_tensor
        else:
            return None
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode("ascii")
    except Exception as e:
        print(f"[H3 Prompt Studio] Image conversion error: {e}")
        return None


def extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {"raw_text": text}


def ollama_chat(
    url: str,
    model: str,
    messages: list,
    temperature: float = 0.2,
    num_ctx: int = 32768,
    keep_alive: str = "20m",
) -> str:
    if not model:
        raise ValueError("[H3 Prompt Studio] Model name is required.")
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": float(temperature), "num_ctx": int(num_ctx)},
        "keep_alive": keep_alive,
    }
    try:
        r = requests.post(url.rstrip("/") + "/api/chat", json=payload, timeout=900)
    except requests.RequestException as e:
        raise RuntimeError(f"[H3 Prompt Studio] Failed to connect to Ollama ({url}): {e}") from e
    if not r.ok:
        try:
            err = r.json().get("error", r.text)
        except Exception:
            err = r.text
        raise RuntimeError(f"[H3 Prompt Studio] Ollama HTTP {r.status_code} for '{model}': {err}")
    try:
        return r.json()["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"[H3 Prompt Studio] Unexpected Ollama response: {r.text[:500]}") from e


def build_writer_prompt(
    scene: str,
    analysis: Optional[Dict[str, Any]],
    motion_plan: Optional[Dict[str, Any]],
    mode: str,
    duration: str,
    aspect: str,
    style: str,
    camera: str,
    motion: str,
    audio: str,
    dialogue: str,
    constraints: str,
    negative: str,
    preset: str,
) -> str:
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

    r2v_requirement = ""
    if mode == "R2V" or "<Audio" in (audio or "") or "<Audio" in (scene or ""):
        r2v_requirement = """
CRITICAL R2V REQUIREMENT:
This generation is in R2V (Reference-to-Video) mode. You MUST explicitly embed the reference anchor `<Audio_1>` (or `<Audio_0>` if specified) into the final prompt (e.g., 'Action, diegetic sound design and rhythm are precisely synchronized with <Audio_1>'). If visual reference analysis is present, also reference <Image_1>. NEVER omit the <Audio_1> reference tag in the output.
"""

    return f"""Create one final production-ready prompt. PRIMARY TARGET: MiniMax H3 / Video Gen. WORKFLOW: {mode}. DURATION: {duration}. ASPECT RATIO: {aspect}. PRESET: {preset}.

USER SCENE / INTENT:
{scene or '(none)'}

VISUAL REFERENCE ANALYSIS:
{analysis_text}

MOTION DIRECTOR PLAN:
{motion_text}

STYLE:
{style or preset_data['style']}

CAMERA:
{camera or preset_data['camera']}

MOTION PRIORITIES:
{motion or preset_data['motion']}

AUDIO:
{audio_text}

DIALOGUE:
{dialogue or '(none unless explicitly requested)'}

CONTINUITY / CONSTRAINTS:
{constraints or '(preserve identity, wardrobe, props, geometry and lighting logic)'}

AVOID:
{negative or preset_data['negative']}
{r2v_requirement}
Write a coherent temporal sequence rather than a keyword list. Make important movements observable and causally connected. Keep the scene spatially consistent. Output only the final English prompt, with no markdown fences and no explanation."""



# =============================================================================
# COMFYUI NODE: H3_PromptStudio_Unified (All-in-One Fast Node with Dual-Frame FLF)
# =============================================================================
class H3_PromptStudio_Unified:
    @classmethod
    def INPUT_TYPES(cls):
        models = get_ollama_models()
        def_vision, def_motion, def_writer = pick_default_models(models)

        return {
            "required": {
                "scene_intent": ("STRING", {"multiline": True, "default": "A character transitioning from standing still to running forwards dynamically."}),
                "preset": (PRESETS, {"default": "H3 Cinematic"}),
                "workflow_mode": (MODES, {"default": "I2VA"}),
                "duration": (DURATIONS, {"default": "10s"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9"}),
                "enable_motion_director": ("BOOLEAN", {"default": True}),
                "vision_model": (models, {"default": def_vision}),
                "motion_model": (models, {"default": def_motion}),
                "writer_model": (models, {"default": def_writer}),
                "temp_vision": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05}),
                "temp_motion": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                "temp_writer": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "num_ctx": ("INT", {"default": 32768, "min": 2048, "max": 131072, "step": 1024}),
            },
            "optional": {
                "first_frame_image": ("IMAGE",),
                "last_frame_image": ("IMAGE",),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "custom_vision_override": ("STRING", {"default": ""}),
                "custom_motion_override": ("STRING", {"default": ""}),
                "custom_writer_override": ("STRING", {"default": ""}),
                "custom_style": ("STRING", {"multiline": False, "default": ""}),
                "custom_camera": ("STRING", {"multiline": False, "default": ""}),
                "custom_motion": ("STRING", {"multiline": False, "default": ""}),
                "custom_audio": ("STRING", {"multiline": False, "default": ""}),
                "custom_dialogue": ("STRING", {"multiline": False, "default": ""}),
                "custom_constraints": ("STRING", {"multiline": False, "default": ""}),
                "custom_negative": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "analysis_json", "motion_plan_json")
    FUNCTION = "generate"
    CATEGORY = "H3_PromptStudio"

    def generate(
        self,
        scene_intent,
        preset,
        workflow_mode,
        duration,
        aspect_ratio,
        enable_motion_director,
        vision_model,
        motion_model,
        writer_model,
        temp_vision,
        temp_motion,
        temp_writer,
        num_ctx,
        first_frame_image=None,
        last_frame_image=None,
        ollama_url="http://127.0.0.1:11434",
        custom_vision_override="",
        custom_motion_override="",
        custom_writer_override="",
        custom_style="",
        custom_camera="",
        custom_motion="",
        custom_audio="",
        custom_dialogue="",
        custom_constraints="",
        custom_negative="",
    ):
        v_model = custom_vision_override.strip() or vision_model
        m_model = custom_motion_override.strip() or motion_model
        w_model = custom_writer_override.strip() or writer_model

        effective_mode = workflow_mode
        if last_frame_image is not None and effective_mode == "I2VA":
            effective_mode = "FLF2VA"

        p_data = PRESETS_DATA.get(preset, PRESETS_DATA["H3 Cinematic"])
        neg_prompt = custom_negative if custom_negative.strip() else p_data["negative"]

        # Step 1: Forensic Image Analysis (Single Frame or First+Last Frame)
        analysis = None
        b64_first = tensor_to_base64(first_frame_image)
        b64_last = tensor_to_base64(last_frame_image)

        if b64_first and b64_last:
            # Dual Frame FLF analysis
            msg_first = [
                {"role": "system", "content": VISION_SYSTEM},
                {
                    "role": "user",
                    "content": "Analyze this FIRST_FRAME (Initial State) for video interpolation. Return valid JSON only.",
                    "images": [b64_first],
                },
            ]
            first_analysis = extract_json(
                ollama_chat(ollama_url, v_model, msg_first, temp_vision, num_ctx)
            )

            msg_last = [
                {"role": "system", "content": VISION_SYSTEM},
                {
                    "role": "user",
                    "content": "Analyze this LAST_FRAME (Target Ending State) for video interpolation. Return valid JSON only.",
                    "images": [b64_last],
                },
            ]
            last_analysis = extract_json(
                ollama_chat(ollama_url, v_model, msg_last, temp_vision, num_ctx)
            )

            analysis = {
                "workflow_type": "FIRST_LAST_FRAME (FLF)",
                "first_frame_analysis": first_analysis,
                "last_frame_analysis": last_analysis,
            }

        elif b64_first:
            # Single Frame analysis
            messages = [
                {"role": "system", "content": VISION_SYSTEM},
                {
                    "role": "user",
                    "content": "Analyze this reference image. Return JSON only.",
                    "images": [b64_first],
                },
            ]
            raw_analysis = ollama_chat(
                ollama_url, v_model, messages, temp_vision, num_ctx
            )
            analysis = extract_json(raw_analysis)

        elif b64_last:
            # Only last frame provided
            messages = [
                {"role": "system", "content": VISION_SYSTEM},
                {
                    "role": "user",
                    "content": "Analyze this target reference image. Return JSON only.",
                    "images": [b64_last],
                },
            ]
            raw_analysis = ollama_chat(
                ollama_url, v_model, messages, temp_vision, num_ctx
            )
            analysis = extract_json(raw_analysis)

        # Step 2: Motion Director
        motion_plan = None
        if enable_motion_director and analysis:
            if b64_first and b64_last:
                m_text = f"""FIRST-LAST FRAME (FLF) REFERENCE ANALYSIS:
FIRST_FRAME (STARTING POSE & SCENE):
{json.dumps(analysis.get('first_frame_analysis'), indent=2, ensure_ascii=False)}

LAST_FRAME (ENDING POSE & SCENE):
{json.dumps(analysis.get('last_frame_analysis'), indent=2, ensure_ascii=False)}

USER INTENT:
{scene_intent or '(none)'}

WORKFLOW: {effective_mode} (Smooth physical interpolation from FIRST_FRAME to LAST_FRAME)
DURATION: {duration}
CAMERA REQUEST: {custom_camera or '(motivate continuous camera move between both compositions)'}

Translate the difference between FIRST_FRAME and LAST_FRAME into an observable, physically coherent transition trajectory. Detail the subject momentum, cloth/hair physics, intermediate actions, lighting transitions and camera motion."""
            else:
                m_text = f"""REFERENCE ANALYSIS:
{json.dumps(analysis, indent=2, ensure_ascii=False)}

USER INTENT:
{scene_intent or '(none)'}

WORKFLOW: {effective_mode}
DURATION: {duration}
CAMERA REQUEST: {custom_camera or '(choose based on reference)'}

Translate the still image into visible, physically plausible motion. Include a compact temporal progression and secondary motion."""

            raw_motion = ollama_chat(
                ollama_url,
                m_model,
                [
                    {"role": "system", "content": MOTION_SYSTEM},
                    {"role": "user", "content": m_text},
                ],
                temp_motion,
                num_ctx,
            )
            motion_plan = extract_json(raw_motion)

        # Step 3: Cinematic Prompt Writer
        wp = build_writer_prompt(
            scene=scene_intent,
            analysis=analysis,
            motion_plan=motion_plan,
            mode=effective_mode,
            duration=duration,
            aspect=aspect_ratio,
            style=custom_style,
            camera=custom_camera,
            motion=custom_motion,
            audio=custom_audio,
            dialogue=custom_dialogue,
            constraints=custom_constraints,
            negative=custom_negative,
            preset=preset,
        )

        final_prompt = ollama_chat(
            ollama_url,
            w_model,
            [
                {"role": "system", "content": WRITER_SYSTEM},
                {"role": "user", "content": wp},
            ],
            temp_writer,
            num_ctx,
        ).strip()

        return (
            final_prompt,
            neg_prompt,
            json.dumps(analysis, indent=2, ensure_ascii=False) if analysis else "",
            json.dumps(motion_plan, indent=2, ensure_ascii=False) if motion_plan else "",
        )


# =============================================================================
# COMFYUI NODE: H3_VisionAnalyzer (Modular Step 1 with Dual FLF Support)
# =============================================================================
class H3_VisionAnalyzer:
    @classmethod
    def INPUT_TYPES(cls):
        models = get_ollama_models()
        def_vision, _, _ = pick_default_models(models)
        return {
            "required": {
                "first_frame_image": ("IMAGE",),
                "vision_model": (models, {"default": def_vision}),
                "temperature": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05}),
                "num_ctx": ("INT", {"default": 32768, "min": 2048, "max": 131072, "step": 1024}),
            },
            "optional": {
                "last_frame_image": ("IMAGE",),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "custom_model_override": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("analysis_json",)
    FUNCTION = "analyze"
    CATEGORY = "H3_PromptStudio/Modular"

    def analyze(self, first_frame_image, vision_model, temperature, num_ctx, last_frame_image=None, ollama_url="http://127.0.0.1:11434", custom_model_override=""):
        model = custom_model_override.strip() or vision_model
        b64_first = tensor_to_base64(first_frame_image)
        b64_last = tensor_to_base64(last_frame_image)

        if b64_first and b64_last:
            msg_first = [
                {"role": "system", "content": VISION_SYSTEM},
                {"role": "user", "content": "Analyze FIRST_FRAME (Initial State) for video interpolation. Return JSON only.", "images": [b64_first]},
            ]
            first_raw = ollama_chat(ollama_url, model, msg_first, temperature, num_ctx)

            msg_last = [
                {"role": "system", "content": VISION_SYSTEM},
                {"role": "user", "content": "Analyze LAST_FRAME (Target Ending State) for video interpolation. Return JSON only.", "images": [b64_last]},
            ]
            last_raw = ollama_chat(ollama_url, model, msg_last, temperature, num_ctx)

            combined = {
                "workflow_type": "FIRST_LAST_FRAME (FLF)",
                "first_frame_analysis": extract_json(first_raw),
                "last_frame_analysis": extract_json(last_raw),
            }
            return (json.dumps(combined, indent=2, ensure_ascii=False),)

        elif b64_first:
            messages = [
                {"role": "system", "content": VISION_SYSTEM},
                {"role": "user", "content": "Analyze this reference image. Return JSON only.", "images": [b64_first]},
            ]
            raw = ollama_chat(ollama_url, model, messages, temperature, num_ctx)
            parsed = extract_json(raw)
            return (json.dumps(parsed, indent=2, ensure_ascii=False),)

        return ("{}",)


# =============================================================================
# COMFYUI NODE: H3_MotionDirector (Modular Step 2)
# =============================================================================
class H3_MotionDirector:
    @classmethod
    def INPUT_TYPES(cls):
        models = get_ollama_models()
        _, def_motion, _ = pick_default_models(models)
        return {
            "required": {
                "scene_intent": ("STRING", {"multiline": True, "default": ""}),
                "workflow_mode": (MODES, {"default": "I2VA"}),
                "duration": (DURATIONS, {"default": "10s"}),
                "motion_model": (models, {"default": def_motion}),
                "temperature": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                "num_ctx": ("INT", {"default": 32768, "min": 2048, "max": 131072, "step": 1024}),
            },
            "optional": {
                "analysis_json": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "camera_request": ("STRING", {"default": ""}),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "custom_model_override": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("motion_plan_json",)
    FUNCTION = "direct_motion"
    CATEGORY = "H3_PromptStudio/Modular"

    def direct_motion(
        self,
        scene_intent,
        workflow_mode,
        duration,
        motion_model,
        temperature,
        num_ctx,
        analysis_json="",
        camera_request="",
        ollama_url="http://127.0.0.1:11434",
        custom_model_override="",
    ):
        model = custom_model_override.strip() or motion_model
        text = f"""REFERENCE ANALYSIS:
{analysis_json or '(none)'}

USER INTENT:
{scene_intent or '(none)'}

WORKFLOW: {workflow_mode}
DURATION: {duration}
CAMERA REQUEST: {camera_request or '(choose based on reference)'}

Translate the visual elements into visible, physically plausible motion. If First-Last Frame analysis is provided, describe the full physical transition trajectory."""
        raw = ollama_chat(
            ollama_url,
            model,
            [{"role": "system", "content": MOTION_SYSTEM}, {"role": "user", "content": text}],
            temperature,
            num_ctx,
        )
        parsed = extract_json(raw)
        return (json.dumps(parsed, indent=2, ensure_ascii=False),)


# =============================================================================
# COMFYUI NODE: H3_PromptWriter (Modular Step 3)
# =============================================================================
class H3_PromptWriter:
    @classmethod
    def INPUT_TYPES(cls):
        models = get_ollama_models()
        _, _, def_writer = pick_default_models(models)
        return {
            "required": {
                "scene_intent": ("STRING", {"multiline": True, "default": ""}),
                "preset": (PRESETS, {"default": "H3 Cinematic"}),
                "workflow_mode": (MODES, {"default": "I2VA"}),
                "duration": (DURATIONS, {"default": "10s"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9"}),
                "writer_model": (models, {"default": def_writer}),
                "temperature": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "num_ctx": ("INT", {"default": 32768, "min": 2048, "max": 131072, "step": 1024}),
            },
            "optional": {
                "analysis_json": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "motion_plan_json": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "custom_model_override": ("STRING", {"default": ""}),
                "custom_style": ("STRING", {"default": ""}),
                "custom_camera": ("STRING", {"default": ""}),
                "custom_motion": ("STRING", {"default": ""}),
                "custom_audio": ("STRING", {"default": ""}),
                "custom_dialogue": ("STRING", {"default": ""}),
                "custom_constraints": ("STRING", {"default": ""}),
                "custom_negative": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "write_prompt"
    CATEGORY = "H3_PromptStudio/Modular"

    def write_prompt(
        self,
        scene_intent,
        preset,
        workflow_mode,
        duration,
        aspect_ratio,
        writer_model,
        temperature,
        num_ctx,
        analysis_json="",
        motion_plan_json="",
        ollama_url="http://127.0.0.1:11434",
        custom_model_override="",
        custom_style="",
        custom_camera="",
        custom_motion="",
        custom_audio="",
        custom_dialogue="",
        custom_constraints="",
        custom_negative="",
    ):
        model = custom_model_override.strip() or writer_model
        p_data = PRESETS_DATA.get(preset, PRESETS_DATA["H3 Cinematic"])
        neg_prompt = custom_negative if custom_negative.strip() else p_data["negative"]

        analysis = extract_json(analysis_json) if analysis_json.strip() else None
        motion_plan = extract_json(motion_plan_json) if motion_plan_json.strip() else None

        wp = build_writer_prompt(
            scene=scene_intent,
            analysis=analysis,
            motion_plan=motion_plan,
            mode=workflow_mode,
            duration=duration,
            aspect=aspect_ratio,
            style=custom_style,
            camera=custom_camera,
            motion=custom_motion,
            audio=custom_audio,
            dialogue=custom_dialogue,
            constraints=custom_constraints,
            negative=custom_negative,
            preset=preset,
        )

        final_prompt = ollama_chat(
            ollama_url,
            model,
            [{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": wp}],
            temperature,
            num_ctx,
        ).strip()

        return (final_prompt, neg_prompt)


# =============================================================================
# COMFYUI NODE: H3_DirectorMode (Storyboarding & Multi-shot Sequencing)
# =============================================================================
class H3_DirectorMode:
    @classmethod
    def INPUT_TYPES(cls):
        models = get_ollama_models()
        def_vision, def_director, def_writer = pick_default_models(models)
        return {
            "required": {
                "scene_script": ("STRING", {"multiline": True, "default": "A 30-second tense bank heist negotiation scene."}),
                "total_duration": ("INT", {"default": 30, "min": 10, "max": 180, "step": 5}),
                "shot_target_duration": ("INT", {"default": 8, "min": 4, "max": 20, "step": 1}),
                "preset": (PRESETS, {"default": "H3 Cinematic"}),
                "workflow_mode": (MODES, {"default": "I2VA"}),
                "director_model": (models, {"default": def_director}),
                "writer_model": (models, {"default": def_writer}),
                "temp_director": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                "temp_writer": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "num_ctx": ("INT", {"default": 32768, "min": 2048, "max": 131072, "step": 1024}),
            },
            "optional": {
                "first_frame_image": ("IMAGE",),
                "last_frame_image": ("IMAGE",),
                "vision_model": (models, {"default": def_vision}),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "custom_director_override": ("STRING", {"default": ""}),
                "custom_writer_override": ("STRING", {"default": ""}),
                "custom_vision_override": ("STRING", {"default": ""}),
                "custom_style": ("STRING", {"default": ""}),
                "custom_constraints": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "shot_1_prompt",
        "shot_2_prompt",
        "shot_3_prompt",
        "shot_4_prompt",
        "continuity_bible",
        "sequence_json",
    )
    FUNCTION = "generate_sequence"
    CATEGORY = "H3_PromptStudio/Director"

    def generate_sequence(
        self,
        scene_script,
        total_duration,
        shot_target_duration,
        preset,
        workflow_mode,
        director_model,
        writer_model,
        temp_director,
        temp_writer,
        num_ctx,
        first_frame_image=None,
        last_frame_image=None,
        vision_model="",
        ollama_url="http://127.0.0.1:11434",
        custom_director_override="",
        custom_writer_override="",
        custom_vision_override="",
        custom_style="",
        custom_constraints="",
    ):
        v_model = custom_vision_override.strip() or vision_model
        d_model = custom_director_override.strip() or director_model
        w_model = custom_writer_override.strip() or writer_model

        ref_analysis = None
        b64_first = tensor_to_base64(first_frame_image)
        b64_last = tensor_to_base64(last_frame_image)

        if b64_first and b64_last:
            raw_v1 = ollama_chat(
                ollama_url,
                v_model,
                [{"role": "system", "content": VISION_SYSTEM}, {"role": "user", "content": "Analyze FIRST_FRAME for continuity bible. Return JSON only.", "images": [b64_first]}],
                0.15,
                num_ctx,
            )
            raw_v2 = ollama_chat(
                ollama_url,
                v_model,
                [{"role": "system", "content": VISION_SYSTEM}, {"role": "user", "content": "Analyze LAST_FRAME for continuity bible. Return JSON only.", "images": [b64_last]}],
                0.15,
                num_ctx,
            )
            ref_analysis = {
                "first_frame": extract_json(raw_v1),
                "last_frame": extract_json(raw_v2),
            }
        elif b64_first:
            raw_v1 = ollama_chat(
                ollama_url,
                v_model,
                [{"role": "system", "content": VISION_SYSTEM}, {"role": "user", "content": "Analyze this reference image. Return JSON only.", "images": [b64_first]}],
                0.15,
                num_ctx,
            )
            ref_analysis = extract_json(raw_v1)

        total = int(total_duration)
        target = int(shot_target_duration)
        count = max(2, round(total / target))
        while count > 2 and total // count < 5:
            count -= 1
        base, rem = divmod(total, count)
        durations = [base + (1 if i < rem else 0) for i in range(count)]

        planner_input = {
            "scene": scene_script,
            "target_workflow": workflow_mode,
            "reference_analysis": ref_analysis,
            "preset": preset,
            "global_style": custom_style or PRESETS_DATA.get(preset, {}).get("style", ""),
            "constraints": custom_constraints,
            "total_duration_seconds": total,
            "target_shot_duration_seconds": target,
            "shot_count": count,
            "shot_durations_seconds": durations,
        }

        raw_plan = ollama_chat(
            ollama_url,
            d_model,
            [
                {"role": "system", "content": DIRECTOR_SYSTEM},
                {"role": "user", "content": json.dumps(planner_input, ensure_ascii=False, indent=2)},
            ],
            temp_director,
            num_ctx,
        )
        plan = extract_json(raw_plan)
        shots = plan.get("shots", [])
        bible = plan.get("continuity_bible", {})

        shot_prompts = []
        timeline = 0
        for i, shot in enumerate(shots, 1):
            dur = int(shot.get("duration") or durations[min(i - 1, len(durations) - 1)])
            shot_id = shot.get("shot_id") or f"SHOT_{i:02d}"
            shot["shot_id"] = shot_id
            shot["duration"] = dur
            payload = {
                "target_workflow": workflow_mode,
                "workflow_target": f"MiniMax H3 ({workflow_mode})",
                "shot": shot,
                "continuity_bible": bible,
                "global_style": plan.get("global_style") or custom_style,
                "reference_analysis": ref_analysis,
                "user_constraints": custom_constraints,
            }
            prompt = ollama_chat(
                ollama_url,
                w_model,
                [
                    {"role": "system", "content": DIRECTOR_PROMPT_SYSTEM},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
                ],
                temp_writer,
                num_ctx,
            ).strip()
            shot_prompts.append(prompt)
            timeline += dur

        p1 = shot_prompts[0] if len(shot_prompts) > 0 else ""
        p2 = shot_prompts[1] if len(shot_prompts) > 1 else ""
        p3 = shot_prompts[2] if len(shot_prompts) > 2 else ""
        p4 = shot_prompts[3] if len(shot_prompts) > 3 else ""

        return (
            p1,
            p2,
            p3,
            p4,
            json.dumps(bible, indent=2, ensure_ascii=False),
            json.dumps(plan, indent=2, ensure_ascii=False),
        )


NODE_CLASS_MAPPINGS = {
    "H3_PromptStudio_Unified": H3_PromptStudio_Unified,
    "H3_VisionAnalyzer": H3_VisionAnalyzer,
    "H3_MotionDirector": H3_MotionDirector,
    "H3_PromptWriter": H3_PromptWriter,
    "H3_DirectorMode": H3_DirectorMode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3_PromptStudio_Unified": "🎬 H3 Prompt Studio (All-in-One)",
    "H3_VisionAnalyzer": "👁️ H3 Vision Forensic Analyzer",
    "H3_MotionDirector": "🏃 H3 Motion Director",
    "H3_PromptWriter": "✍️ H3 Cinematic Prompt Writer",
    "H3_DirectorMode": "🎥 H3 Director Storyboard & Sequencer",
}
