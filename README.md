# ComfyUI-H3PromptStudio

[![ComfyUI Custom Node](https://img.shields.io/badge/ComfyUI-Custom--Node-blue.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Ollama](https://img.shields.io/badge/Backend-Ollama%20Local%20LLM-black.svg)](https://ollama.com/)
[![Target Models](https://img.shields.io/badge/Target-MiniMax%20H3%20%7C%20LTX--Video%20%7C%20Wan%202.x-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**ComfyUI-H3PromptStudio** is a suite of custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that automates high-fidelity cinematic video prompt engineering for **MiniMax H3**, **LTX-Video**, **Wan 2.x**, and other generative video models using local **Ollama** LLMs.

It implements a 3-tier multimodal chain (**Forensic Vision Analysis $\rightarrow$ Motion Director $\rightarrow$ Cinematic Prompt Writer**) and multi-shot storyboard generation with a shared **Continuity Bible** (**Director Mode**).

---

## 🏗️ Architecture

```
                    ┌─────────────────────────┐          ┌─────────────────────────┐
                    │  LoadImage (First Frame)│          │  LoadImage (Last Frame) │
                    └────────────┬────────────┘          └────────────┬────────────┘
                                 │ first_frame_image                  │ last_frame_image (FLF)
                                 ▼                                    ▼
         ┌─────────────────────────────────────────────────────────────────────────┐
         │ 🎬 H3 Prompt Studio (All-in-One / Modular Nodes)                       │
         │                                                                         │
         │  1. 👁️ Forensic Vision Analysis  ──► Structured JSON (Anchors, Light)   │
         │  2. 🏃 Motion Director (Physics)  ──► Temporal Trajectory & Secondary    │
         │  3. ✍️ Cinematic Prompt Writer    ──► Final Production-Ready Video Prompt│
         └───────────────────────────────┬─────────────────────────┬───────────────┘
                                         │ positive_prompt         │ negative_prompt
                                         ▼                         ▼
                         ┌───────────────────────────────┐   ┌───────────────────┐
                         │ MiniMax H3 / LTX / Wan Nodes │   │ Negative / CLIP   │
                         └───────────────────────────────┘   └───────────────────┘
```

---

## ✨ Key Features

* 🎬 **All-in-One Node (`H3_PromptStudio_Unified`)**: Execute the entire vision-to-motion-to-prompt pipeline in a single, clean node.
* 👁️ **Forensic Visual Analysis (`H3_VisionAnalyzer`)**: Conservative visual breakdown (materials, lighting direction, composition, wardrobe, continuity anchors) without hallucinating unseen elements.
* 🏃 **Physical Motion Director (`H3_MotionDirector`)**: Converts static reference frames into observable physical motion (inertia, cloth/hair physics, fluid dynamics, camera momentum).
* ✍️ **Cinematic Prompt Writer (`H3_PromptWriter`)**: Produces structured, temporally sequenced English prompts optimized for generative video engines.
* 🎞️ **First-Last Frame (FLF) Support**: Dual image inputs (`first_frame_image` & `last_frame_image`) for calculating smooth physical transition trajectories.
* 🎵 **R2V Audio Reference Support**: Automatic embedding and preservation of functional tags like `<Audio 1>`, `<Audio 0>`, and `<Voice 1>` for Reference-to-Video workflows.
* 🎥 **Director Mode (`H3_DirectorMode`)**: Storyboard and shot sequencer for 20–180s scenes with a shared Continuity Bible across consecutive shots.
* ⚡ **Dynamic Ollama Model Selection**: Automatically queries your local Ollama instance (`/api/tags`) and populates native searchable dropdowns with smart default fallbacks.

---

## 🖥️ Web GUI (standalone)

The repo also includes **`h3_gui.py`**, a standalone Gradio app that runs without ComfyUI. It works with Ollama or llama.cpp and adds Director Mode, a Plano-secuencia tab (one continuous take in segments), prompt history and persistent form state.

```bash
git clone https://github.com/tonetxo/ComfyUI-H3PromptStudio.git
cd ComfyUI-H3PromptStudio
./launch_gui.sh
```

The script creates a `.venv`, installs `requirements-gui.txt` and opens the app, by default at `http://127.0.0.1:7860` (or the next free port). See [README_GUI.md](README_GUI.md) and [DIRECTOR_MODE.md](DIRECTOR_MODE.md).

With llama.cpp, models come from your server's `/v1/models`. Scanning a `.gguf` folder and running a command from the "Rescan" button are optional: see [Optional local settings](README_GUI.md#ajustes-locales-opcionales).

---

## 📦 Installation

### Method 1: Git Clone (Recommended)

1. Open your terminal and navigate to your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   ```
2. Clone this repository:
   ```bash
   git clone https://github.com/your-username/ComfyUI-H3PromptStudio.git
   ```
3. Install dependencies (if not already installed in your ComfyUI environment):
   ```bash
   pip install -r ComfyUI-H3PromptStudio/requirements.txt
   ```
4. Restart ComfyUI.

### Method 2: ComfyUI Manager
Search for `ComfyUI-H3PromptStudio` in the ComfyUI Manager and click **Install**.

---

## 🦙 Ollama Prerequisites

Make sure you have [Ollama](https://ollama.com/) running locally:

```bash
# Vision Model (Multimodal)
ollama pull llama3.2-vision
# or
ollama pull qwen2.5-vl  # (or qwen3-vl)

# Reasoning / Writer Models
ollama pull qwen2.5:latest
# or
ollama pull mistral:latest
```

---

## 🧩 Node Reference

### 1. 🎬 `H3 Prompt Studio (All-in-One)` (`H3_PromptStudio_Unified`)
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `scene_intent` | STRING (Multiline) | What should happen in the scene or shot description. |
| `preset` | COMBO | Cinematic style preset (`H3 Cinematic`, `Wan 2.x`, `LTX Video`, etc.). |
| `workflow_mode` | COMBO | Target workflow (`I2VA`, `FLF2VA`, `T2VA`, `L2VA`, `R2V`). |
| `duration` | COMBO | `5s`, `6s`, `8s`, `10s`, `12s`, `15s`, `18s`, `20s`, `25s`, `30s`. |
| `aspect_ratio` | COMBO | `16:9`, `9:16`, `1:1`, `4:3`, `3:4`, `21:9`, `2.39:1`. |
| `enable_motion_director`| BOOLEAN | Enable intermediate physics and secondary motion planning. |
| `vision_model` | COMBO | Ollama vision model (shown when `backend = ollama`). |
| `motion_model` | COMBO | Ollama text model for the Motion Director (shown when `backend = ollama`). |
| `writer_model` | COMBO | Ollama text model for the prompt writer (shown when `backend = ollama`). |
| `vision_model_llamacpp` / `motion_model_llamacpp` / `writer_model_llamacpp` | COMBO | Same three roles for the llama.cpp server (shown when `backend = llamacpp`). |
| `temp_vision` / `temp_motion` / `temp_writer` | FLOAT | Sampling temperature for each stage (defaults 0.15 / 0.25 / 0.35). |
| `num_ctx` | INT | Context window passed to the LLM backend (default 32768). |
| `unload_models_after_gen` | BOOLEAN (Optional, Default: True) | Unloads the stage models after generation to free VRAM. |
| `first_frame_image` | IMAGE (Optional) | Starting reference image. Maps to the functional anchor `<Picture 1>`. |
| `last_frame_image` | IMAGE (Optional) | Ending reference image. Connecting it switches `I2VA` to `FLF2VA` automatically. |
| `backend` | COMBO (Optional) | `ollama` or `llamacpp`. Switches the visible model widgets and the URL. |
| `ollama_url` | STRING (Optional) | Backend URL; set automatically from `backend`. |
| `custom_vision_override` / `custom_motion_override` / `custom_writer_override` | STRING (Optional) | Model name typed by hand; when non-empty it replaces the model selected in the dropdown for that stage. |
| `custom_style` / `custom_camera` / `custom_motion` / `custom_audio` / `custom_negative` | STRING (Optional, Multiline) | Filled with the preset's text, and updated when the preset changes as long as you have not edited them. Edited text is kept as an override; an empty field also falls back to the preset. |
| `custom_dialogue` / `custom_constraints` | STRING (Optional, Multiline) | Dialogue and continuity instructions. Filled with the default text when empty; an empty field also falls back to that default. |
| `custom_video` | STRING (Optional) | Note describing how the `<Video 1>` reference should drive the motion; enables the video-reference requirement. |

**Outputs:**
* `positive_prompt` (STRING): Final production-ready prompt.
* `negative_prompt` (STRING): Targeted negative prompt based on the chosen preset.
* `analysis_json` (STRING): Forensic visual breakdown in JSON format.
* `motion_plan_json` (STRING): Physical motion plan in JSON format.

---

### 2. 👁️ `H3 Vision Forensic Analyzer` (`H3_VisionAnalyzer`)
Performs conservative forensic visual analysis on one or two reference frames (`first_frame_image` and `last_frame_image`).
* **Outputs:** `analysis_json` (STRING).

### 3. 🏃 `H3 Motion Director` (`H3_MotionDirector`)
Translates forensic analysis and user intent into concrete, observable physical motion.
* **Outputs:** `motion_plan_json` (STRING).

### 4. ✍️ `H3 Cinematic Prompt Writer` (`H3_PromptWriter`)
Assembles forensic analysis, motion plan, and stylistic constraints into the final prompt.
* **Outputs:** `positive_prompt` (STRING), `negative_prompt` (STRING).

### 5. 🎥 `H3 Director Storyboard & Sequencer` (`H3_DirectorMode`)
Converts a 20–180 second brief into sequential short shots connected by a shared continuity bible.
* **Outputs:** `shot_1_prompt`, `shot_2_prompt`, `shot_3_prompt`, `shot_4_prompt`, `continuity_bible`, `sequence_json`.

---

## 🎨 Cinematic Presets

| Preset | Aesthetic & Directorial Priorities |
| :--- | :--- |
| **`H3 Cinematic`** | Photorealistic cinematic lighting, natural physical motion, grounded continuity, diegetic stereo sound. |
| **`H3 Dialogue`** | Natural character performance, eye-line continuity, synchronized speech and subtle facial micro-expressions. |
| **`H3 Horror`** | Oppressive atmosphere, restrained practical lighting, eerie secondary motion (smoke, dust, flickering lights). |
| **`H3 Action`** | High-intensity momentum, visible physical impact, recoil, debris dynamics, motivated camera tracking. |
| **`Wan 2.x`** | High temporal consistency, clear subject silhouettes, strong motion readability. |
| **`LTX Video`** | Explicit temporal progression, fluid environmental dynamics (particles, water, cloth, foliage). |
| **`LTX Video FLF`** | Smooth first-to-last frame trajectory interpolation, lighting continuity, deformation-free physics. |

---

## 💡 Quick Start & Node Connections

1. **Add Node**: Right-click on the ComfyUI canvas $\rightarrow$ `H3_PromptStudio` $\rightarrow$ select **`🎬 H3 Prompt Studio (All-in-One)`**.
2. **Connect Reference Frame(s)**:
   * Connect an image output from a `LoadImage` node to `first_frame_image`.
   * *(Optional for FLF transition)*: Connect a second `LoadImage` to `last_frame_image` and select `workflow_mode = FLF2VA` (or preset `LTX Video FLF`).
3. **Route Outputs**:
   * Connect `positive_prompt` $\rightarrow$ your video generator's prompt input (or `CLIPTextEncode` / MiniMax API node).
   * Connect `negative_prompt` $\rightarrow$ your negative prompt / conditioning input.
4. **Select Local Models**: Choose your installed Ollama models from the dynamic dropdowns (e.g. `llama3.2-vision` / `qwen2.5-vl` for Vision, `qwen2.5` for Motion and Writer).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
