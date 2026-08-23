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
* 🎵 **R2V Audio Reference Support**: Automatic embedding and preservation of functional tags like `<Audio_1>`, `<Audio_0>`, and `<Voice_1>` for Reference-to-Video workflows.
* 🎥 **Director Mode (`H3_DirectorMode`)**: Storyboard and shot sequencer for 20–180s scenes with a shared Continuity Bible across consecutive shots.
* ⚡ **Dynamic Ollama Model Selection**: Automatically queries your local Ollama instance (`/api/tags`) and populates native searchable dropdowns with smart default fallbacks.

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
| `vision_model` | COMBO | Dropdown populated with local Ollama vision models. |
| `motion_model` | COMBO | Dropdown populated with local Ollama text models. |
| `writer_model` | COMBO | Dropdown populated with local Ollama text models. |
| `first_frame_image` | IMAGE (Optional) | Starting reference image. |
| `last_frame_image` | IMAGE (Optional) | Ending reference image for First-Last Frame (FLF) interpolation. |
| `custom_*_override` | STRING (Optional) | Direct model name string or style/camera/audio overrides. |

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
