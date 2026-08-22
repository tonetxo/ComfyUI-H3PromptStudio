# ComfyUI-H3PromptStudio

Suite de nodos personalizados para **ComfyUI** basados en la arquitectura de 3 etapas de **H3 Prompt Studio** (Vision Forense → Motion Director → Cinematic Prompt Writer) y **Director Mode** (Storyboarding multi-plano con biblia de continuidad).

Permite generar prompts cinematográficos hiper-coherentes para **MiniMax H3**, **LTX Video**, **Wan 2.x** y otros generadores de video usando modelos locales en **Ollama**.

---

## 🚀 Instalación en ComfyUI

1. Copia la carpeta `ComfyUI-H3PromptStudio` a tu directorio `ComfyUI/custom_nodes/`:
   ```bash
   cp -r custom_nodes/ComfyUI-H3PromptStudio /ruta/a/tu/ComfyUI/custom_nodes/
   ```
2. Inicia o reinicia ComfyUI. Los nodos detectarán automáticamente todos los modelos de Ollama instalados en tu sistema y rellenarán los menús desplegables.

---

## 🧩 Nodos Disponibles

### 1. 🎬 `H3 Prompt Studio (All-in-One)` (`H3_PromptStudio_Unified`)
El nodo más rápido y limpio. Ejecuta la cadena completa (Vision → Motion → Writer) en una sola caja:
- **Entradas de Imagen:**
  - `first_frame_image` (`IMAGE`, opcional): Imagen de referencia inicial o estado de partida.
  - `last_frame_image` (`IMAGE`, opcional): Imagen de destino para presets First-Last Frame (**`LTX Video FLF`** o **`FLF2VA`**).
- **Parámetros:** `scene_intent`, `preset`, `workflow_mode` (`I2VA`, `FLF2VA`, `T2VA`, `L2VA`, `R2V`), `duration`, `aspect_ratio`, selectores de modelos Ollama.
- **Salida:**
  - `positive_prompt` (STRING) -> Conectar directo a `MiniMaxH3ImageToVideo`, `LTX Video` o `CLIPTextEncode`.
  - `negative_prompt` (STRING) -> Conectar al prompt negativo.
  - `analysis_json` (STRING) -> Diagnóstico forense en JSON (con análisis dual de transición si se usan 2 imágenes).
  - `motion_plan_json` (STRING) -> Plan de trayectoria física y temporal en JSON.

### 2. 👁️ `H3 Vision Forensic Analyzer` (`H3_VisionAnalyzer`)
Analiza una o dos imágenes (`first_frame_image` y `last_frame_image`) de forma forense para interpolación de video.
- **Entrada:** `first_frame_image`, `last_frame_image` (opcional), `vision_model`, `temperature`.
- **Salida:** `analysis_json`.

### 3. 🏃 `H3 Motion Director` (`H3_MotionDirector`)
Traduce la imagen (o la diferencia física entre First Frame y Last Frame) y la intención del usuario en una trayectoria de movimiento continua y verosímil.
- **Entrada:** `scene_intent`, `analysis_json`, `workflow_mode`, `duration`.
- **Salida:** `motion_plan_json`.

### 4. ✍️ `H3 Cinematic Prompt Writer` (`H3_PromptWriter`)
Compila el análisis forense, el plan de movimiento y el preset de estilo en un prompt cinematográfico en inglés listo para producción.
- **Entrada:** `scene_intent`, `analysis_json`, `motion_plan_json`, `preset`, etc.
- **Salida:** `positive_prompt`, `negative_prompt`.

### 5. 🎥 `H3 Director Storyboard & Sequencer` (`H3_DirectorMode`)
Convierte una escena larga (20–180s) en planos individuales con una **biblia de continuidad global** (personajes, vestuario, iluminación).
- **Salida:** `shot_1_prompt`, `shot_2_prompt`, `shot_3_prompt`, `shot_4_prompt`, `continuity_bible`, `sequence_json`.

---

## 📂 Workflows de Ejemplo Incluidos

- `workflows/workflow_flf_prompt_studio_ui.json`: **Workflow FLF (First-Last Frame)** con 2 nodos `LoadImage` (Frame inicial y Frame final) conectados al Prompt Studio para interpolación guiada.
- `workflows/minimax_h3_i2v_with_prompt_studio_API.json`: Pipeline completo de generación de video con MiniMax H3 I2V conectado automáticamente al Prompt Studio.
- `workflows/workflow_prompt_studio_ui.json`: Flujo visual para ComfyUI con vista previa de prompts.
