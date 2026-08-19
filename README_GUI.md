# MiniMax H3 Prompt Studio — GUI

GUI local para construir prompts de vídeo a partir de texto y referencias visuales usando Ollama.

## Arquitectura

```text
Imagen
  │
  ▼
Vision model ──► análisis visual JSON
  │
  ▼
Motion model ─► plan de movimiento físico JSON
  │
  ▼
Writer model ─► prompt final
```

## Funciones

### 1. Modelos Ollama dinámicos

La interfaz consulta `GET /api/tags` y muestra los modelos realmente instalados en Ollama. Los roles son independientes:

- Vision model: analiza la imagen.
- Motion model: convierte la imagen estática en movimiento observable.
- Writer model: redacta el prompt final.

Puedes usar el mismo modelo en varios roles. `Refresh models` vuelve a consultar Ollama.

### 2. Presets cinematográficos

Incluidos:

- H3 Cinematic
- H3 Dialogue
- H3 Horror
- H3 Action
- Wan 2.x
- LTX Video

El preset rellena estilo, cámara, prioridades de movimiento, audio y restricciones negativas. Todos los campos siguen siendo editables.

### 3. Motion Director

Con una imagen de referencia, el Motion Director genera un JSON intermedio con:

- acciones principales
- movimiento de sujetos
- movimiento ambiental
- partículas
- cambios de iluminación
- movimiento de cámara
- reacciones físicas
- beats temporales
- eventos de audio
- restricciones de movimiento

El Writer incorpora este plan al prompt final para reducir escenas estáticas y favorecer movimiento físico continuo.

### 4. Historial de versiones

Cada generación se guarda en `prompt_history.json`, hasta 100 versiones.

La GUI permite:

- seleccionar una versión
- restaurarla
- comparar dos versiones
- conservar escena, preset, parámetros y prompt
- limpiar el historial

### 5. Exportación

Los botones guardan en `outputs/`:

- `h3_prompt.txt`
- `vision_analysis.json`
- `motion_plan.json`

## Arranque

```bash
./launch_gui.sh
```

Por defecto:

```text
http://127.0.0.1:7860
```

También:

```bash
python h3_gui.py --host 0.0.0.0 --port 7860
```

## Dependencias

```bash
pip install -r requirements.txt
```

Ollama debe estar accesible en la URL configurada, normalmente:

```text
http://127.0.0.1:11434
```

No hay modelos hardcodeados: la GUI utiliza lo que encuentre instalado en Ollama.

## Director Mode

La pestaña **Director Mode** permite introducir una escena de 20–60 segundos, opcionalmente una referencia visual, y dividirla automáticamente en planos cortos consecutivos. El proceso crea una **Continuity Bible**, un storyboard y un prompt H3 independiente para cada plano.

El selector `Motion model` se utiliza como **Director/Planner model** en este modo. Puedes seleccionar cualquier modelo Ollama que tengas instalado; no se presupone un modelo concreto.

Las salidas se guardan con **Save Director package** en `outputs/director_sequence.json` y `outputs/director_shots/`.
