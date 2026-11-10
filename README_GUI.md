# MiniMax H3 Prompt Studio — GUI

GUI local para construir prompts de vídeo a partir de texto y referencias visuales usando Ollama o llama.cpp.

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
pip install -r requirements-gui.txt
```

El backend se elige en la GUI y debe estar accesible en la URL configurada, normalmente:

```text
Ollama:    http://127.0.0.1:11434
llama.cpp: http://127.0.0.1:8080
```

No hay modelos hardcodeados: la GUI utiliza los que liste el servidor (`/api/tags` en Ollama, `/v1/models` en llama.cpp). Sin más configuración, **Rescan llama.cpp models** vuelve a consultar el servidor.

## Ajustes locales opcionales

Para ajustes propios de cada máquina, crea `~/.config/h3ps/settings.json`. Lo leen la GUI y los nodos de `custom_nodes/ComfyUI-H3PromptStudio`, y todas las claves son opcionales:

```json
{
  "gguf_models_dir": "~/gguf-models",
  "llamacpp_rescan_command": "systemctl --user restart mi-llamacpp.service",
  "llamacpp_url": "http://127.0.0.1:8080",
  "llamacpp_extra_models": ["mi-alias"]
}
```

| Clave | Qué hace | Dónde |
|---|---|---|
| `gguf_models_dir` | Añade a la lista los `.gguf` de esta carpeta y sus subcarpetas | GUI y nodos |
| `llamacpp_rescan_command` | Comando que ejecuta el botón de reescaneo. Se lanza sin shell, así que para encadenar varios pasos usa un script | GUI y nodos |
| `llamacpp_url` | URL del servidor. La GUI usa la de su campo `URL` | Nodos |
| `llamacpp_extra_models` | Modelos que se listan siempre, para que los workflows guardados sigan siendo válidos con el servidor apagado | Nodos |

## Director Mode

La pestaña **Director Mode** permite introducir una escena de 20–60 segundos, opcionalmente una referencia visual, y dividirla automáticamente en planos cortos consecutivos. El proceso crea una **Continuity Bible**, un storyboard y un prompt H3 independiente para cada plano.

El selector `Motion model` se utiliza como **Director/Planner model** en este modo. Puedes seleccionar cualquier modelo Ollama que tengas instalado; no se presupone un modelo concreto.

Las salidas se guardan con **Save Director package** en `outputs/director_sequence.json` y `outputs/director_shots/`.

## Plano-secuencia

La pestaña **Plano-secuencia** escribe un prompt por segmento de una toma continua, para MiniMax H3 Ref y VideoConcat de SwarmUI. Usa los modelos Vision y Writer seleccionados en la pestaña principal y necesita al menos una imagen de referencia y dos duraciones.

No hay conexión con SwarmUI: los resultados se pegan a mano. **Save prompts (txt)** los guarda en `outputs/plano_secuencia/`.
