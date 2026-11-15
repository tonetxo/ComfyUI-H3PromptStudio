# Director Mode

Director Mode convierte una escena de 20–180 s en una secuencia de planos cortos generables por separado y unidos por una biblia de continuidad.

## Flujo

1. Opcionalmente analiza una galería de múltiples imágenes de referencia con etiquetas opcionales (`archivo | ROLE | Etiqueta`, asignando `<Picture 1>`, `<Picture 2>`, ...) mediante el modelo Vision seleccionado.
2. El modelo Director/Planner (el selector `Motion model` de la GUI) crea:
   - continuidad: personajes, vestuario, localización, props, iluminación, atmósfera y reglas;
   - storyboard con planos consecutivos y etiquetas `reference_tags` por plano;
   - propósito, encuadre, cámara, acción, movimiento secundario, reacciones físicas, iluminación y audio por plano.
3. El Writer model transforma cada plano en un prompt H3 autosuficiente.
4. La GUI permite seleccionar un plano y copiar/revisar únicamente ese prompt.
5. `Save Director package` guarda la secuencia actual y archiva un paquete completo por fecha (incluyendo referencias) reutilizable desde la propia pestaña.

## Parámetros

- Total: 20–180 s.
- Plano objetivo: 5–30 s.
- La herramienta calcula automáticamente el número de planos y reparte la duración total.
- Cada prompt repite los anclajes de continuidad importantes para reducir drift al generar los clips por separado.

## Ejemplo conceptual

30 s, objetivo 8 s → normalmente 4 planos:

- SHOT_01 0–8 s — establishing / geography
- SHOT_02 8–16 s — primary action
- SHOT_03 16–23 s — reaction / escalation
- SHOT_04 23–30 s — resolution / final beat

El Director puede adaptar esta estructura al contenido real de la escena.

## Archivos de salida

`outputs/director_sequence.json` y `outputs/reference_library.json` contienen el último proyecto guardado y su biblioteca de referencias.

`outputs/director_shots/<shot_id>.txt` contienen los prompts independientes listos para copiar.

`outputs/director_packages/package_<timestamp>/` archiva una copia completa de cada paquete guardado (`director_sequence.json`, `reference_library.json`, los `<shot_id>.txt` y las imágenes de referencia).
