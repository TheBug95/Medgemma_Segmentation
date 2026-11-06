# AGENTS.md — Convenciones de Código

Para el contexto del proyecto, la arquitectura del pipeline y la relación entre el código y el paper, ver [`README.md`](README.md). Este archivo recoge únicamente las reglas que debe seguir quien escriba código aquí.

## Estructura modular

- Cada módulo vive en su carpeta `modules/` con una clase principal y su interfaz pública documentada en el docstring de cabecera del archivo.
- **NO mezclar** la lógica de un módulo con otro. Si necesitas funcionalidad de otro módulo, impórtalo y llama a su interfaz pública.
- Cada módulo debe funcionar de forma independiente y tener su propio test.

## Replicabilidad

- **Semilla global:** `SEED = 42`, definida en el `config.yaml` de cada etapa (`Stage_I/Experiments/Classifier_Selection/config.yaml`, `Stage_III/config.yaml`).
- Todo módulo debe recibir `seed` en su config y llamar a `set_global_seed(seed)` al inicializarse.
- Los splits del dataset se cargan desde `splits.json` / `split.json` — **nunca regenerarlos**.
- Usar `torch.backends.cudnn.deterministic = True`.

## Convenciones de código

- **Idioma del código:** inglés (nombres de variables, funciones, clases).
- **Idioma de comentarios:** español (explicar la lógica en español).
- **Type hints** obligatorios en todas las funciones públicas.
- **Docstrings** obligatorios en todas las clases y métodos públicos.
- Formato: `black` con línea máxima de 100 caracteres.

## Configuración

- Toda configuración va en el `config.yaml` de la etapa, **NO** hardcodeada en el código.
- Los módulos reciben su sección del config como `dict` en el constructor.
- Rutas de archivos: usar `pathlib.Path`, nunca strings crudos.

## Logging y resultados

- Usar el `logging` estándar de Python, **NO** `print()`.
- Los resultados se guardan en `results/` como JSON con estructura definida.
- Cada resultado debe incluir: timestamp, config usada, semilla y métricas.

## Qué NO hacer

1. **NO mezclar lógica entre módulos.** Cada módulo tiene su interfaz definida.
2. **NO inventar columnas o campos de datos.** Si un campo no existe en la especificación, preguntar.
3. **NO generar splits nuevos.** Siempre cargar los existentes.
4. **NO hardcodear rutas, hiperparámetros ni semillas.** Todo va en `config.yaml`.
5. **NO usar `print()`.** Usar `logging`.
