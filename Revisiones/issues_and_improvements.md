# Errores y Puntos de Mejora — MedGemma-Seg

**Fecha:** 2026-05-13  
**Versión:** v1.0  
**Archivos analizados:** 28 (22 fuente + 6 tests)  
**Total líneas analizadas:** ~2,155

---

## Leyenda de Severidad

| Icono | Severidad | Impacto | Acción requerida |
|---|---|---|---|
| 🔴 | **Crítico** | Pipeline no ejecuta o produce resultados incorrectos | Corregir inmediatamente |
| 🟠 | **Alto** | Funcionalidad degradada, riesgo de fallo en edge cases | Corregir antes de producción |
| 🟡 | **Medio** | Code smell, inconsistencia, o falta de robustez | Corregir en siguiente sprint |
| 🟢 | **Bajo** | Mejora de calidad, refactor, o optimización | Aplicar cuando sea conveniente |

---

## 1. Errores y Problemas Funcionales

### 🔴 CRÍTICO-1: `fd_kde.py` — Mezcla NumPy/Torch inconsistente

**Archivo:** `opcion_A/fd_kde.py`  
**Línea:** 58  
**Código:**
```python
log_per_dim = log_per_dim - np.log(self.k * self.h)  # (768,)
```

**Problema:** `log_per_dim` es un `torch.Tensor` (posiblemente en CUDA), pero `np.log()` retorna un `numpy.float64` (CPU). PyTorch hace el casting implícito, pero:
1. Es inconsistente (mezcla de frameworks)
2. `np.log` con valores pequeños puede producir `-inf` sin que torch lo detecte
3. El resultado es `float64` en lugar de `float32`, duplicando memoria innecesariamente

**Corrección:**
```python
log_per_dim = log_per_dim - torch.log(torch.tensor(self.k * self.h, device=log_per_dim.device))
```

**Impacto:** Si `self.k * self.h` es 0 (imposible con k≥1) o muy pequeño, `np.log` puede comportarse diferente a `torch.log` en edge cases numéricos.

---

### 🔴 CRÍTICO-2: `ood_window.py` — División por cero en `normalize()`

**Archivo:** `opcion_A/ood_window.py`  
**Línea:** 54  
**Código:**
```python
def normalize(self, log_density: float) -> float:
    if log_density < self.theta_min:
        return 0.0
    elif log_density > self.theta_max:
        return 1.0
    return (log_density - self.theta_min) / (self.theta_max - self.theta_min)
```

**Problema:** Si `theta_min == theta_max` (caso degenerado donde todos los support embeddings tienen idéntica log-densidad), la división produce `NaN` o `ZeroDivisionError`.

**Escenario de reproducción:**
- Support set con k=6 máscaras idénticas → todas las log-densidades son iguales
- `q1 == q3` → `iqr = 0` → `theta_min == theta_max`
- Cualquier llamada a `normalize()` falla

**Corrección:**
```python
def normalize(self, log_density: float) -> float:
    if log_density < self.theta_min:
        return 0.0
    elif log_density > self.theta_max:
        return 1.0
    elif self.theta_max == self.theta_min:
        return 1.0  # Caso degenerado: todo está en el centro
    return (log_density - self.theta_min) / (self.theta_max - self.theta_min)
```

---

### 🔴 CRÍTICO-3: `simple_shot.py` — `torch.softmax` sobre cosenos sin normalización

**Archivo:** `opcion_A/simple_shot.py`  
**Línea:** 42  
**Código:**
```python
scores = torch.softmax(torch.stack([cos_pos, cos_neg]), dim=0)
```

**Problema:** `F.cosine_similarity` retorna valores en `[-1, 1]`. Si ambos cosenos son negativos (ej. query es muy diferente de ambos prototipos), `softmax` sobre valores negativos puede dar una distribución no informativa (ej. [0.52, 0.48]). Además, no hay temperature scaling, lo que hace que softmax sea muy "dura".

**Impacto:** El score de la clase positiva no es calibrado. Un query lejano de ambas clases podría dar un score alto solo por cómo funciona softmax.

**Corrección:**
```python
# Aplicar temperature scaling para calibración
T = 0.1  # temperature
scores = torch.softmax(torch.stack([cos_pos, cos_neg]) / T, dim=0)
# O alternativa: usar directamente el coseno sin softmax
score = (cos_pos + 1) / 2  # Normalizar a [0, 1]
```

---

### 🟠 ALTO-1: `fsl_fd.py` — Tensor sin device en `_calibrate()`

**Archivo:** `opcion_A/fsl_fd.py`  
**Línea:** 45  
**Código:**
```python
self.ood_window.calibrate(torch.tensor(support_log_densities))
```

**Problema:** `torch.tensor()` crea un tensor en CPU por defecto. Si `self.support_embs` está en CUDA, esto funciona (la conversión es implícita en `calibrate()`), pero es ineficiente y puede causar warnings de PyTorch sobre tensores en dispositivos diferentes.

**Corrección:**
```python
self.ood_window.calibrate(torch.tensor(support_log_densities, device=self.support_embs.device))
```

---

### 🟠 ALTO-2: `feature_extractor.py` — `pooler_output` podría no existir

**Archivo:** `opcion_A/feature_extractor.py`  
**Línea:** 72  
**Código:**
```python
emb = outputs.pooler_output.squeeze(0).float()
```

**Problema:** Algunas variantes de `SiglipVisionModel` (especialmente modelos base sin head de pooling) no tienen `pooler_output`. En ese caso, el atributo es `None` y `.squeeze(0)` fallará con `AttributeError`.

**Nota:** Se agregó un fallback a `last_hidden_state[:, 0, :]`, pero el fallback solo se activa si la dimensión no coincide, no si `pooler_output` es `None`.

**Corrección:**
```python
if outputs.pooler_output is not None:
    emb = outputs.pooler_output.squeeze(0).float()
else:
    emb = outputs.last_hidden_state[:, 0, :].squeeze(0).float()
```

---

### 🟠 ALTO-3: `text_encoder.py` — Carga duplicada del modelo vision+text

**Archivo:** `opcion_A/text_encoder.py`  
**Línea:** 33-37  
**Código:**
```python
self.model = SiglipModel.from_pretrained(
    self.model_id,
    torch_dtype=torch.bfloat16,
).to(self.device)
```

**Problema:** `SiglipModel` carga **ambos** encoders (vision + text). Pero `feature_extractor.py` ya carga `SiglipVisionModel` desde el mismo `model_id`. Esto significa:
- Vision encoder se carga 2 veces
- ~1.5 GB extra de VRAM
- Tiempo de carga duplicado

**Soluciones posibles:**

| Opción | Pros | Cons |
|---|---|---|
| A: Compartir modelo entre feature_extractor y text_encoder | Ahorra 1.5 GB VRAM | Acopla los dos módulos |
| B: Usar `SiglipTextModel` standalone (si existe) | Carga solo text encoder | No está claro si existe para medsiglip-448 |
| C: Cargar `SiglipModel` una vez y extraer sub-modelos | Limpio | Requiere refactor del constructor |

**Recomendación:** Opción C. Crear un `SharedSigLIPModel` que cargue `SiglipModel` una vez y exponga `vision_model` y `text_model`:

```python
class SharedSigLIPModel:
    _instance = None
    
    @classmethod
    def get_instance(cls, model_id="google/medsiglip-448"):
        if cls._instance is None:
            cls._instance = SiglipModel.from_pretrained(model_id, ...)
        return cls._instance
```

---

### 🟠 ALTO-4: `scoring.py` — Duplicación de lógica `s_total`

**Archivo:** `opcion_A/scoring.py`  
**Líneas:** 124-126  
**Código:**
```python
s_total = (scorer.alpha * s_fsl +
           scorer.beta * s_cos +
           scorer.gamma * s_sam)
```

**Problema:** `CompositeScorer.compute()` hace exactamente el mismo cálculo (línea 31). Hay duplicación de lógica. Si se cambian los pesos en un lugar, el otro queda inconsistente.

**Corrección:**
```python
s_total = scorer.compute(s_fsl, s_cos, s_sam)['s_total']
```

**Impacto similar en:** `multi_pathology.py` líneas 62-64 y `fd_uncertainty.py` línea 52.

---

### 🟠 ALTO-5: `fd_uncertainty.py` — `_u_dimvar()` usa umbral arbitrario no documentado

**Archivo:** `opcion_A/fd_uncertainty.py`  
**Línea:** 86  
**Código:**
```python
return (1 / (1 + torch.exp(-5 * (mean_contrib - 0.1)))).item()
```

**Problema:** El valor `0.1` es un umbral mágico sin documentación. ¿Por qué 0.1 y no 0.05 o 0.2? Este umbral determina qué tan "alta" debe ser la contribución media para considerarla baja incertidumbre.

**Recomendación:** Hacerlo configurable:
```python
DIMVAR_THRESHOLD = 0.1  # Documentar: umbral basado en análisis empírico del paper
```

---

### 🟠 ALTO-6: `multi_pathology.py` — `fd_uq` y `scorer` pueden tener pesos inconsistentes

**Archivo:** `opcion_A/multi_pathology.py`  
**Línea:** 66-67  
**Código:**
```python
u_composite, u_components = self.fd_uq.estimate(
    emb, entry['kde'], entry['ood'], entry['support_embs'],
    s_fsl, s_cos, s_sam,
)
```

**Problema:** `FDUncertaintyEstimator` recalcula `s_total` internamente con sus propios `alpha/beta/gamma` (línea 52 de fd_uncertainty.py). Si estos no coinciden con los del `scorer`, la incertidumbre U_magnitud se calcula con pesos diferentes al score compuesto real.

**Verificación:**
```python
# En fd_uncertainty.py línea 52:
s_total = self.alpha * s_fsl + self.beta * s_cos + self.gamma * s_sam
# ¿Son estos alpha/beta/gamma los mismos que scorer.alpha/beta/gamma?
# Respuesta: NO necesariamente. fd_uq tiene sus propios defaults (0.5, 0.3, 0.2).
```

**Corrección:** Pasar el `scorer` a `fd_uq.estimate()` o asegurar que ambos compartan los mismos pesos.

---

### 🟡 MEDIO-1: `medgemma_loader.py` — `apply_chat_template` puede no ser soportado

**Archivo:** `opcion_A/medgemma_loader.py`  
**Línea:** 58-62  
**Código:**
```python
messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
```

**Problema:** No todos los modelos `AutoModelForImageTextToText` soportan `apply_chat_template`. Si MedGemma no tiene template de chat registrado, esto falla con `ValueError`.

**Mitigación:** Agregar fallback:
```python
try:
    inputs = self.processor.apply_chat_template(...)
except (ValueError, AttributeError):
    # Fallback: procesamiento manual
    inputs = self.processor(text=prompt, images=image, return_tensors="pt")
```

---

### 🟡 MEDIO-2: `sam_loader.py` — Conversión de imagen asume rango [0,1]

**Archivo:** `opcion_A/sam_loader.py`  
**Línea:** 52-54  
**Código:**
```python
if image.dtype != np.uint8:
    image = (image * 255).clip(0, 255).astype(np.uint8)
```

**Problema:** Si la imagen ya está en rango `[0, 255]` como `float32` (común en pipelines de CV), la multiplicación produce valores hasta `65025` (255×255), y el `.clip()` los trunca a 255, perdiendo toda la información.

**Corrección:**
```python
if image.dtype != np.uint8:
    if image.max() <= 1.0:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    else:
        image = image.clip(0, 255).astype(np.uint8)
```

---

### 🟡 MEDIO-3: `pathology_parser.py` — Falsos positivos por substring matching

**Archivo:** `opcion_A/pathology_parser.py`  
**Línea:** 50  
**Código:**
```python
if any(kw.lower() in text_lower for kw in keywords):
```

**Problema:** Substring matching produce falsos positivos:
- "subcatarata" detectaría "catarata"
- "pseudoglaucoma" detectaría "glaucoma"
- "retinopatía hipertensiva" detectaría "retinopatía" (correcto, pero también detectaría "retinopatía diabética")

**Recomendación:** Usar word boundaries:
```python
import re
pattern = r'\b' + re.escape(kw.lower()) + r'\b'
if re.search(pattern, text_lower):
```

---

### 🟡 MEDIO-4: `pipeline.py` — Falta manejo de excepciones robusto

**Archivo:** `opcion_A/pipeline.py`  
**Línea:** 91-140 (`__call__` y `run_with_fsl`)

**Problema:** Ninguna función del pipeline tiene `try/except`. Si:
- SAM 2 falla (imagen muy grande)
- MedGemma genera texto vacío
- Feature extractor recibe máscara vacía (área = 0)
- CUDA OOM

...el programa crashea con stack trace sin información útil para el usuario.

**Recomendación:** Agregar manejo de errores por etapa:
```python
try:
    text = self.medgemma.generate(image, prompt)
except Exception as e:
    logger.error("Etapa 1 (MedGemma) falló: %s", e)
    return {'error': 'text_generation_failed', 'detail': str(e)}
```

---

### 🟡 MEDIO-5: `xai.py` — `uncertainty_breakdown()` suma señales crudas, no ponderadas

**Archivo:** `opcion_A/xai.py`  
**Línea:** 76-80  
**Código:**
```python
distributional = (u_components['U_LD'] + u_components['U_BORDE'] +
                  u_components['U_DIMVAR'] + u_components['U_PROTO'])
pipeline = (u_components['U_acuerdo'] + u_components['U_magnitud'] +
            u_components['U_margen'])
```

**Problema:** El composite usa pesos (0.25, 0.20, 0.15, etc.), pero el breakdown suma las señales crudas. Esto produce porcentajes inconsistentes. Ejemplo: si U_LD=1.0 (máximo) y U_proto=0.0, el composite da 0.25, pero el breakdown muestra 50% distributional.

**Corrección:** Aplicar los mismos pesos del composite en el breakdown:
```python
from opcion_A.fd_uncertainty import FDUncertaintyEstimator

weights = FDUncertaintyEstimator.WEIGHTS
distributional = (weights['U_LD']*u_components['U_LD'] + ...)
```

---

### 🟡 MEDIO-6: `metrics.py` — Faltan métricas médicas específicas

**Archivo:** `opcion_A/metrics.py`  

**Problema:** Solo tiene IoU y Dice. Para segmentación médica, métricas adicionales son estándar:
- **Hausdorff Distance**: mide qué tan lejos están los contornos (crítico para segmentación quirúrgica)
- **Sensitivity / Recall**: % de píxeles patológicos correctamente detectados
- **Specificity**: % de píxeles sanos correctamente clasificados
- **PPV / Precision**: % de píxeles predichos como patológicos que realmente lo son

**Recomendación:** Agregar:
```python
def compute_medical_metrics(pred_mask, gt_mask):
    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, ~gt_mask).sum()
    fn = np.logical_and(~pred_mask, gt_mask).sum()
    tn = np.logical_and(~pred_mask, ~gt_mask).sum()
    
    return {
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0.0,
    }
```

---

### 🟡 MEDIO-7: `fd_bank.py` — `register()` no maneja support set vacío

**Archivo:** `opcion_A/fd_bank.py`  
**Línea:** 38-58  

**Problema:** Si `support_masks` es vacío, `torch.stack([])` falla. Si tiene 1 elemento, `k=1` y `h=1.0`, lo que puede producir KDE degenerado.

**Recomendación:**
```python
if len(support_masks) < 2:
    raise ValueError(f"Support set debe tener k≥2 máscaras. Recibido: {len(support_masks)}")
```

---

### 🟢 BAJO-1: `__init__.py` — Vacío, sin exports

**Archivo:** `opcion_A/__init__.py`  

**Problema:** Solo tiene `__version__`. No exporta las clases principales, obligando a imports largos:
```python
from opcion_A.pipeline import MedGemmaSegPipeline  # Largo
```

**Recomendación:**
```python
from .pipeline import MedGemmaSegPipeline
from .medgemma_loader import MedGemmaLoader
from .sam_loader import SAMLoader
# ... etc
```

---

### 🟢 BAJO-2: `cosine_fusion.py` — Asume dimensión fija 768

**Archivo:** `opcion_A/cosine_fusion.py`  
**Línea:** 19  
**Docstring:**
```python
mask_embeddings: (N, 768) — embeddings de N máscaras candidatas
text_embedding:  (1, 768) — embedding del texto diagnóstico
```

**Problema:** Si `feature_extractor` o `text_encoder` retornan una dimensión diferente (ver ALTO-2), la matmul `@` fallará con dimension mismatch.

**Mitigación:** Agregar assert o manejo dinámico:
```python
assert mask_embeddings.shape[-1] == text_embedding.shape[-1], \
    f"Dim mismatch: masks={mask_embeddings.shape}, text={text_embedding.shape}"
```

---

### 🟢 BAJO-3: `tests/test_pipeline.py` — Vacío

**Archivo:** `tests/test_pipeline.py`  

**Problema:** Solo tiene docstring. No hay tests de integración que verifiquen que todo el pipeline funciona junto.

**Tests recomendados:**
1. Test de carga de modelos (sin GPU, usando mocks)
2. Test de flujo end-to-end con mocks de modelos
3. Test de CLI con argumentos válidos/inválidos
4. Test de manejo de errores (imagen corrupta, etc.)

---

### 🟢 BAJO-4: `notebooks/demo.ipynb` — Probablemente vacío

**Archivo:** `notebooks/demo.ipynb`  
**Tamaño:** 243 bytes

**Problema:** Un notebook de 243 bytes está vacío o tiene solo la celda de imports. No hay demo ejecutable.

**Recomendación:** Crear un notebook con:
1. Celda de setup e instalación
2. Celda de carga de modelos
3. Celda de ejecución con imagen de ejemplo
4. Celda de visualización de resultados

---

### 🟢 BAJO-5: `AGENTS.md` — Información desactualizada

**Archivo:** `AGENTS.md`  

| Línea actual | Problema | Corrección |
|---|---|---|
| "448×448 resolution, 768-dim embeddings" | MedGemma internamente usa 896×896 | "896×896 (MedGemma) / 448×448 (MedSigLIP standalone)" |
| "MedGemma ~8 GB" | Con dual strategy total es ~10 GB | "~8 GB (MedGemma) + ~1.5 GB (MedSigLIP) + ~0.3 GB (SAM 2) = ~10 GB" |
| "segment-anything" en Stack | Ya fue reemplazado por SAM 2 | "SAM 2 (build_sam2)" |
| "Backbone: Reuse MedSigLIP" | Actualmente usa standalone | "Backbone: MedSigLIP standalone (google/medsiglip-448)" |

---

## 2. Resumen por Severidad

| Severidad | Cantidad | Archivos afectados |
|---|---|---|
| 🔴 Crítico | 3 | `fd_kde.py`, `ood_window.py`, `simple_shot.py` |
| 🟠 Alto | 6 | `fsl_fd.py`, `feature_extractor.py`, `text_encoder.py`, `scoring.py`, `fd_uncertainty.py`, `multi_pathology.py` |
| 🟡 Medio | 7 | `medgemma_loader.py`, `sam_loader.py`, `pathology_parser.py`, `pipeline.py`, `xai.py`, `metrics.py`, `fd_bank.py` |
| 🟢 Bajo | 5 | `__init__.py`, `cosine_fusion.py`, `test_pipeline.py`, `demo.ipynb`, `AGENTS.md` |
| **Total** | **21** | — |

---

## 3. Prioridad de Corrección Recomendada

### Sprint 1 (Inmediato — bloquean ejecución correcta)

1. 🔴 **CRÍTICO-2**: `ood_window.py` — División por cero (línea 54)
2. 🔴 **CRÍTICO-1**: `fd_kde.py` — Consistencia NumPy/Torch (línea 58)
3. 🟠 **ALTO-4**: `scoring.py` — Duplicación de lógica `s_total` (línea 124)
4. 🟠 **ALTO-6**: `multi_pathology.py` — Consistencia pesos fd_uq/scorer (línea 66)

### Sprint 2 (Antes de primera ejecución en Colab)

5. 🟠 **ALTO-2**: `feature_extractor.py` — `pooler_output` None check (línea 72)
6. 🟠 **ALTO-1**: `fsl_fd.py` — Device del tensor (línea 45)
7. 🟡 **MEDIO-2**: `sam_loader.py` — Conversión de imagen (línea 52)
8. 🟡 **MEDIO-4**: `pipeline.py` — Manejo de excepciones

### Sprint 3 (Mejoras de calidad)

9. 🟠 **ALTO-3**: `text_encoder.py` — Carga duplicada (requiere refactor arquitectónico)
10. 🔴 **CRÍTICO-3**: `simple_shot.py` — Calibración softmax (línea 42)
11. 🟡 **MEDIO-3**: `pathology_parser.py` — Word boundaries (línea 50)
12. 🟡 **MEDIO-5**: `xai.py` — Ponderación en breakdown (línea 76)
13. 🟡 **MEDIO-6**: `metrics.py` — Métricas médicas adicionales
14. 🟢 **BAJO-1**: `__init__.py` — Exports
15. 🟢 **BAJO-3**: `test_pipeline.py` — Tests de integración

---

## 4. Recomendaciones Arquitectónicas

### 4.1 Compartir modelo SigLIP entre feature_extractor y text_encoder

**Problema:** Se cargan ~3 GB de vision encoder duplicados.

**Solución:**
```python
# opcion_A/shared_siglip.py
class SharedSigLIPModel:
    """Singleton para compartir SiglipModel entre componentes."""
    _instance = None
    _model = None
    _processor = None
    
    @classmethod
    def load(cls, model_id="google/medsiglip-448", device="cuda"):
        if cls._model is None:
            cls._model = SiglipModel.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
            cls._processor = AutoProcessor.from_pretrained(model_id)
        return cls._model, cls._processor
```

### 4.2 Configuración centralizada de pesos

**Problema:** `CompositeScorer`, `FDUncertaintyEstimator`, y `MultiPathologyPipeline` tienen sus propios `alpha/beta/gamma`.

**Solución:**
```python
# opcion_A/config.py
PIPELINE_CONFIG = {
    'scorer': {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2, 'tau': 0.6},
    'fd_uq': {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2, 'tau': 0.6},
    'kde': {'method': 'iqr'},
}
```

### 4.3 Logging estructurado por etapa

**Problema:** El pipeline no tiene visibilidad de qué etapa falla.

**Solución:** Usar el logger ya configurado en cada etapa con métricas de timing:
```python
import time
start = time.time()
# ... etapa ...
logger.info("Etapa X completada en %.2f ms", (time.time() - start) * 1000)
```

---

## 5. Verificación Post-Corrección

Después de aplicar las correcciones, ejecutar:

```bash
# 1. Validación de sintaxis
python -m py_compile opcion_A/*.py

# 2. Tests unitarios (sin GPU)
pytest tests/test_kde.py tests/test_ood_window.py tests/test_fd_uncertainty.py -v

# 3. Smoke test del pipeline (con mocks)
pytest tests/test_pipeline.py -v  # Después de implementar

# 4. Verificación de dimensiones (en Colab)
python -c "from opcion_A.pipeline import MedGemmaSegPipeline; p = MedGemmaSegPipeline(); p.load()"
```
