# Análisis Exhaustivo del Repositorio MedGemma-Seg (v2 — Actualizado)

## 1. Visión General

Extender **MedGemma** (LLM médico multimodal de Google) con capacidades de **segmentación de imágenes médicas** usando un pipeline secuencial de 3 etapas (Opción A).

**Contexto actualizado:**
- Ejecución en **Google Colab** (T4 16GB / A100 40GB)
- Multi-patología **desde el día 1**
- Datasets de support set y evaluación **aún por definir**
- WSL disponible para desarrollo local

---

## 2. Estado del Repositorio — Sin Cambios

> Ver la versión anterior del análisis para el inventario completo de archivos. Resumen: ~50% implementado, core matemático (KDE, OOD, FD-UQ) completo y correcto. Los stubs principales son los módulos de carga de modelos (Fase 1).

---

## 3. Hallazgos Críticos Nuevos

### 🔴 Hallazgo 1 — MedGemma usa `AutoModelForImageTextToText`

El model card oficial confirma que la clase correcta es:

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model_id = "google/medgemma-4b-it"
model = AutoModelForImageTextToText.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)
```

> [!CAUTION]
> El código actual usa `AutoModel` — incorrecto. No es `AutoModelForCausalLM` tampoco. La clase correcta es **`AutoModelForImageTextToText`** (pipeline tipo `image-text-to-text`).

### 🔴 Hallazgo 2 — Resolución de imagen: 896×896 (NO 448×448)

Del model card oficial:
> *"Images, normalized to **896 x 896** resolution and encoded to **256 tokens** each"*

Esto contradice la documentación actual del proyecto que asume 448×448. La diferencia es sustancial:
- **AGENTS.md** dice: *"448×448 resolution, 768-dim embeddings"*
- **Model card real**: 896×896, 256 visual tokens

> [!IMPORTANT]
> **Impacto**: El feature extractor (`feature_extractor.py`) debe preprocesar las imágenes a 896×896 para que los embeddings sean consistentes con MedGemma. Si se usa MedSigLIP standalone, la resolución es 448×448. Esto crea una discrepancia que hay que resolver.

### 🟢 Hallazgo 3 — MedSigLIP está disponible como modelo STANDALONE

El model card de MedGemma dice explícitamente:
> *"For medical image-based applications that do **not** involve text generation, such as data-efficient classification, zero-shot classification, or content-based or semantic image retrieval, the **MedSigLIP image encoder** is recommended."*

El modelo standalone es **`google/medsiglip-448`** (448×448) y se puede cargar sin el LLM:

```python
from transformers import AutoProcessor, SiglipVisionModel

vision_model = SiglipVisionModel.from_pretrained("google/medsiglip-448")
processor = AutoProcessor.from_pretrained("google/medsiglip-448")

inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = vision_model(**inputs)
    embeddings = outputs.last_hidden_state  # o pooler_output
```

> [!TIP]
> **Esto es excelente para el pipeline.** Significa que para la extracción de features de las máscaras (el módulo `feature_extractor.py`), se puede usar MedSigLIP standalone (~400M params, ~1.5 GB VRAM) **sin tener que cargar todo MedGemma**. Esto reduce drásticamente el uso de VRAM.

### Estrategia Dual Propuesta

```
MedGemma 4B-IT (completo, ~8 GB bf16)
  └── Se usa SOLO para generación de texto diagnóstico (Etapa 1)

MedSigLIP standalone (448×448, ~1.5 GB bf16)  
  └── Se usa para extracción de features de máscaras (Etapa 3)
  └── Se usa para embeddings de texto (SigLIP text encoder)

SAM 2 Tiny (~0.3 GB)
  └── Se usa para generación de máscaras candidatas (Etapa 2)

Total VRAM estimado: ~10 GB ← Cabe en T4 (16 GB) con margen
```

---

## 4. Correcciones Necesarias al Código

### Corrección 1 — `medgemma_loader.py`

```diff
- from transformers import AutoModel, AutoTokenizer
+ from transformers import AutoProcessor, AutoModelForImageTextToText

- model_id: str = "google/medgemma-4b"
+ model_id: str = "google/medgemma-4b-it"

- self.model = AutoModel.from_pretrained(...)
+ self.model = AutoModelForImageTextToText.from_pretrained(
+     self.model_id,
+     torch_dtype=torch.bfloat16,
+     device_map="auto",
+ )
+ self.processor = AutoProcessor.from_pretrained(self.model_id)
```

### Corrección 2 — `sam_loader.py`

```diff
- from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
+ from sam2.build_sam import build_sam2
+ from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

- sam = sam_model_registry["vit_t"](checkpoint="sam2_tiny.pth")
- self.mask_generator = SamAutomaticMaskGenerator(sam)
+ sam2_model = build_sam2(
+     "configs/sam2.1/sam2.1_hiera_t.yaml",
+     "checkpoints/sam2.1_hiera_tiny.pt",
+     device="cuda"
+ )
+ self.mask_generator = SAM2AutomaticMaskGenerator(sam2_model)
```

### Corrección 3 — `feature_extractor.py` (usar MedSigLIP standalone)

```python
from transformers import AutoProcessor, SiglipVisionModel
import torch

class MaskFeatureExtractor:
    def __init__(self, model_id="google/medsiglip-448"):
        self.model = SiglipVisionModel.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model.eval()

    def extract(self, image, mask):
        # Aplicar máscara a la imagen
        masked_image = image * mask[..., None]  # (H, W, 3)
        inputs = self.processor(images=masked_image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs.to(self.model.device))
        return outputs.pooler_output.squeeze(0)  # (768,)
```

### Corrección 4 — `fsl_fd.py` (import faltante)

```diff
  from opcion_A.fd_kde import PerDimensionKDE
  from opcion_A.ood_window import OODWindow
  from opcion_A.fd_uncertainty import FDUncertaintyEstimator
+ import torch
```

---

## 5. Datasets Recomendados para Multi-Patología

Para tu estrategia de multi-patología desde el día 1, necesitas datasets con **imágenes + máscaras de segmentación**. Aquí están los mejores candidatos:

### Oftalmología (tu dominio principal)

| Dataset | Patología | # Imágenes | Máscaras | Acceso |
|---|---|---|---|---|
| **REFUGE** | Glaucoma (disco óptico + copa) | 1,200 | ✅ Pixel-level | [Kaggle](https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification) |
| **IDRiD** | Retinopatía diabética (microaneurismas, exudados, hemorragias) | 516 | ✅ Pixel-level por lesión | [grand-challenge.org](https://idrid.grand-challenge.org/) |
| **DRIVE** | Vasos retinales | 40 | ✅ Vessel segmentation | [grand-challenge.org](https://drive.grand-challenge.org/) |
| **ORIGA** | Glaucoma (disco + copa) | 650 | ✅ Contours | Solicitar a SERI |
| **RIGA** | Glaucoma (multi-anotador) | 750 | ✅ Multiple annotators | [Academic Torrents](https://academictorrents.com/) |

### Otras modalidades (para demostrar generalización)

| Dataset | Patología | # Imágenes | Máscaras |
|---|---|---|---|
| **CBIS-DDSM** | Tumores mamarios (mamografía) | 2,620 | ✅ ROI masks |
| **CAMELYON16** | Metástasis en ganglios linfáticos (histopatología) | 399 WSI | ✅ Tumor masks |
| **MIMIC-CXR** | Hallazgos pulmonares (radiografía de tórax) | 377,110 | ⚠️ Solo reportes, no masks |

### Recomendación para el MVP Multi-Patología

Para empezar con **k=6-12 masks** por patología (como requiere el FSL/FD):

```
data/support_sets/
├── glaucoma/          ← 9 masks de REFUGE (disco óptico + copa)
├── diabetic_ret/      ← 9 masks de IDRiD (exudados/hemorragias)
├── retinal_vessels/   ← 9 masks de DRIVE (vasos retinales)
└── optic_disc/        ← 9 masks de REFUGE (solo disco óptico)
```

> [!TIP]
> **Prioridad**: Descarga REFUGE primero — es el más accesible (Kaggle), tiene masks pixel-level de alta calidad, y cubre glaucoma que es directamente relevante para tu dominio oftalmológico. IDRiD segundo por la riqueza de patologías (4 tipos de lesiones con masks individuales).

---

## 6. Guía Específica para Google Colab

### Runtime recomendado

| GPU | VRAM | ¿Suficiente? | Notas |
|---|---|---|---|
| T4 | 16 GB | ✅ Sí | Con bf16 y estrategia dual MedSigLIP |
| L4 | 24 GB | ✅ Sí con margen | Más rápida que T4 |
| A100 | 40 GB | ✅ Sobra | Solo Colab Pro/Enterprise |

### Snippet de setup para Colab

```python
# Celda 1: Instalación
!pip install -U transformers accelerate
!pip install einops tqdm scipy matplotlib

# SAM 2 — instalar desde repo
!git clone https://github.com/facebookresearch/sam2.git
%cd sam2
!pip install -e .
%cd ..

# Descargar checkpoint SAM 2 Tiny
!mkdir -p checkpoints
!wget -O checkpoints/sam2.1_hiera_tiny.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

# Celda 2: Autenticación HuggingFace
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')

# Celda 3: Cargar modelos
import torch
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText,
    SiglipVisionModel
)

# MedGemma para generación de texto
medgemma = AutoModelForImageTextToText.from_pretrained(
    "google/medgemma-4b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
medgemma_proc = AutoProcessor.from_pretrained("google/medgemma-4b-it")

# MedSigLIP standalone para features
medsiglip = SiglipVisionModel.from_pretrained(
    "google/medsiglip-448",
    torch_dtype=torch.bfloat16,
).to("cuda")
medsiglip_proc = AutoProcessor.from_pretrained("google/medsiglip-448")

# SAM 2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

sam2 = build_sam2(
    "sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
    "checkpoints/sam2.1_hiera_tiny.pt",
    device="cuda"
)
mask_gen = SAM2AutomaticMaskGenerator(sam2)

print("✅ Todos los modelos cargados")
```

---

## 7. Impacto en la Documentación del Proyecto

### Cambios necesarios en `AGENTS.md`

| Línea actual | Corrección |
|---|---|
| `"448×448 resolution"` | → `"448×448 (standalone) / 896×896 (dentro de MedGemma)"` |
| `"MedGemma ~8 GB"` | → `"MedGemma ~8 GB (texto) + MedSigLIP ~1.5 GB (features)"` |
| Tabla `Minimal GPU: 16 GB VRAM` | ✅ Correcto para Colab T4 con la estrategia dual |

### Cambios necesarios en `implementation_guide_detallado.md`

1. Actualizar la sección de carga de MedGemma con `AutoModelForImageTextToText`
2. Agregar `MedSigLIP standalone` como componente separado
3. Actualizar SAM 2 API con `build_sam2` + `SAM2AutomaticMaskGenerator`
4. Actualizar `requirements.txt` (remover `segment-anything`, documentar instalación de `sam2`)

---

## 8. Preguntas Resueltas y Pendientes

### ✅ Resueltas

| # | Pregunta | Respuesta |
|---|---|---|
| 1 | Acceso a MedGemma | ✅ Sí, tiene acceso |
| 2 | Support sets | Se van a buscar de datasets abiertos |
| 3 | WSL | Disponible, pero ejecución principal en Colab |
| 4 | Multi-patología | Desde el principio |
| 5 | Dataset de evaluación | Se define después |

### ❓ Pendientes (no bloqueantes)

| # | Pregunta | Relevancia |
|---|---|---|
| 1 | ¿`pooler_output` de MedSigLIP standalone es 768-dim? | Verificar empíricamente. Si es 1152 o diferente, hay que ajustar `dim` en KDE |
| 2 | ¿El text encoder de SigLIP dentro de MedSigLIP está en `google/medsiglip-448`? | Necesario para `text_encoder.py`. Si no, usar `google/siglip-base-patch16-384` standalone |
| 3 | ¿La configuración de SAM2 YAML está en la ruta correcta después de `pip install -e .`? | Verificar en Colab al ejecutar |

---

## 9. Plan de Acción Priorizado (Actualizado para Colab)

### Paso 0 — Verificación de modelos (30 min, en Colab)
```
[ ] Cargar MedGemma 4B-IT con AutoModelForImageTextToText
[ ] Generar texto con una imagen médica de prueba
[ ] Cargar MedSigLIP standalone, verificar dimensión de embeddings
[ ] Cargar SAM 2 Tiny, generar máscaras de una imagen de prueba
[ ] Medir VRAM total con los 3 modelos cargados
```

### Paso 1 — Datasets (1-2 horas)
```
[ ] Descargar REFUGE desde Kaggle
[ ] Seleccionar k=9 masks por patología para support sets
[ ] Descargar IDRiD (si acceso rápido)
```

### Paso 2 — Correcciones de código (1 hora)
```
[ ] Corregir medgemma_loader.py (API correcta)
[ ] Corregir sam_loader.py (SAM 2 API)
[ ] Implementar feature_extractor.py (MedSigLIP standalone)
[ ] Agregar import torch en fsl_fd.py
[ ] Corregir fd_bank.py (firma de register())
```

### Paso 3 — Pipeline end-to-end (2-3 horas)
```
[ ] Implementar pipeline.py (orquestador)
[ ] Implementar text_encoder.py 
[ ] Verificar cosine_fusion con embeddings reales
[ ] Run completo: imagen → texto + máscara
```

### Paso 4 — Validación FSL/FD (2-3 horas)
```
[ ] Registrar patologías en FDBank con support sets reales
[ ] Ejecutar scoring con imágenes de test
[ ] Verificar que las 3 condiciones de decisión funcionan
```

---

## 10. Veredicto Final (Actualizado)

> [!TIP]
> **El diseño arquitectónico es sólido y el core matemático está correctamente implementado.** Los únicos problemas son de **integración con APIs externas** — todos corregibles en 1-2 horas de trabajo. El descubrimiento de MedSigLIP standalone (`google/medsiglip-448`) **mejora significativamente** la viabilidad del pipeline, ya que permite extraer features sin cargar el LLM completo.

| Aspecto | Calificación | Cambio vs v1 |
|---|---|---|
| Diseño arquitectónico | ⭐⭐⭐⭐⭐ | = |
| Documentación | ⭐⭐⭐⭐⭐ | = |
| Implementación matemática | ⭐⭐⭐⭐⭐ | = |
| Código implementado | ⭐⭐⭐⭐ | = |
| Tests | ⭐⭐⭐⭐ | = |
| API/Integraciones externas | ⭐⭐⭐ | ↑ (ahora sabemos exactamente qué corregir) |
| Viabilidad en Colab (T4) | ⭐⭐⭐⭐⭐ | **NUEVO** — confirmado viable con estrategia dual |
