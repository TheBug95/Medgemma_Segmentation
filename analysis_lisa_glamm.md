# Análisis Arquitectónico: LISA y GLaMM

Análisis detallado de los dos repositorios de segmentación multimodal basados en LLM presentes en la carpeta `MedGemma Segmentation`.

---

## 1. LISA — Large Language Instructed Segmentation Assistant

### 1.1 Explicación Global

LISA es un sistema que combina un **Modelo de Lenguaje Grande Multimodal** (LLM, por sus siglas en inglés — un modelo de inteligencia artificial capaz de entender texto e imágenes simultáneamente) con un **modelo de segmentación de imágenes** (SAM — Segment Anything Model, un modelo de Meta capaz de "recortar" objetos en imágenes).

**¿Qué hace LISA?** Recibe una imagen y una pregunta/instrucción en texto natural (por ejemplo: *"¿Quién es el presidente en esta imagen? Por favor genera una máscara de segmentación"*), y produce dos cosas:
1. Una **respuesta en texto** explicando su razonamiento.
2. Una **máscara de segmentación** (una imagen binaria que delimita exactamente el objeto identificado).

La innovación clave es que LISA puede realizar **"reasoning segmentation"** (segmentación por razonamiento): entiende instrucciones implícitas y complejas que requieren conocimiento del mundo real, no solo referencias directas como "segmenta el gato rojo".

### 1.2 Diagrama de Arquitectura

```mermaid
graph TB
    subgraph ENTRADA["📥 Entrada del Usuario"]
        IMG["🖼️ Imagen Original"]
        TXT["💬 Texto/Instrucción"]
    end

    subgraph DUAL_ENC["🔍 Doble Codificación de Imagen"]
        direction LR
        CLIP["CLIP ViT-L/14<br/>(Encoder Global)<br/>Congela parámetros"]
        SAM_ENC["SAM ViT-H<br/>(Encoder de Segmentación)<br/>Congela parámetros"]
    end

    subgraph LLM_CORE["🧠 Núcleo LLM — LLaVA + LLaMA"]
        MM_PROJ["MM Projector<br/>(Proyector Multimodal)<br/>Congela parámetros"]
        TOKENIZER["Tokenizer<br/>+ Token especial [SEG]"]
        LLAMA["LLaMA (7B/13B)<br/>Backbone de lenguaje<br/>LoRA adapters entrenables"]
        LM_HEAD["LM Head<br/>(Cabeza de lenguaje)<br/>Entrenable"]
    end

    subgraph SEG_DECODER["🎯 Decodificador de Segmentación"]
        TEXT_FC["Text Hidden FCs<br/>(Capa de Proyección)<br/>Linear→ReLU→Linear<br/>Entrenable"]
        PROMPT_ENC["SAM Prompt Encoder<br/>Recibe embeddings de texto"]
        MASK_DEC["SAM Mask Decoder<br/>Entrenable"]
    end

    subgraph SALIDA["📤 Salida"]
        TEXT_OUT["📝 Respuesta en Texto"]
        MASK_OUT["🎭 Máscara de Segmentación"]
    end

    IMG --> CLIP
    IMG --> SAM_ENC
    TXT --> TOKENIZER
    CLIP --> MM_PROJ
    MM_PROJ --> LLAMA
    TOKENIZER --> LLAMA
    LLAMA --> LM_HEAD
    LM_HEAD --> TEXT_OUT
    LLAMA -->|"Hidden states<br/>del token [SEG]"| TEXT_FC
    TEXT_FC -->|"Embedding proyectado<br/>(256-dim)"| PROMPT_ENC
    SAM_ENC -->|"Image embeddings"| MASK_DEC
    PROMPT_ENC -->|"Sparse + Dense<br/>embeddings"| MASK_DEC
    MASK_DEC --> MASK_OUT

    style ENTRADA fill:#e8f4f8,stroke:#2196F3
    style DUAL_ENC fill:#fff3e0,stroke:#FF9800
    style LLM_CORE fill:#f3e5f5,stroke:#9C27B0
    style SEG_DECODER fill:#e8f5e9,stroke:#4CAF50
    style SALIDA fill:#fce4ec,stroke:#E91E63
```

### 1.3 Diagrama del Flujo de Datos (Entrenamiento)

```mermaid
flowchart LR
    subgraph LOSS["⚖️ Función de Pérdida Compuesta"]
        CE["CE Loss<br/>×1.0"]
        BCE["BCE Loss<br/>×2.0"]
        DICE["DICE Loss<br/>×0.5"]
    end

    DS["4 Datasets Mixtos<br/>sem_seg, refer_seg,<br/>vqa, reason_seg"] --> HYB["HybridDataset<br/>Muestreo ponderado<br/>9:3:3:1"]
    HYB --> MODEL["LISAForCausalLM"]
    MODEL -->|"Texto predicho vs GT"| CE
    MODEL -->|"Máscara predicha vs GT"| BCE
    MODEL -->|"Máscara predicha vs GT"| DICE
    CE & BCE & DICE --> TOTAL["Loss Total<br/>= CE + BCE + DICE"]
    TOTAL -->|"DeepSpeed ZeRO-2"| MODEL
```

### 1.4 Explicación Detallada de Componentes

#### A. Codificadores de Imagen (Image Encoders) — Doble Pipeline

LISA utiliza **dos encoders de imagen separados** que procesan la misma imagen de formas distintas:

| Componente | Modelo | Resolución | Función | ¿Se entrena? |
|---|---|---|---|---|
| **Global Encoder** | CLIP ViT-L/14 | 224×224 | Extrae features semánticas globales para que el LLM "entienda" la imagen | ❌ Congelado |
| **Grounding Encoder** | SAM ViT-H | 1024×1024 | Extrae features espaciales de alta resolución para generar máscaras precisas | ❌ Congelado |

- **CLIP** ([Contrastive Language-Image Pre-training](file:///g:/My%20Drive/Experiments/MedGemma%20Segmentation/LISA/model/llava)): Codifica la imagen en un espacio compartido con el texto. Sus features pasan por el `MM Projector` para alinearlas con la dimensión del LLM.
- **SAM Image Encoder**: Codifica la imagen a alta resolución. Su salida se usa **solo** en el decodificador de máscaras, no pasa por el LLM.

#### B. LLM Core — LLaVA + LLaMA

- **Tokenizer**: Procesa el texto y agrega un token especial `[SEG]` al vocabulario. Este token actúa como una "señal" que le dice al sistema: *"aquí debe generarse una máscara de segmentación"*.
- **LLaMA** (7B o 13B parámetros): El modelo de lenguaje base. Se entrena con **LoRA** (Low-Rank Adaptation — una técnica que agrega pequeñas matrices entrenables a las capas `q_proj` y `v_proj` del transformer, sin modificar los pesos originales). Esto reduce drásticamente la memoria necesaria.
- **MM Projector**: Un MLP (red neuronal de 2 capas) que transforma los features de CLIP al espacio de dimensiones del LLM. Está **congelado** durante el entrenamiento.

#### C. Decodificador de Segmentación

- **Text Hidden FCs** ([LISA.py L89-101](file:///g:/My%20Drive/Experiments/MedGemma%20Segmentation/LISA/model/LISA.py#L89-L101)): Una red `Linear(4096→4096) → ReLU → Linear(4096→256)` que proyecta el hidden state del token `[SEG]` a un embedding de 256 dimensiones compatible con SAM.
- **SAM Prompt Encoder**: Toma el embedding proyectado y lo convierte en "sparse embeddings" (prompts de texto para SAM) y "dense embeddings".
- **SAM Mask Decoder**: Combina los image embeddings del SAM Encoder con los prompts del Prompt Encoder para generar la máscara final. Este componente **sí se entrena**.

#### D. Función de Pérdida

La pérdida total combina tres componentes ([LISA.py L305-335](file:///g:/My%20Drive/Experiments/MedGemma%20Segmentation/LISA/model/LISA.py#L305-L335)):

- **Cross-Entropy Loss** (peso 1.0): Mide qué tan bien el LLM genera el texto correcto.
- **Binary Cross-Entropy Loss** (peso 2.0): Compara pixel a pixel la máscara predicha vs. la real.
- **DICE Loss** (peso 0.5): Mide el solapamiento global entre la máscara predicha y la real (similar a IoU — Intersection over Union).

#### E. Pipeline de Entrenamiento

Definido en [train_ds.py](file:///g:/My%20Drive/Experiments/MedGemma%20Segmentation/LISA/train_ds.py):
- Usa **DeepSpeed ZeRO Stage 2** para entrenamiento distribuido eficiente.
- Combina 4 tipos de datos con muestreo ponderado (9:3:3:1): segmentación semántica, segmentación referencial, VQA, y segmentación por razonamiento.
- Validación con métricas **gIoU** (global Intersection over Union) y **cIoU** (class IoU).

#### F. Pipeline de Inferencia

Definido en [chat.py](file:///g:/My%20Drive/Experiments/MedGemma%20Segmentation/LISA/chat.py): un loop interactivo donde el usuario ingresa texto + ruta de imagen, y el modelo genera texto + máscara.

---

## 2. GLaMM — Grounding Large Multimodal Model

### 2.1 Explicación Global

GLaMM es una evolución más ambiciosa del concepto de LISA. Mientras que LISA solo procesa instrucciones a nivel de imagen completa, GLaMM introduce **comprensión a nivel de región**: el usuario puede señalar una zona específica de la imagen (con un bounding box — un rectángulo delimitador) y preguntar sobre ella.

**¿Qué hace GLaMM?** Introduce una tarea nueva llamada **Grounded Conversation Generation (GCG)**: generar texto conversacional donde cada frase relevante está "anclada" a una máscara de segmentación específica. Es decir, el modelo no solo dice "hay un perro", sino que produce `<p>perro</p>[SEG]` donde `[SEG]` se convierte en una máscara que muestra exactamente dónde está el perro.

GLaMM tiene **tres encoders** en vez de dos, añadiendo un **Region Encoder** que permite procesar zonas específicas de la imagen.

### 2.2 Diagrama de Arquitectura

```mermaid
graph TB
    subgraph ENTRADA["📥 Entrada del Usuario"]
        IMG2["🖼️ Imagen Original"]
        TXT2["💬 Texto/Instrucción"]
        BBOX["📦 Bounding Box<br/>(Opcional — Región de interés)"]
    end

    subgraph TRIPLE_ENC["🔍 Triple Codificación"]
        CLIP2["CLIP ViT-L/14-336<br/>(Global Image Encoder)<br/>Congelado"]
        SAM_ENC2["SAM ViT-H<br/>(Grounding Encoder)<br/>Congelado"]
        REG_ENC["Region Encoder<br/>(MLVLROIQueryModule)<br/>Entrenable"]
    end

    subgraph REGION_DETAIL["🔎 Detalle del Region Encoder"]
        MLVL_FUSE["MLVLFuseModule<br/>Fusión multi-escala<br/>Channel Shuffle + Conv"]
        ROI_ALIGN["RoI Align<br/>Extracción de features<br/>de la región"]
        POS_EMB["Positional Embedding<br/>Codificación de posición<br/>del bounding box"]
        FLAT_LIN["Flatten + Linear<br/>1024 → 4096 dims"]
    end

    subgraph LLM_CORE2["🧠 Núcleo LLM — LLaVA + LLaMA (con Region-Aware)"]
        MM_PROJ2["MM Projector<br/>Congelado"]
        TOK2["Tokenizer<br/>Tokens: [SEG], &lt;bbox&gt;,<br/>&lt;p&gt;, &lt;/p&gt;, &lt;point&gt;"]
        LLAMA2["LLaMA 7B/13B<br/>LoRA adapters"]
        LM_HEAD2["LM Head<br/>Entrenable"]
    end

    subgraph SEG_DEC2["🎯 Decodificador de Segmentación"]
        TEXT_FC2["Text Hidden FCs<br/>Proyección a 256-dim"]
        PROMPT2["SAM Prompt Encoder"]
        MASK_DEC2["SAM Mask Decoder<br/>Entrenable"]
    end

    subgraph SALIDA2["📤 Salida"]
        TEXT_OUT2["📝 Texto con frases<br/>ancladas: &lt;p&gt;gato&lt;/p&gt;[SEG]"]
        MASK_OUT2["🎭 Múltiples Máscaras<br/>una por cada [SEG]"]
    end

    IMG2 --> CLIP2
    IMG2 --> SAM_ENC2
    BBOX --> REG_ENC
    TXT2 --> TOK2

    CLIP2 -->|"Features multi-nivel<br/>(hidden_states[-2::-3])"| REG_ENC
    CLIP2 --> MM_PROJ2

    REG_ENC --> MLVL_FUSE
    MLVL_FUSE --> ROI_ALIGN
    ROI_ALIGN --> FLAT_LIN
    POS_EMB --> FLAT_LIN
    FLAT_LIN -->|"Region Query (4096-dim)<br/>Reemplaza embedding de &lt;bbox&gt;"| LLAMA2

    MM_PROJ2 --> LLAMA2
    TOK2 --> LLAMA2
    LLAMA2 --> LM_HEAD2
    LM_HEAD2 --> TEXT_OUT2

    LLAMA2 -->|"Hidden states de [SEG]"| TEXT_FC2
    TEXT_FC2 --> PROMPT2
    SAM_ENC2 --> MASK_DEC2
    PROMPT2 --> MASK_DEC2
    MASK_DEC2 --> MASK_OUT2

    style ENTRADA fill:#e8f4f8,stroke:#2196F3
    style TRIPLE_ENC fill:#fff3e0,stroke:#FF9800
    style REGION_DETAIL fill:#fff8e1,stroke:#FFC107
    style LLM_CORE2 fill:#f3e5f5,stroke:#9C27B0
    style SEG_DEC2 fill:#e8f5e9,stroke:#4CAF50
    style SALIDA2 fill:#fce4ec,stroke:#E91E63
```

### 2.3 Diagrama del Flujo de Entrenamiento (Multi-Dataset)

```mermaid
flowchart TB
    subgraph DATASETS["📚 3 Familias de Datos"]
        CAP["Caption Datasets<br/>CocoCap, LLaVA-Instruct<br/>Peso: 0.15"]
        REG["Region Datasets<br/>RefCOCO, RefCOCOg,<br/>RefCOCO+, VisualGenome<br/>Peso: 0.40"]
        SEG["Segmentation Datasets<br/>Semantic, Referring,<br/>GCG (PSG, Flickr, RefCOCOg, GranDf)<br/>Peso: 0.45"]
    end

    SAMPLER["🎲 Random Weighted Sampler<br/>Selección por paso según pesos"]

    CAP & REG & SEG --> SAMPLER
    SAMPLER --> ENGINE["DeepSpeed Engine<br/>ZeRO Stage 2"]
    ENGINE --> LOSS2["Loss = CE + BCE + DICE"]
    LOSS2 --> ENGINE
```

### 2.4 Explicación Detallada de Componentes

#### A. Triple Codificación de Imagen

GLaMM extiende la doble codificación de LISA con un tercer encoder:

| Componente | Modelo | Resolución | Función | ¿Se entrena? |
|---|---|---|---|---|
| **Global Encoder** | CLIP ViT-L/14-**336** | 336×336 | Features semánticas para el LLM (resolución mayor que LISA) | ❌ Congelado |
| **Grounding Encoder** | SAM ViT-H | 1024×1024 | Features espaciales para máscaras | ❌ Congelado |
| **Region Encoder** | MLVLROIQueryModule | Multi-escala | Extrae features de regiones específicas delimitadas por bounding boxes | ✅ Entrenable |

#### B. Region Encoder — El Diferenciador Clave

Este componente ([layers.py](file:///g:/My%20Drive/Experiments/MedGemma%20Segmentation/groundingLMM/model/layers.py)) es lo que distingue a GLaMM de LISA. Funciona así:

1. **MLVLFuseModule** (Multi-Level Fusion): Toma features de CLIP de múltiples capas ocultas (4 niveles), los redimensiona a escalas crecientes, agrega coordenadas espaciales, y realiza "channel shuffle" (mezcla de canales entre niveles vecinos) con convoluciones.
2. **RoI Align** (Region of Interest Alignment): Usa las coordenadas del bounding box para "recortar" las features fusionadas de la región de interés, produciendo un tensor de tamaño fijo (14×14) independiente del tamaño de la región.
3. **Positional Embedding**: Codifica las coordenadas (x1,y1,x2,y2) del bounding box en un embedding posicional de 1024 dimensiones.
4. **Flatten + Linear**: Aplana las features RoI, las proyecta a 1024 dims, suma el positional embedding, y finalmente proyecta a 4096 dims (la dimensión del LLM).

El resultado (un "Region Query") **reemplaza el embedding del token `<bbox>`** en la secuencia de entrada del LLM, inyectando información espacial de la región directamente en el flujo de atención del transformer.

#### C. Tokens Especiales Extendidos

GLaMM agrega más tokens especiales que LISA:

| Token | Función |
|---|---|
| `[SEG]` | Señal para generar una máscara de segmentación (igual que LISA) |
| `<bbox>` | Marcador que se reemplaza por el Region Query del encoder de regiones |
| `<p>` / `</p>` | Delimitadores de frase para GCG — marcan qué texto está "anclado" a un `[SEG]` |
| `<point>` | Marcador para prompts de punto (alternativa al bounding box) |

#### D. LLaVA con Region-Aware Architecture

El archivo [llava_with_region_arch.py](file:///g:/My%20Drive/Experiments/MedGemma%20Segmentation/groundingLMM/model/llava/llava_with_region_arch.py) modifica la preparación de embeddings de LLaVA para:

1. Extraer features multi-nivel de CLIP (`hidden_states[-2::-3]` — cada 3 capas empezando desde la penúltima).
2. Pasarlos por el `Region Encoder` junto con los bounding boxes.
3. Reemplazar los embeddings de los tokens `<bbox>` con los Region Queries resultantes.

#### E. Entrenamiento Multi-Dataset con Muestreo Ponderado

Definido en [train.py](file:///g:/My%20Drive/Experiments/MedGemma%20Segmentation/groundingLMM/train.py):

- **3 familias de datos**: Caption (15%), Region (40%), Segmentation (45%).
- En cada paso de entrenamiento, se selecciona aleatoriamente una familia según sus pesos.
- Cada familia contiene múltiples sub-datasets con sus propias tasas de muestreo internas.
- **Módulos entrenables**: LoRA adapters, `lm_head`, `embed_tokens`, `mask_decoder`, `text_hidden_fcs`, y `region_encoder`.

#### F. GranD Dataset y Pipeline de Anotación

GLaMM incluye un directorio [GranD/](file:///g:/My%20Drive/Experiments/MedGemma%20Segmentation/groundingLMM/GranD) con un pipeline automatizado de anotación en 4 niveles:
1. **Level 1**: Inferencia inicial (detección + segmentación).
2. **Level 2**: Inferencia secundaria (refinamiento).
3. **Level 3**: Generación de captions densos.
4. **Level 4**: Contexto extra y enriquecimiento.

El dataset final, **GranD-f**, contiene ~214K pares imagen-texto anclado para fine-tuning de GCG.

#### G. Demo Interactiva

El archivo [app.py](file:///g:/My%20Drive/Experiments/MedGemma%20Segmentation/groundingLMM/app.py) implementa una interfaz Gradio que permite:
- Dibujar bounding boxes sobre la imagen.
- Hacer preguntas en lenguaje natural.
- Ver las máscaras generadas superpuestas.
- Generar nuevas imágenes con inpainting (SDXL) usando las máscaras como guía.

---

## 3. Tabla Comparativa: LISA vs GLaMM

| Aspecto | LISA | GLaMM |
|---|---|---|
| **Paper** | arXiv:2308.00692 (CVPR 2024 Oral) | arXiv:2311.03356 (CVPR 2024) |
| **Tarea principal** | Reasoning Segmentation | Grounded Conversation Generation (GCG) |
| **Número de encoders** | 2 (CLIP + SAM) | 3 (CLIP + SAM + Region Encoder) |
| **CLIP versión** | ViT-L/14 (224px) | ViT-L/14-**336** (336px) |
| **Entrada de región** | ❌ No soporta | ✅ Bounding boxes + puntos |
| **Token especial** | `[SEG]` | `[SEG]`, `<bbox>`, `<p>`/`</p>`, `<point>` |
| **Grounding de frases** | ❌ | ✅ Cada frase se ancla a una máscara |
| **Datasets de entrenamiento** | 4 tipos, 1 pipeline | 3 familias, ~10 sub-datasets |
| **Muestreo** | Ponderado fijo (9:3:3:1) | Selección aleatoria por peso por paso |
| **Dataset propio** | ReasonSeg (1218 imágenes) | GranD (7.5M conceptos, 810M regiones) |
| **Demo** | Gradio básica (texto + imagen) | Gradio avanzada (dibujar bbox + inpainting) |
| **Dependencias extra** | — | mmdet (MMDetection para RoI ops) |
| **Complejidad del modelo** | Menor (más sencillo de implementar) | Mayor (Region Encoder + multi-nivel) |

---

## 4. Diagrama Comparativo de Flujo

```mermaid
flowchart LR
    subgraph LISA_FLOW["LISA — Flujo Simplificado"]
        direction TB
        L1["Imagen + Texto"] --> L2["CLIP → LLM"]
        L1 --> L3["SAM Encoder"]
        L2 --> L4["Token [SEG]<br/>→ Proyección"]
        L4 --> L5["SAM Decoder"]
        L3 --> L5
        L5 --> L6["Texto + 1 Máscara"]
    end

    subgraph GLAMM_FLOW["GLaMM — Flujo Extendido"]
        direction TB
        G1["Imagen + Texto + BBox"] --> G2["CLIP → LLM"]
        G1 --> G3["SAM Encoder"]
        G1 --> G4["Region Encoder<br/>(Multi-Level + RoI)"]
        G4 -->|"Reemplaza &lt;bbox&gt;"| G2
        G2 --> G5["Múltiples [SEG]<br/>→ Proyección"]
        G5 --> G6["SAM Decoder"]
        G3 --> G6
        G6 --> G7["Texto anclado +<br/>N Máscaras"]
    end

    style LISA_FLOW fill:#e3f2fd,stroke:#1565C0
    style GLAMM_FLOW fill:#fce4ec,stroke:#C62828
```

> [!IMPORTANT]
> **Relación entre ambos repos**: GLaMM reconoce explícitamente a LISA como una de sus bases. Comparten la misma filosofía fundamental (LLM + token `[SEG]` + SAM decoder), pero GLaMM la extiende con comprensión de regiones, anclaje de frases, y un dataset masivamente más grande.
