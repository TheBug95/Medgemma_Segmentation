"""
Módulo de carga de MedGemma 4B-IT via HuggingFace.

Fase 1 — Tarea 1.1
- Carga MedGemma en bf16 con device_map="auto"
- NO modifica pesos (inferencia pura, caja negra)
- Usa AutoModelForImageTextToText (NO AutoModel)
- Modelo: google/medgemma-4b-it (instruction-tuned)
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


class MedGemmaLoader:
    """Carga y gestiona MedGemma 4B-IT para inferencia multimodal."""

    DEFAULT_MODEL_ID = "google/medgemma-4b-it"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.model_id = model_id
        self.model = None
        self.processor = None

    def load(self):
        """
        Carga modelo y processor desde HuggingFace.

        Returns:
            tuple: (model, processor)
        """
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        return self.model, self.processor

    def generate(self, image, prompt: str = "Describe y segmenta la patología",
                 max_new_tokens: int = 512) -> str:
        """
        Genera texto diagnóstico a partir de una imagen médica.

        Args:
            image: PIL.Image o np.ndarray — imagen médica
            prompt: instrucción para el modelo
            max_new_tokens: máximo de tokens a generar

        Returns:
            str — texto generado (descripción diagnóstica)
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Modelo no cargado. Llama a load() primero.")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Decodificar solo los tokens nuevos (sin el prompt)
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output[0][input_len:]
        text = self.processor.decode(generated_ids, skip_special_tokens=True)
        return text

    def unload(self):
        """Libera memoria del modelo."""
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        torch.cuda.empty_cache()
