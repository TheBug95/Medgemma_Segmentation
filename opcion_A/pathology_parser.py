"""
Parser de patología desde texto MedGemma.

Fase 4 — Tarea 4.2
- Detecta TODAS las patologías mencionadas en el texto
- Keyword matching en español e inglés
- Soporta múltiples patologías por imagen
"""

PATHOLOGY_KEYWORDS = {
    "cataract": [
        "catarata", "cataract", "opacidad del cristalino",
        "opacidad cristaliniana",
    ],
    "glaucoma": [
        "glaucoma", "presión intraocular", "excavación papilar",
        "copa óptica",
    ],
    "retinopathy": [
        "retinopatía", "retinopathy", "microaneurismas",
        "exudados", "hemorragias retinianas",
    ],
    "macular_degen": [
        "degeneración macular", "DMAE", "drusen",
        "macular degeneration",
    ],
    "pterygium": [
        "pterigión", "pterygium", "carnosidad",
    ],
    "diabetic_ret": [
        "retinopatía diabética", "diabetic retinopathy",
        "RDNP", "RDP",
    ],
}


def parse_pathology(medgemma_text: str) -> list:
    """
    Detecta TODAS las patologías mencionadas en el texto de MedGemma.

    Args:
        medgemma_text: texto diagnóstico generado por MedGemma

    Returns:
        List[str] — nombres de patologías detectadas, o ["unknown"] si ninguna coincide
    """
    text_lower = medgemma_text.lower()
    detected = []
    for pathology, keywords in PATHOLOGY_KEYWORDS.items():
        if any(kw.lower() in text_lower for kw in keywords):
            detected.append(pathology)
    return detected if detected else ["unknown"]
