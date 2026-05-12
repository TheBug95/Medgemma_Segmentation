"""
Framework BIP — Benchmark de configuraciones del pipeline.

Fase 6 — Tarea 6.1
- Grid search de hiperparámetros (k, α, method, τ)
- Evaluación sistemática de configuraciones
- Comparativa coseno vs FSL vs combinado
"""

from typing import List


class BIPBenchmark:
    """
    Framework para evaluar sistemáticamente configuraciones del pipeline.
    """

    def __init__(self, pipeline, val_dataset):
        self.pipeline = pipeline
        self.val_dataset = val_dataset

    def grid_search(self, k_values: list = None, alpha_values: list = None,
                    methods: list = None, tau_values: list = None) -> list:
        """
        Barrido de hiperparámetros.

        Default:
            k: [3, 6, 9, 12, 30]
            alpha: [0.3, 0.5, 0.7]
            method: ['fd', 'simple_shot']
            tau: [0.5, 0.6, 0.7]

        Returns:
            List[dict] con configuración + métricas
        """
        k_values = k_values or [3, 6, 9, 12, 30]
        alpha_values = alpha_values or [0.3, 0.5, 0.7]
        methods = methods or ['fd', 'simple_shot']
        tau_values = tau_values or [0.5, 0.6, 0.7]

        configs = []
        for k in k_values:
            for alpha in alpha_values:
                for method in methods:
                    for tau in tau_values:
                        configs.append({
                            'k': k, 'alpha': alpha, 'method': method, 'tau': tau,
                            'beta': 1.0 - alpha - 0.2,
                            'gamma': 0.2,
                        })

        results = []
        for cfg in configs:
            metrics = self.evaluate_config(cfg)
            results.append({**cfg, **metrics})

        return results

    def evaluate_config(self, config: dict) -> dict:
        """Evalúa una configuración específica sobre el dataset de validación."""
        raise NotImplementedError("Tarea 6.1 pendiente de implementación")

    def compare_strategies(self) -> dict:
        """
        Compara las 3 estrategias:
        - Cosine-only (baseline)
        - FSL-only
        - Combinado
        """
        raise NotImplementedError("Tarea 6.3 pendiente de implementación")
