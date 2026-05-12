"""
Tests para PerDimensionKDE — Fase 2, Tarea 2.1.

Valida:
- Bandwidth h = k^(-1/5)
- Log-densidad de queries dentro del support > queries outlier
- Shape correcto de outputs
- Simetría: log_density(query) == log_density(query) (determinismo)
"""

import torch
import numpy as np
import pytest


class TestPerDimensionKDEInit:
    """Tests de inicialización del KDE."""

    def test_bandwidth_formula(self, support_embeddings):
        """h debe ser k^(-1/5) exactamente como el paper IGPL."""
        from opcion_A.fd_kde import PerDimensionKDE
        kde = PerDimensionKDE(support_embeddings)
        k = support_embeddings.shape[0]
        expected_h = k ** (-1.0 / 5.0)
        assert kde.h == pytest.approx(expected_h, rel=1e-10)

    def test_dimensions(self, support_embeddings):
        """k y dim deben reflejar el support set."""
        from opcion_A.fd_kde import PerDimensionKDE
        kde = PerDimensionKDE(support_embeddings)
        assert kde.k == support_embeddings.shape[0]
        assert kde.dim == support_embeddings.shape[1] == 768

    def test_support_stored(self, support_embeddings):
        """El support set debe almacenarse sin copia."""
        from opcion_A.fd_kde import PerDimensionKDE
        kde = PerDimensionKDE(support_embeddings)
        assert torch.equal(kde.support, support_embeddings)


class TestPerDimensionKDELogDensity:
    """Tests de log_density."""

    def test_output_shapes(self, kde, query_embedding):
        """total_log_density es scalar, per_dim es (768,)."""
        total, per_dim = kde.log_density(query_embedding)
        assert total.dim() == 0  # scalar
        assert per_dim.shape == (768,)

    def test_query_in_support_high_density(self, kde, support_embeddings):
        """Un elemento del support set debe tener alta log-densidad."""
        # Tomar el primer elemento del support set como query
        query = support_embeddings[0]
        total, _ = kde.log_density(query)
        assert total.item() > -100  # threshold conservador

    def test_outlier_lower_density_than_inlier(self, kde, support_embeddings, query_outlier):
        """Un outlier debe tener log-densidad menor que un elemento del support."""
        inlier_ld, _ = kde.log_density(support_embeddings[0])
        outlier_ld, _ = kde.log_density(query_outlier)
        assert inlier_ld.item() > outlier_ld.item()

    def test_deterministic(self, kde, query_embedding):
        """Misma query debe dar mismo resultado siempre."""
        ld1, dim1 = kde.log_density(query_embedding)
        ld2, dim2 = kde.log_density(query_embedding)
        assert torch.equal(ld1, ld2)
        assert torch.equal(dim1, dim2)

    def test_per_dim_sums_to_total(self, kde, query_embedding):
        """La suma de per_dim debe ser igual al total."""
        total, per_dim = kde.log_density(query_embedding)
        assert torch.allclose(total, per_dim.sum(), rtol=1e-5)

    def test_all_support_elements_positive_relative(self, kde, support_embeddings):
        """Todos los elementos del support deben tener log-densidad > outlier extremo."""
        extreme_outlier = torch.randn(768) * 100.0
        outlier_ld, _ = kde.log_density(extreme_outlier)
        for i in range(len(support_embeddings)):
            ld, _ = kde.log_density(support_embeddings[i])
            assert ld.item() > outlier_ld.item()


class TestPerDimensionKDEEdgeCases:
    """Tests de casos borde."""

    def test_single_support_element(self):
        """KDE con k=1 debe funcionar sin errores."""
        from opcion_A.fd_kde import PerDimensionKDE
        single = torch.randn(1, 768)
        kde = PerDimensionKDE(single)
        total, per_dim = kde.log_density(torch.randn(768))
        assert total.dim() == 0
        assert per_dim.shape == (768,)

    def test_query_as_1d_tensor(self, kde):
        """Query como (768,) debe funcionar igual que (1, 768)."""
        q = torch.randn(768)
        total, per_dim = kde.log_density(q)
        assert total.dim() == 0

    def test_query_as_2d_tensor(self, kde):
        """Query como (1, 768) debe funcionar igual que (768,)."""
        q = torch.randn(1, 768)
        total, per_dim = kde.log_density(q)
        assert total.dim() == 0
