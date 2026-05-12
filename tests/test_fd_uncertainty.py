"""
Tests para FDUncertaintyEstimator — Fase 2, Tarea 2.4.

Valida:
- Las 7 señales están en [0, 1]
- U_composite está en [0, 1]
- Pesos suman 1.0
- Señales individuales con inputs controlados
"""

import torch
import pytest


class TestFDUncertaintyWeights:
    """Tests de configuración de pesos."""

    def test_weights_sum_to_one(self):
        """Los 7 pesos deben sumar 1.0."""
        from opcion_A.fd_uncertainty import FDUncertaintyEstimator
        total = sum(FDUncertaintyEstimator.WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_seven_signals(self):
        """Debe haber exactamente 7 señales."""
        from opcion_A.fd_uncertainty import FDUncertaintyEstimator
        assert len(FDUncertaintyEstimator.WEIGHTS) == 7

    def test_weight_values(self):
        """Los pesos deben coincidir con el spec: 0.25, 0.20, 0.15, 0.10, 0.15, 0.10, 0.05."""
        from opcion_A.fd_uncertainty import FDUncertaintyEstimator
        w = FDUncertaintyEstimator.WEIGHTS
        assert w['U_LD'] == 0.25
        assert w['U_BORDE'] == 0.20
        assert w['U_DIMVAR'] == 0.15
        assert w['U_PROTO'] == 0.10
        assert w['U_acuerdo'] == 0.15
        assert w['U_magnitud'] == 0.10
        assert w['U_margen'] == 0.05


class TestFDUncertaintyULD:
    """Tests de la señal U_LD (Log-Density)."""

    def test_u_ld_below_min_is_one(self, fd_estimator, ood_window_calibrated):
        """Log-density debajo de theta_min → U_LD = 1.0 (total incertidumbre)."""
        very_low = ood_window_calibrated.theta_min - 100.0
        result = fd_estimator._u_ld(very_low, ood_window_calibrated)
        assert result == 1.0

    def test_u_ld_above_max_is_zero(self, fd_estimator, ood_window_calibrated):
        """Log-density encima de theta_max → U_LD = 0.0 (certidumbre total)."""
        very_high = ood_window_calibrated.theta_max + 100.0
        result = fd_estimator._u_ld(very_high, ood_window_calibrated)
        assert result == 0.0

    def test_u_ld_at_midpoint_is_half(self, fd_estimator, ood_window_calibrated):
        """Log-density en el medio de la ventana → U_LD ≈ 0.5."""
        mid = (ood_window_calibrated.theta_min + ood_window_calibrated.theta_max) / 2
        result = fd_estimator._u_ld(mid.item(), ood_window_calibrated)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_u_ld_in_range_0_to_1(self, fd_estimator, ood_window_calibrated):
        """U_LD debe estar en [0, 1] para cualquier valor dentro de la ventana."""
        for offset in [0.0, 0.25, 0.5, 0.75, 1.0]:
            val = ood_window_calibrated.theta_min + offset * (
                ood_window_calibrated.theta_max - ood_window_calibrated.theta_min
            )
            result = fd_estimator._u_ld(val.item(), ood_window_calibrated)
            assert 0.0 <= result <= 1.0


class TestFDUncertaintyUBorde:
    """Tests de la señal U_BORDE."""

    def test_u_borde_at_center_is_zero(self, fd_estimator, ood_window_calibrated):
        """En el centro exacto de la ventana → U_BORDE = 0 (más seguro)."""
        mid = (ood_window_calibrated.theta_min + ood_window_calibrated.theta_max) / 2
        result = fd_estimator._u_borde(mid.item(), ood_window_calibrated)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_u_borde_at_boundary_is_one(self, fd_estimator, ood_window_calibrated):
        """En el borde de la ventana → U_BORDE = 1 (más incierto)."""
        result_min = fd_estimator._u_borde(
            ood_window_calibrated.theta_min.item(), ood_window_calibrated
        )
        result_max = fd_estimator._u_borde(
            ood_window_calibrated.theta_max.item(), ood_window_calibrated
        )
        assert result_min == pytest.approx(1.0, abs=0.01)
        assert result_max == pytest.approx(1.0, abs=0.01)

    def test_u_borde_range_0_to_1(self, fd_estimator, ood_window_calibrated):
        """U_BORDE debe estar en [0, 1]."""
        for offset in [0.0, 0.25, 0.5, 0.75, 1.0]:
            val = ood_window_calibrated.theta_min + offset * (
                ood_window_calibrated.theta_max - ood_window_calibrated.theta_min
            )
            result = fd_estimator._u_borde(val.item(), ood_window_calibrated)
            assert 0.0 <= result <= 1.0 + 1e-6


class TestFDUncertaintyUDimvar:
    """Tests de la señal U_DIMVAR."""

    def test_u_dimvar_range_0_to_1(self, fd_estimator, kde, query_embedding):
        """U_DIMVAR debe estar en [0, 1]."""
        _, per_dim = kde.log_density(query_embedding)
        result = fd_estimator._u_dimvar(per_dim)
        assert 0.0 <= result <= 1.0

    def test_u_dimvar_low_contrib_high_uncertainty(self, fd_estimator):
        """Contribuciones bajas → mayor incertidumbre."""
        low_contrib = torch.zeros(768)
        high_contrib = torch.ones(768) * 5.0
        u_low = fd_estimator._u_dimvar(low_contrib)
        u_high = fd_estimator._u_dimvar(high_contrib)
        # Menor contribución → mayor incertidumbre
        assert u_low >= u_high


class TestFDUncertaintyUProto:
    """Tests de la señal U_PROTO."""

    def test_u_proto_near_centroid_low_uncertainty(self, fd_estimator, support_embeddings):
        """Query cerca del centroide → baja incertidumbre."""
        proto = support_embeddings.mean(dim=0)
        # Query = centroide exacto → distancia 0
        result = fd_estimator._u_proto(proto, support_embeddings)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_u_proto_far_centroid_high_uncertainty(self, fd_estimator, support_embeddings):
        """Query lejos del centroide → alta incertidumbre."""
        far_query = torch.randn(768) * 100.0
        result = fd_estimator._u_proto(far_query, support_embeddings)
        near_query = support_embeddings.mean(dim=0)
        near_result = fd_estimator._u_proto(near_query, support_embeddings)
        assert result > near_result

    def test_u_proto_range_0_to_1(self, fd_estimator, support_embeddings):
        """U_PROTO debe estar en [0, 1]."""
        query = torch.randn(768)
        result = fd_estimator._u_proto(query, support_embeddings)
        assert 0.0 <= result <= 1.0


class TestFDUncertaintyUAcuerdo:
    """Tests de la señal U_acuerdo."""

    def test_u_acuerdo_identical_scores_is_zero(self, fd_estimator):
        """Scores idénticos → sin desacuerdo → U_acuerdo ≈ 0."""
        result = fd_estimator._u_acuerdo(0.8, 0.8, 0.8)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_u_acuerdo_divergent_scores_is_high(self, fd_estimator):
        """Scores muy diferentes → alto desacuerdo."""
        result = fd_estimator._u_acuerdo(0.9, 0.1, 0.5)
        assert result > 0.1

    def test_u_acuerdo_range_0_to_1(self, fd_estimator):
        """U_acuerdo debe estar en [0, 1]."""
        for s_fsl, s_cos, s_sam in [(0.5, 0.5, 0.5), (0.9, 0.1, 0.5), (0.0, 1.0, 0.0)]:
            result = fd_estimator._u_acuerdo(s_fsl, s_cos, s_sam)
            assert 0.0 <= result <= 1.0 + 1e-6


class TestFDUncertaintyEstimate:
    """Tests del método estimate (composición de las 7 señales)."""

    def test_estimate_returns_two_values(self, fd_estimator, query_embedding,
                                         kde, ood_window_calibrated, support_embeddings):
        """estimate debe retornar (u_composite, components)."""
        u, components = fd_estimator.estimate(
            query_embedding, kde, ood_window_calibrated,
            support_embeddings, s_fsl=0.8, s_cos=0.7, s_sam=0.9,
        )
        assert isinstance(u, float)
        assert isinstance(components, dict)

    def test_estimate_components_has_7_signals(self, fd_estimator, query_embedding,
                                                kde, ood_window_calibrated, support_embeddings):
        """Components debe tener las 7 señales."""
        _, components = fd_estimator.estimate(
            query_embedding, kde, ood_window_calibrated,
            support_embeddings, s_fsl=0.8, s_cos=0.7, s_sam=0.9,
        )
        expected_keys = {'U_LD', 'U_BORDE', 'U_DIMVAR', 'U_PROTO',
                         'U_acuerdo', 'U_magnitud', 'U_margen'}
        assert set(components.keys()) == expected_keys

    def test_estimate_composite_in_range(self, fd_estimator, query_embedding,
                                          kde, ood_window_calibrated, support_embeddings):
        """U_composite debe estar en [0, 1]."""
        u, _ = fd_estimator.estimate(
            query_embedding, kde, ood_window_calibrated,
            support_embeddings, s_fsl=0.8, s_cos=0.7, s_sam=0.9,
        )
        assert 0.0 <= u <= 1.0

    def test_estimate_high_confidence_low_uncertainty(self, fd_estimator,
                                                       support_embeddings,
                                                       kde, ood_window_calibrated):
        """Query dentro del support con scores altos → baja incertidumbre."""
        query = support_embeddings.mean(dim=0)  # centroide
        u, _ = fd_estimator.estimate(
            query, kde, ood_window_calibrated,
            support_embeddings, s_fsl=0.9, s_cos=0.9, s_sam=0.9,
        )
        assert u < 0.5  # Debe ser baja incertidumbre

    def test_estimate_outlier_high_uncertainty(self, fd_estimator, query_outlier,
                                                kde, ood_window_calibrated,
                                                support_embeddings):
        """Outlier con scores bajos → alta incertidumbre."""
        u, _ = fd_estimator.estimate(
            query_outlier, kde, ood_window_calibrated,
            support_embeddings, s_fsl=0.1, s_cos=0.1, s_sam=0.1,
        )
        assert u > 0.3  # Debe ser alta incertidumbre
