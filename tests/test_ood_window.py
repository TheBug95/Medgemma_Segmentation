"""
Tests para OODWindow — Fase 2, Tarea 2.2.

Valida:
- Calibración IQR: [Q1 - 1.5·IQR, Q3 + 1.5·IQR]
- Calibración MinMax: [min, max]
- is_in_window y normalize
- Manejo de casos borde
"""

import torch
import pytest


class TestOODWindowCalibration:
    """Tests de calibración de la ventana OOD."""

    def test_iqr_calibration_bounds(self, ood_window_iqr):
        """Método IQR debe generar theta_min < theta_max."""
        data = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
        ood_window_iqr.calibrate(data)
        assert ood_window_iqr.theta_min < ood_window_iqr.theta_max

    def test_iqr_formula(self):
        """IQR debe ser Q3 - Q1 y los bounds Q1-1.5*IQR, Q3+1.5*IQR."""
        from opcion_A.ood_window import OODWindow
        window = OODWindow(method='iqr')
        # Datos simétricos para facilitar verificación manual
        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        window.calibrate(data)
        q1 = torch.quantile(data, 0.25)
        q3 = torch.quantile(data, 0.75)
        iqr = q3 - q1
        expected_min = q1 - 1.5 * iqr
        expected_max = q3 + 1.5 * iqr
        assert window.theta_min == pytest.approx(expected_min.item(), rel=1e-5)
        assert window.theta_max == pytest.approx(expected_max.item(), rel=1e-5)

    def test_minmax_calibration(self):
        """Método MinMax debe usar min y max directamente."""
        from opcion_A.ood_window import OODWindow
        window = OODWindow(method='minmax')
        data = torch.tensor([5.0, 15.0, 25.0, 35.0, 45.0])
        window.calibrate(data)
        assert window.theta_min == pytest.approx(5.0)
        assert window.theta_max == pytest.approx(45.0)

    def test_iqr_wider_than_minmax(self, support_embeddings):
        """IQR debe ser más amplio que MinMax (más permisivo)."""
        from opcion_A.fd_kde import PerDimensionKDE
        from opcion_A.ood_window import OODWindow

        kde = PerDimensionKDE(support_embeddings)
        lds = []
        for i in range(len(support_embeddings)):
            ld, _ = kde.log_density(support_embeddings[i])
            lds.append(ld)
        lds_tensor = torch.tensor(lds)

        w_iqr = OODWindow(method='iqr')
        w_minmax = OODWindow(method='minmax')
        w_iqr.calibrate(lds_tensor)
        w_minmax.calibrate(lds_tensor)

        iqr_range = w_iqr.theta_max - w_iqr.theta_min
        minmax_range = w_minmax.theta_max - w_minmax.theta_min
        assert iqr_range >= minmax_range

    def test_invalid_method_raises(self):
        """Método inválido debe lanzar ValueError."""
        from opcion_A.ood_window import OODWindow
        window = OODWindow(method='invalid')
        with pytest.raises(ValueError, match="Método desconocido"):
            window.calibrate(torch.tensor([1.0, 2.0, 3.0]))


class TestOODWindowIsInWindow:
    """Tests de is_in_window."""

    def test_value_inside_window(self, ood_window_calibrated, support_embeddings):
        """Un elemento del support set debe estar dentro de la ventana."""
        from opcion_A.fd_kde import PerDimensionKDE
        kde = PerDimensionKDE(support_embeddings)
        ld, _ = kde.log_density(support_embeddings[0])
        assert ood_window_calibrated.is_in_window(ld.item()) is True

    def test_value_below_min(self, ood_window_calibrated):
        """Valor muy bajo debe estar fuera de la ventana."""
        very_low = ood_window_calibrated.theta_min - 1000.0
        assert ood_window_calibrated.is_in_window(very_low) is False

    def test_value_above_max(self, ood_window_calibrated):
        """Valor muy alto debe estar fuera de la ventana."""
        very_high = ood_window_calibrated.theta_max + 1000.0
        assert ood_window_calibrated.is_in_window(very_high) is False

    def test_boundary_values_inclusive(self, ood_window_calibrated):
        """Los valores exactos en los bounds deben estar dentro."""
        assert ood_window_calibrated.is_in_window(
            ood_window_calibrated.theta_min.item()
        ) is True
        assert ood_window_calibrated.is_in_window(
            ood_window_calibrated.theta_max.item()
        ) is True


class TestOODWindowNormalize:
    """Tests de normalize."""

    def test_normalize_midpoint(self, ood_window_calibrated):
        """El punto medio de la ventana debe normalizarse a ~0.5."""
        mid = (ood_window_calibrated.theta_min + ood_window_calibrated.theta_max) / 2
        normalized = ood_window_calibrated.normalize(mid.item())
        assert normalized == pytest.approx(0.5, abs=0.01)

    def test_normalize_below_min_returns_zero(self, ood_window_calibrated):
        """Valor debajo de theta_min debe retornar 0.0."""
        very_low = ood_window_calibrated.theta_min - 100.0
        assert ood_window_calibrated.normalize(very_low.item()) == 0.0

    def test_normalize_above_max_returns_one(self, ood_window_calibrated):
        """Valor encima de theta_max debe retornar 1.0."""
        very_high = ood_window_calibrated.theta_max + 100.0
        assert ood_window_calibrated.normalize(very_high.item()) == 1.0

    def test_normalize_at_min_returns_zero(self, ood_window_calibrated):
        """Valor exacto en theta_min debe retornar 0.0."""
        assert ood_window_calibrated.normalize(
            ood_window_calibrated.theta_min.item()
        ) == 0.0

    def test_normalize_at_max_returns_one(self, ood_window_calibrated):
        """Valor exacto en theta_max debe retornar 1.0."""
        assert ood_window_calibrated.normalize(
            ood_window_calibrated.theta_max.item()
        ) == 1.0

    def test_normalize_range_0_to_1(self, ood_window_calibrated, support_embeddings):
        """Todos los elementos del support deben normalizarse en [0, 1]."""
        from opcion_A.fd_kde import PerDimensionKDE
        kde = PerDimensionKDE(support_embeddings)
        for i in range(len(support_embeddings)):
            ld, _ = kde.log_density(support_embeddings[i])
            norm = ood_window_calibrated.normalize(ld.item())
            assert 0.0 <= norm <= 1.0
