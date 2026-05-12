"""
Fixtures compartidos para tests del pipeline MedGemma-Seg.

Siguiendo patrón AAA (Arrange-Act-Assert) de python-testing-patterns.
"""

import pytest
import torch
import numpy as np

from opcion_A.fd_kde import PerDimensionKDE
from opcion_A.ood_window import OODWindow
from opcion_A.fd_uncertainty import FDUncertaintyEstimator


# ── Fixtures de tensores ──────────────────────────────────────────────

@pytest.fixture
def seed():
    """Semilla reproducible para todos los tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def support_embeddings(seed):
    """
    Support set de k=9 embeddings 768-dim.
    Simula 9 máscaras etiquetadas como 'catarata' extraídas por MedSigLIP.
    """
    k = 9
    dim = 768
    # Centroide conocido + ruido gaussiano pequeño
    centroid = torch.randn(dim)
    embeddings = centroid + 0.1 * torch.randn(k, dim)
    return embeddings


@pytest.fixture
def query_embedding(seed):
    """
    Embedding de una máscara candidata (query).
    Cercano al centroide del support set → debería tener alta log-densidad.
    """
    dim = 768
    return torch.randn(dim)


@pytest.fixture
def query_outlier(seed):
    """
    Embedding de una máscara claramente fuera de distribución.
    Lejos del centroide → debería tener baja log-densidad (fuera de ventana OOD).
    """
    dim = 768
    return torch.randn(dim) * 10.0  # Escala 10x → outlier


@pytest.fixture
def sample_masks():
    """
    Lista de 3 máscaras binarias fake (64×64 para test rápido).
    Simula las candidatas de SAM 2.
    """
    masks = []
    for i in range(3):
        mask = np.zeros((64, 64), dtype=np.uint8)
        # Diferente región activa por máscara
        r = 10 + i * 5
        mask[r:r + 30, r:r + 30] = 1
        masks.append(mask)
    return masks


# ── Fixtures de componentes del pipeline ──────────────────────────────

@pytest.fixture
def kde(support_embeddings):
    """Instancia de PerDimensionKDE calibrada sobre el support set."""
    return PerDimensionKDE(support_embeddings)


@pytest.fixture
def ood_window_iqr():
    """Ventana OOD sin calibrar (método IQR)."""
    return OODWindow(method='iqr')


@pytest.fixture
def ood_window_minmax():
    """Ventana OOD sin calibrar (método MinMax)."""
    return OODWindow(method='minmax')


@pytest.fixture
def ood_window_calibrated(support_embeddings, kde):
    """
    Ventana OOD calibrada sobre el support set.
    Calcula log-densidades del support y calibra con IQR.
    """
    window = OODWindow(method='iqr')
    log_densities = []
    for i in range(len(support_embeddings)):
        ld, _ = kde.log_density(support_embeddings[i])
        log_densities.append(ld)
    window.calibrate(torch.tensor(log_densities))
    return window


@pytest.fixture
def fd_estimator():
    """Instancia de FDUncertaintyEstimator con pesos default."""
    return FDUncertaintyEstimator(alpha=0.5, beta=0.3, gamma=0.2, tau=0.6)
