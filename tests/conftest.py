"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import wmr_scenario as Sc  # noqa: E402


@pytest.fixture
def hand():
    """Handbeispiel: fünf Punkte auf einer Geraden, x = (0, 1, 2, 3, 4), y = (−1, −1, +1, −1, +1), nicht trennbar (Handrechnung in README und im Expander)."""
    return Sc.fuenf_punkte()


@pytest.fixture
def zwei_punkte():
    """Zwei Punkte x = ±1 mit Labels ±1: Q = [[1, 1], [1, 1]], Dual 2a − 2a², also α = (1/2, 1/2) für C ≥ 1/2 und α = (C, C) darunter; w = 1, Marge 2 (von Hand)."""
    return Sc.Instance(np.array([[1.0], [-1.0]]), np.array([1.0, -1.0]), "zwei Punkte")
