"""
Fabric subsystem defines the spatial substrate and underlying storage for
field quantities and species mixtures.

Fabric is responsible for allocating numpy arrays of appropriate size
and type for the continuous scalar and vector fields used in the
simulation (density, momentum, heat, radiation, etc.). It also stores
sparse per-cell species mixtures and black hole masks. Fabric does not
implement dynamics by itself; it provides convenient accessors for
reading and writing field values and defines utility functions for
boundary handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

from .config import EngineConfig
from .types import Cell


@dataclass
class Mixture:
    """Sparse mixture representation for species concentrations in a cell.

    Each mixture holds at most ``K`` ``SpeciesAmount`` entries. The
    simulation ensures that only the top K species by mass in a cell are
    stored explicitly; any remaining mass is discarded or merged to
    reduce state complexity. ``mass`` values represent absolute mass
    (units of density per cell); they are not normalised fractions. The
    sum of masses across the mixture equals the cell's total density.
    """
    species_ids: List[str]
    masses: List[float]

    def total_mass(self) -> float:
        return float(sum(self.masses))

    def normalise(self, target_total: float) -> None:
        """Scale the masses so that they sum to ``target_total``.

        Useful after advection or mass redistribution when the mixture
        mass does not match the cell's density exactly. Zeros are
        preserved.
        """
        total = self.total_mass()
        if total <= 0 or target_total <= 0:
            return
        s = target_total / total
        for idx in range(len(self.masses)):
            self.masses[idx] *= s

    def get_weighted_property(self, prop_table: dict[str, float], prop_name: str) -> float:
        """Compute a weighted average of a property for all species present.

        ``prop_table`` is expected to map species ids to the property in
        question. If a species is unknown, a default value of 0.0 is
        returned. This function is used to compute effective values
        (e.g. ``rho_max,eff``) for EOS calculations.
        """
        total = self.total_mass()
        if total <= 0:
            return 0.0
        accum = 0.0
        for sid, mass in zip(self.species_ids, self.masses):
            value = prop_table.get(sid, 0.0)
            accum += mass * value
        return accum / total

    def add_species_mass(self, species_id: str, mass: float, max_k: int) -> None:
        """Add a species mass to the mixture, respecting the top K limit.

        If the species already exists, its mass is incremented. If not and
        there is space, a new entry is appended. If the mixture is full,
        the species with the smallest mass is replaced if the new mass is
        larger. This keeps the mixture focused on the dominant species.
        """
        if mass <= 0:
            return
        # If species exists, update mass
        for idx, sid in enumerate(self.species_ids):
            if sid == species_id:
                self.masses[idx] += mass
                return
        # else, consider adding or replacing
        if len(self.species_ids) < max_k:
            self.species_ids.append(species_id)
            self.masses.append(mass)
        else:
            # find smallest mass
            min_idx = 0
            min_mass = self.masses[0]
            for idx, mval in enumerate(self.masses):
                if mval < min_mass:
                    min_mass = mval
                    min_idx = idx
            if mass > min_mass:
                # replace smallest
                self.species_ids[min_idx] = species_id
                self.masses[min_idx] = mass

    def cleanup(self, eps: float, max_k: int) -> None:
        """Remove negligible masses and trim list to at most ``max_k`` entries.
        """
        # Remove near zero entries
        new_ids = []
        new_masses = []
        for sid, mass in zip(self.species_ids, self.masses):
            if mass > eps:
                new_ids.append(sid)
                new_masses.append(mass)
        # If still too many species, keep largest
        if len(new_ids) > max_k:
            # sort by mass descending and keep top max_k
            sorted_items = sorted(zip(new_ids, new_masses), key=lambda x: -x[1])[:max_k]
            new_ids, new_masses = zip(*sorted_items)
            new_ids = list(new_ids)
            new_masses = list(new_masses)
        self.species_ids = new_ids
        self.masses = new_masses


class Fabric:
    """Spatial field storage for the Stratum engine.

    The Fabric holds all continuous scalar and vector fields as numpy
    arrays. Each field is initialised according to the grid size in
    ``EngineConfig``. Mixtures are stored as a separate per-cell
    structure. Access to individual fields is provided via attributes.
    """

    def __init__(self, config: EngineConfig):
        self.cfg = config
        W, H = config.grid_w, config.grid_h
        # continuous scalar fields
        self.rho = np.zeros((W, H), dtype=np.float64)
        self.px = np.zeros((W, H), dtype=np.float64)
        self.py = np.zeros((W, H), dtype=np.float64)
        self.E_heat = np.zeros((W, H), dtype=np.float64)
        self.E_rad = np.zeros((W, H), dtype=np.float64)
        self.influence = np.zeros((W, H), dtype=np.float64)
        self.BH_mass = np.zeros((W, H), dtype=np.float64)
        self.EH_mask = np.zeros((W, H), dtype=np.float64)
        # mixtures per cell
        self.mixtures: list[list[Mixture]] = [[Mixture([], []) for _ in range(H)] for _ in range(W)]

    def reset_influence(self) -> None:
        self.influence.fill(0.0)

    def reset_event_horizon(self) -> None:
        self.EH_mask.fill(0.0)

    def boundary_coord(self, i: int, j: int) -> tuple[int, int]:
        """Return a valid coordinate according to the boundary mode.

        If ``boundary`` in ``EngineConfig`` is PERIODIC then indices
        wrap around. If REFLECTIVE they are clamped. If OPEN then
        indices outside the grid are returned as‑is (and callers must
        check before accessing arrays).
        """
        W, H = self.cfg.grid_w, self.cfg.grid_h
        if self.cfg.boundary == "PERIODIC":
            return i % W, j % H
        elif self.cfg.boundary == "REFLECTIVE":
            return max(0, min(i, W - 1)), max(0, min(j, H - 1))
        else:
            return i, j

    def gradient_scalar(self, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute finite difference gradient of a scalar field.

        Returns two arrays ``grad_x`` and ``grad_y`` of the same shape
        as ``field``. Uses simple central differences for interior cells
        and one‑sided differences at boundaries consistent with the
        configured boundary condition. For efficiency, numpy operations
        are used instead of explicit Python loops.
        """
        W, H = self.cfg.grid_w, self.cfg.grid_h
        grad_x = np.zeros_like(field)
        grad_y = np.zeros_like(field)
        # interior: central differences
        grad_x[1:-1, :] = (field[2:, :] - field[:-2, :]) * 0.5
        grad_y[:, 1:-1] = (field[:, 2:] - field[:, :-2]) * 0.5
        # boundaries: periodic or reflective
        if self.cfg.boundary == "PERIODIC":
            grad_x[0, :] = (field[1, :] - field[-1, :]) * 0.5
            grad_x[-1, :] = (field[0, :] - field[-2, :]) * 0.5
            grad_y[:, 0] = (field[:, 1] - field[:, -1]) * 0.5
            grad_y[:, -1] = (field[:, 0] - field[:, -2]) * 0.5
        elif self.cfg.boundary == "REFLECTIVE":
            # one sided difference at boundaries
            grad_x[0, :] = field[1, :] - field[0, :]
            grad_x[-1, :] = field[-1, :] - field[-2, :]
            grad_y[:, 0] = field[:, 1] - field[:, 0]
            grad_y[:, -1] = field[:, -1] - field[:, -2]
        else:
            # OPEN: treat outside values as zero
            grad_x[0, :] = field[1, :] - field[0, :]
            grad_x[-1, :] = -field[-1, :]
            grad_y[:, 0] = field[:, 1] - field[:, 0]
            grad_y[:, -1] = -field[:, -1]
        return grad_x, grad_y

    def divergence_vector(self, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
        """Compute divergence of a vector field given separate components.

        Uses central differences for interior cells and simple differences
        at boundaries following the boundary condition. Returns an array
        of the same shape as the input fields.
        """
        W, H = self.cfg.grid_w, self.cfg.grid_h
        div = np.zeros((W, H), dtype=vx.dtype)
        div[1:-1, :] = (vx[2:, :] - vx[:-2, :]) * 0.5
        div[:, 1:-1] += (vy[:, 2:] - vy[:, :-2]) * 0.5
        # boundaries
        if self.cfg.boundary == "PERIODIC":
            div[0, :] += (vx[1, :] - vx[-1, :]) * 0.5
            div[-1, :] += (vx[0, :] - vx[-2, :]) * 0.5
            div[:, 0] += (vy[:, 1] - vy[:, -1]) * 0.5
            div[:, -1] += (vy[:, 0] - vy[:, -2]) * 0.5
        elif self.cfg.boundary == "REFLECTIVE":
            div[0, :] += vx[1, :] - vx[0, :]
            div[-1, :] += vx[-1, :] - vx[-2, :]
            div[:, 0] += vy[:, 1] - vy[:, 0]
            div[:, -1] += vy[:, -1] - vy[:, -2]
        else:
            div[0, :] += vx[1, :] - vx[0, :]
            div[-1, :] += -vx[-1, :]
            div[:, 0] += vy[:, 1] - vy[:, 0]
            div[:, -1] += -vy[:, -1]
        return div