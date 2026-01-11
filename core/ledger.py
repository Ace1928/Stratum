"""
Ledger subsystem responsible for conservation, energy bookkeeping and
stochastic barrier sampling.

This module implements the accounting rules that maintain energy
conservation to the extent allowed by dissipative terms, provides
helpers for computing kinetic energy and converting energy between
forms, and offers barrier evaluation with optional stochastic
temperature tunnelling. All random draws are channelled through the
``EntropySource`` class to ensure deterministic replays and auditing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .types import Vec2, dot, clamp


@dataclass
class EntropyRecord:
    checkpoint_id: str
    tick: int
    cell: tuple[int, int]
    attempt: int
    conditioning_summary: int
    value: float


class EntropySource:
    """Centralised source of pseudorandomness with replay support."""

    def __init__(self, base_seed: int, entropy_mode: bool = False, replay_mode: bool = False):
        self.base_seed = base_seed
        self.entropy_mode = entropy_mode
        self.replay_mode = replay_mode
        self.run_salt: int = np.random.randint(0, 2**31 - 1) if entropy_mode else 0
        self.replay_log: list[EntropyRecord] = []
        self.replay_cursor: int = 0

    def _derive_seed(self, checkpoint_id: str, tick: int, cell: tuple[int, int], attempt: int, cond_hash: int) -> int:
        # Combine pieces into a 64‑bit integer seed deterministically.
        return (
            hash((self.base_seed, self.run_salt, checkpoint_id, tick, cell, attempt, cond_hash)) & 0xFFFFFFFFFFFF
        )

    def sample_uniform(self, checkpoint_id: str, tick: int, cell: tuple[int, int], attempt: int, conditioning: Dict[str, float]) -> float:
        """Return a uniform random sample in [0,1].

        The sample is deterministic given the base seed, run salt,
        checkpoint id and conditioning values. When ``replay_mode`` is
        enabled, previously recorded samples are returned instead of
        generating new ones.
        """
        if self.replay_mode and self.replay_cursor < len(self.replay_log):
            rec = self.replay_log[self.replay_cursor]
            self.replay_cursor += 1
            return rec.value
        # compute a summary hash of conditioning dict to incorporate into seed
        cond_hash = 0
        for k, v in sorted(conditioning.items()):
            cond_hash ^= hash((k, round(v, 6)))
        seed = self._derive_seed(checkpoint_id, tick, cell, attempt, cond_hash)
        # Use Python's built in PRNG to avoid dependencies
        rng = np.random.default_rng(seed)
        u = float(rng.random())
        if self.replay_mode:
            self.replay_log.append(
                EntropyRecord(
                    checkpoint_id, tick, cell, attempt, cond_hash, u
                )
            )
        return u


class Ledger:
    """Maintain energy conservation and evaluate probabilistic barriers."""

    def __init__(self, fabric, config: 'EngineConfig'):
        from .config import EngineConfig  # type: ignore
        self.fabric = fabric
        self.cfg = config
        self.entropy = EntropySource(config.base_seed, config.entropy_mode, config.replay_mode)

    @staticmethod
    def kinetic_energy(rho: float, p: Vec2) -> float:
        """Return kinetic energy based on mass and momentum.

        This helper computes the scalar kinetic energy in a manner that
        avoids overflow and numerical warnings.  Squaring large
        momenta using the ``**`` operator can result in overflow or
        ``inf`` values.  Instead, we multiply components directly and
        clamp the result to a large but finite value.  When ``rho`` is
        very small, we return zero to prevent division by zero.
        """
        if rho <= 1e-12:
            # negligible mass: treat kinetic energy as zero
            return 0.0
        # Compute squared momentum magnitude using multiplication to
        # avoid ``**`` overflow.  Use Python floats (double precision).
        px = float(p.x)
        py = float(p.y)
        sum_sq = px * px + py * py
        # Clamp to a large finite value to avoid inf during division.
        # 1e300 is within double precision range (~1e308).
        if not math.isfinite(sum_sq) or sum_sq > 1e300:
            sum_sq = 1e300
        return sum_sq / (2.0 * rho)

    def barrier_crossed(
        self,
        event_id: str,
        tick: int,
        cell: tuple[int, int],
        attempt: int,
        E_act: float,
        E_avail: float,
        T: float,
        gate: float,
        kT_scale: float = 1.0,
    ) -> bool:
        """Return True if a barrier event crosses the threshold.

        ``gate`` can be used to modulate the maximum probability (0..1),
        reflecting additional domain specific gating (e.g. fusion affinity).
        ``kT_scale`` controls the effective temperature scaling for
        thermal tunnelling.
        """
        # If available energy meets or exceeds activation energy then the
        # event can occur deterministically subject to gate.
        if E_avail >= E_act:
            return self.entropy.sample_uniform(
                f"barrier:{event_id}", tick, cell, attempt, {"gate": gate}
            ) < gate
        # Otherwise use Arrhenius-like probability: exp(-(E_act - E_avail)/(kT T))
        deficit = E_act - E_avail
        # Avoid division by zero: if T is zero then probability is zero
        if T <= 0:
            return False
        # kT_scale can modulate how easily barriers are crossed
        p = math.exp(-deficit / (kT_scale * T)) * gate
        u = self.entropy.sample_uniform(
            f"barrier:{event_id}", tick, cell, attempt, {"deficit": deficit, "T": T, "gate": gate}
        )
        return u < p

    def convert_kinetic_to_heat(self, i: int, j: int, delta_E: float, rad_fraction: float) -> None:
        """Convert kinetic energy to heat and optional radiation at a grid cell.

        ``rad_fraction`` controls the fraction of lost kinetic energy that
        is converted to radiation. The rest becomes heat. This helper
        enforces that no negative energies appear on the fields.
        """
        if delta_E <= 0:
            return
        # clamp rad_fraction to [0,1]
        rad_fraction = clamp(rad_fraction, 0.0, 1.0)
        heat_delta = delta_E * (1.0 - rad_fraction)
        rad_delta = delta_E * rad_fraction
        self.fabric.E_heat[i, j] += heat_delta
        self.fabric.E_rad[i, j] += rad_delta

    def finalize_tick(self, tick: int) -> None:
        """End-of-tick housekeeping. Currently a no-op.

        This function can be expanded to perform global conservation
        auditing or to dump stats. For the prototype a simple placeholder
        is provided.
        """
        pass