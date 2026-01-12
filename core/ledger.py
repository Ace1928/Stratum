"""
Ledger subsystem responsible for conservation, energy bookkeeping and
stochastic barrier sampling.

This module implements the accounting rules that maintain energy
conservation to the extent allowed by dissipative terms, provides
helpers for computing kinetic energy and converting energy between
forms, and offers barrier evaluation with optional stochastic
temperature tunnelling. All random draws are channelled through the
``EntropySource`` class to ensure deterministic replays and auditing.

The entropy source uses blake2s for stable hash derivation, ensuring
deterministic behavior across Python interpreter sessions and versions.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .types import Vec2, dot, clamp

# Numerical stability constants
# These limits are chosen to be well within float64 range (~1e308) while
# leaving headroom for intermediate calculations.

#: Minimum density threshold below which mass is considered negligible
RHO_EPSILON = 1e-12

#: Maximum density for kinetic energy calculation to prevent overflow
RHO_MAX_CLAMP = 1e150

#: Maximum squared momentum sum before clamping
MOMENTUM_SQ_MAX = 1e300

#: Maximum kinetic energy result
KINETIC_ENERGY_MAX = 1e100


@dataclass
class EntropyRecord:
    """Record of a single entropy sample for replay support.
    
    Attributes:
        checkpoint_id: Identifier for the sampling checkpoint (e.g., "barrier:fusion").
        tick: Simulation tick when the sample was drawn.
        cell: Grid coordinates (i, j) of the cell.
        attempt: Microtick attempt index within the cell.
        conditioning_summary: Hash summary of conditioning parameters.
        value: The sampled value in [0, 1].
    """
    checkpoint_id: str
    tick: int
    cell: tuple[int, int]
    attempt: int
    conditioning_summary: int
    value: float


class EntropySource:
    """Centralised source of pseudorandomness with replay support.
    
    This class provides deterministic random number generation that is
    stable across Python interpreter sessions. It uses blake2s for seed
    derivation, ensuring that the same inputs always produce the same
    outputs regardless of PYTHONHASHSEED settings.
    
    Attributes:
        base_seed: The base seed for deterministic generation.
        entropy_mode: If True, adds run-specific salt for variation.
        replay_mode: If True, records samples for later replay.
        run_salt: Random salt added when entropy_mode is True.
        replay_log: List of recorded samples when replay_mode is True.
        replay_cursor: Current position in replay_log during replay.
    """

    def __init__(self, base_seed: int, entropy_mode: bool = False, replay_mode: bool = False):
        """Initialize the entropy source.
        
        Args:
            base_seed: Base seed for deterministic generation.
            entropy_mode: If True, add run-specific variation.
            replay_mode: If True, record samples for replay.
        """
        self.base_seed = base_seed
        self.entropy_mode = entropy_mode
        self.replay_mode = replay_mode
        self.run_salt: int = np.random.randint(0, 2**31 - 1) if entropy_mode else 0
        self.replay_log: list[EntropyRecord] = []
        self.replay_cursor: int = 0

    def _derive_seed(self, checkpoint_id: str, tick: int, cell: tuple[int, int], attempt: int, cond_hash: int) -> int:
        """Derive a deterministic seed using blake2s hash.
        
        This method produces stable seeds across Python interpreter sessions
        by using blake2s instead of Python's built-in hash() function.
        
        Args:
            checkpoint_id: String identifier for the checkpoint.
            tick: Current simulation tick.
            cell: Grid cell coordinates (i, j).
            attempt: Microtick attempt index.
            cond_hash: Hash of conditioning parameters.
            
        Returns:
            A 64-bit integer seed suitable for numpy's random generator.
        """
        # Build a canonical byte representation of all seed components
        # Use blake2s for incremental hashing to avoid struct packing issues
        h = hashlib.blake2s(digest_size=8)
        
        # Add each component as bytes in a canonical format
        # Convert to Python int to ensure to_bytes works for numpy types
        h.update(int(self.base_seed).to_bytes(8, byteorder='big', signed=True))
        h.update(int(self.run_salt).to_bytes(8, byteorder='big', signed=False))
        h.update(int(tick).to_bytes(8, byteorder='big', signed=True))
        h.update(int(cell[0]).to_bytes(4, byteorder='big', signed=True))
        h.update(int(cell[1]).to_bytes(4, byteorder='big', signed=True))
        h.update(int(attempt).to_bytes(4, byteorder='big', signed=True))
        # cond_hash is unsigned 64-bit from blake2s
        h.update((int(cond_hash) & 0xFFFFFFFFFFFFFFFF).to_bytes(8, byteorder='big', signed=False))
        # Add checkpoint_id as UTF-8 bytes
        h.update(checkpoint_id.encode('utf-8'))
        
        # Convert 8-byte digest to unsigned 64-bit integer
        return int.from_bytes(h.digest(), byteorder='big', signed=False)

    def _hash_conditioning(self, conditioning: Dict[str, float]) -> int:
        """Compute a stable hash of conditioning parameters.
        
        Uses blake2s to hash the sorted key-value pairs, ensuring
        deterministic results across Python sessions.
        
        Args:
            conditioning: Dictionary of conditioning parameters.
            
        Returns:
            A 64-bit integer hash of the conditioning dict.
        """
        if not conditioning:
            return 0
        
        # Build canonical representation: sorted key-value pairs
        # Round floats to 6 decimal places for stability
        h = hashlib.blake2s(digest_size=8)
        for k in sorted(conditioning.keys()):
            v = conditioning[k]
            # Pack key and rounded value as bytes
            h.update(k.encode('utf-8'))
            # Use struct.pack for double precision float
            h.update(struct.pack('>d', round(v, 6)))
        
        return int.from_bytes(h.digest(), byteorder='big', signed=False)

    def sample_uniform(self, checkpoint_id: str, tick: int, cell: tuple[int, int], attempt: int, conditioning: Dict[str, float]) -> float:
        """Return a uniform random sample in [0,1].

        The sample is deterministic given the base seed, run salt,
        checkpoint id and conditioning values. When ``replay_mode`` is
        enabled, previously recorded samples are returned instead of
        generating new ones.
        
        This method uses blake2s for hash derivation, ensuring stable
        results across Python interpreter sessions regardless of
        PYTHONHASHSEED settings.
        
        Args:
            checkpoint_id: Identifier for the sampling checkpoint.
            tick: Current simulation tick.
            cell: Grid cell coordinates (i, j).
            attempt: Microtick attempt index.
            conditioning: Dictionary of conditioning parameters.
            
        Returns:
            A uniform random sample in [0, 1].
        """
        if self.replay_mode and self.replay_cursor < len(self.replay_log):
            rec = self.replay_log[self.replay_cursor]
            self.replay_cursor += 1
            return rec.value
        
        # Compute stable hash of conditioning parameters
        cond_hash = self._hash_conditioning(conditioning)
        
        # Derive seed using blake2s
        seed = self._derive_seed(checkpoint_id, tick, cell, attempt, cond_hash)
        
        # Use numpy's random generator with the derived seed
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
        if rho <= RHO_EPSILON:
            # negligible mass: treat kinetic energy as zero
            return 0.0
        # Clamp rho to prevent overflow in the division
        rho_clamped = min(rho, RHO_MAX_CLAMP)
        # Compute squared momentum magnitude using multiplication to
        # avoid ``**`` overflow.  Use Python floats (double precision).
        px = float(p.x)
        py = float(p.y)
        sum_sq = px * px + py * py
        # Clamp to a large finite value to avoid inf during division.
        if not math.isfinite(sum_sq) or sum_sq > MOMENTUM_SQ_MAX:
            sum_sq = MOMENTUM_SQ_MAX
        result = sum_sq / (2.0 * rho_clamped)
        # Final clamp to ensure result is finite
        if not math.isfinite(result) or result > KINETIC_ENERGY_MAX:
            return KINETIC_ENERGY_MAX
        return result

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