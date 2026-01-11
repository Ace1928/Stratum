"""
Simulation configuration definitions.

This module defines configuration dataclasses used to parameterise the
simulation. Configurations are defined with explicit defaults so that
test runs can be created easily without requiring the user to supply
values for every field. See ``EngineConfig`` for the top‑level
configuration consumed by the Stratum engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EngineConfig:
    """Top level configuration for Stratum engine runs.

    Fields correspond closely to the configuration table described in
    the specification. Where appropriate, fields include defaults that
    work reasonably for small demonstrations. Users are encouraged to
    override these as needed when constructing a scenario.
    """

    # Grid dimensions
    grid_w: int = 128
    grid_h: int = 128

    # Random seeds and entropy
    base_seed: int = 42
    entropy_mode: bool = False  # if True, inject run salt for fuzziness
    replay_mode: bool = False   # if True, record entropy draws for exact replay

    # Boundary condition
    boundary: str = "PERIODIC"  # PERIODIC, REFLECTIVE or OPEN

    # Propagation speeds (cells per tick)
    v_max: float = 5.0
    v_influence: float = 2.0
    v_radiation: float = 5.0

    # Time and compute budgets
    tick_budget_ms: float = 50.0  # approximate time slice per tick
    degrade_first: bool = True
    microtick_cap_per_region: int = 10
    active_region_max: int = 2048

    # Mixture handling
    mixture_top_k: int = 4
    mixture_eps_merge: float = 1e-6

    # Physics coefficients
    gravity_strength: float = 0.05
    eos_gamma: float = 2.0
    thermal_pressure_coeff: float = 0.1
    repulsion_k: float = 50.0
    repulsion_n: float = 2.0
    shock_k: float = 0.2
    viscosity_global: float = 0.005

    # Radiation and absorption
    rad_to_heat_absorb_rate: float = 0.01

    # Regime thresholds for Z index
    Z_fuse_min: float = 1.5
    Z_deg_min: float = 3.0
    Z_bh_min: float = 4.5
    Z_abs_max: float = 6.0
    Z_star_flip: float = 2.5

    # Chemistry gating thresholds and tick ratio
    chemistry_tick_ratio: int = 5
    Z_chem_max: float = 1.0
    T_chem_max: float = 0.5

    # Black hole parameters
    EH_k: float = 0.5
    BH_absorb_energy_scale: float = 0.1

    # Stability coefficients for high‑energy stability function
    stability_low_coeff: float = 1.0
    stability_high_coeff: float = 1.0
    stability_temp_coeff: float = 0.5

    # Derived parameters reserved for future use (left empty to allow
    # extension via dataclass fields without rewriting defaults)
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a dict representation of the configuration.

        Useful for serialisation or interfacing with dynamic
        configuration loaders.
        """
        return self.__dict__.copy()