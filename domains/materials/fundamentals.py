"""
Fundamental material rules for the Stratum simulation.

This module defines the high‑energy material behaviours for the
prototype. It provides a ``MaterialsFundamentals`` class that
implements the equation of state (EOS), effective property mixing,
global diffusion/decay operations for heat and radiation, and a
rudimentary event handler for fusion, degeneracy and black hole
formation. The implementation here is intentionally simplified: it
defines only two fundamental species (a ``StellarGas`` species and a
``DEG`` degenerate matter species) and uses basic heuristics for
energy release/consumption. Fusion events create new species by
perturbing the parent's high energy properties; degenerate transitions
convert mass into the ``DEG`` species. Black holes are handled by the
``Quanta`` subsystem via the absorb_into_black_hole helper.

As the simulation evolves, new species may be created via fusion.
Their high energy properties are quantised and stored in the species
registry. Low energy properties are initialised to zero; these can be
migrated later via registry migrations. This design allows the
chemistry layer to extend species definitions without affecting the
physics layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    # When imported as part of the stratum package
    from ...core.config import EngineConfig
    from ...core.fabric import Fabric, Mixture
    from ...core.ledger import Ledger
    from ...core.types import Vec2
    from ...core.registry import SpeciesRegistry, Species
    from ...core.types import clamp
except ImportError:
    # When imported directly (e.g., during testing)
    from core.config import EngineConfig
    from core.fabric import Fabric, Mixture
    from core.ledger import Ledger
    from core.types import Vec2
    from core.registry import SpeciesRegistry, Species
    from core.types import clamp


@dataclass
class MaterialDefinition:
    """Simple container for fundamental material definitions.

    Each material has a name and a dictionary of high energy (HE)
    properties. Only properties defined in the registry's ``he_prop_defs``
    are used. Unknown properties are ignored. This class exists to
    clarify the default material definitions used by the prototype.
    """
    name: str
    he_props: Dict[str, float]


class MaterialsFundamentals:
    """High energy material logic for Stratum.

    The ``MaterialsFundamentals`` class encapsulates domain rules for
    high energy physics. It is responsible for:

      * Defining a small set of base materials (StellarGas and
        Degenerate matter) and registering them in the ``SpeciesRegistry``.
      * Computing effective mixture‑weighted properties such as
        ``rho_max`` and EOS stiffness.
      * Applying global operations such as heat and radiation diffusion
        and simple viscosity smoothing.
      * Handling high energy events (fusion, degenerate transition, BH
        formation) by computing activation energies and releasing or
        consuming energy accordingly.

    The design intentionally keeps the physics simple so that the
    prototype can be extended iteratively. More sophisticated EOS
    models, binding energy curves and barrier functions can be added
    later by extending this class.
    """

    def __init__(self, registry: SpeciesRegistry, config: EngineConfig):
        self.registry = registry
        self.cfg = config
        # Define fundamental materials: StellarGas and Degenerate matter
        # Values chosen heuristically within [0,1] range
        stellar_props = {
            "HE/rho_max": 0.3,
            "HE/chi": 0.5,
            "HE/eta": 0.1,
            "HE/opacity": 0.5,
            "HE/kappa_t": 0.05,
            "HE/kappa_r": 0.05,
            "HE/beta": 0.4,
            "HE/nu": 0.3,
            "HE/lambda": 0.1,
        }
        deg_props = {
            "HE/rho_max": 0.9,
            "HE/chi": 0.9,
            "HE/eta": 0.2,
            "HE/opacity": 0.1,
            "HE/kappa_t": 0.2,
            "HE/kappa_r": 0.2,
            "HE/beta": 0.8,
            "HE/nu": 0.05,
            "HE/lambda": 0.9,
        }
        # Register fundamental species if not already present
        self.stellar_species = self.registry.get_or_create_species(stellar_props, provenance={"source": "fundamental"})
        self.deg_species = self.registry.get_or_create_species(deg_props, provenance={"source": "fundamental"})

    # ------------------------------------------------------------------
    # Effective property computation

    def effective_property(self, mix: Mixture, registry: SpeciesRegistry, prop_name: str) -> float:
        """Return the mixture weighted average of a given high energy property.

        If no species are present, returns zero. Unknown species ids
        default to zero. This helper uses the registry's species table to
        map property names to values.
        """
        total = mix.total_mass()
        if total <= 0.0:
            return 0.0
        accum = 0.0
        for sid, mass in zip(mix.species_ids, mix.masses):
            sp = registry.species.get(sid)
            if not sp:
                continue
            accum += mass * sp.he_props.get(prop_name, 0.0)
        return accum / total

    # ------------------------------------------------------------------
    # Global operators: diffusion and smoothing

    def apply_global_ops(self, fabric: Fabric, cfg: EngineConfig) -> None:
        """Apply global diffusion/decay operations for heat and radiation.

        This simplified diffusion uses explicit finite difference with
        small diffusion coefficients derived from the mixture in each
        cell. Because this is called once per tick, coefficients should
        be small to maintain stability. In addition to diffusion, a
        small global viscosity can be applied to smooth momentum fields.
        """
        W, H = cfg.grid_w, cfg.grid_h
        # Precompute effective diffusivities per cell
        D_heat = np.zeros((W, H), dtype=np.float64)
        D_rad = np.zeros((W, H), dtype=np.float64)
        for i in range(W):
            for j in range(H):
                mix = fabric.mixtures[i][j]
                if mix.total_mass() > 0:
                    D_heat[i, j] = self.effective_property(mix, self.registry, "HE/kappa_t")
                    D_rad[i, j] = self.effective_property(mix, self.registry, "HE/kappa_r")
        # Diffuse heat
        E_heat_new = fabric.E_heat.copy()
        E_rad_new = fabric.E_rad.copy()
        for i in range(W):
            for j in range(H):
                # 4-neighbour Laplacian
                neighbors = [((i+1)%W, j), ((i-1+W)%W, j), (i, (j+1)%H), (i, (j-1+H)%H)]
                lap_h = 0.0
                lap_r = 0.0
                for (ni, nj) in neighbors:
                    lap_h += fabric.E_heat[ni, nj] - fabric.E_heat[i, j]
                    lap_r += fabric.E_rad[ni, nj] - fabric.E_rad[i, j]
                # update heat and radiation
                E_heat_new[i, j] += D_heat[i, j] * lap_h
                E_rad_new[i, j] += D_rad[i, j] * lap_r
        fabric.E_heat[:, :] = np.maximum(E_heat_new, 0.0)
        fabric.E_rad[:, :] = np.maximum(E_rad_new, 0.0)
        # optional: global viscosity smoothing on momentum
        # apply simple Laplacian smoothing with small coefficient
        eta = cfg.viscosity_global
        if eta > 0.0:
            px_new = fabric.px.copy()
            py_new = fabric.py.copy()
            for i in range(W):
                for j in range(H):
                    neighbors = [((i+1)%W, j), ((i-1+W)%W, j), (i, (j+1)%H), (i, (j-1+H)%H)]
                    lap_px = 0.0
                    lap_py = 0.0
                    for (ni, nj) in neighbors:
                        lap_px += fabric.px[ni, nj] - fabric.px[i, j]
                        lap_py += fabric.py[ni, nj] - fabric.py[i, j]
                    px_new[i, j] += eta * lap_px
                    py_new[i, j] += eta * lap_py
            fabric.px[:, :] = px_new
            fabric.py[:, :] = py_new

    # ------------------------------------------------------------------
    # Barrier and energy models for high energy events

    def E_avail_local(self, rho: float, heat: float, kin: float) -> float:
        """Compute available energy for crossing a barrier.

        The available energy is a weighted combination of thermal and
        kinetic energy.  Radiation energy is not considered directly
        available because it propagates away.  Tunable weights could be
        exposed through config if needed.
        """
        return 0.5 * heat + 0.5 * kin

    def E_act_fusion(self, he_props: Dict[str, float], Z: float, T: float, rho: float) -> float:
        """Activation energy for fusion.

        Higher values of ``rho_max`` and ``chi`` indicate stronger bonds
        that resist fusion, so the activation energy increases with
        these parameters.  Compression (high Z) lowers the activation
        energy, making fusion easier, while high temperature reduces the
        barrier via thermal tunnelling.
        """
        rho_max = he_props.get("HE/rho_max", 0.0)
        chi = he_props.get("HE/chi", 0.0)
        # Base barrier scales with binding strength
        base = 0.5 + 0.5 * (rho_max + chi)
        # Lower barrier if Z exceeds threshold
        z_factor = clamp((Z - self.cfg.Z_fuse_min) / max(self.cfg.Z_abs_max - self.cfg.Z_fuse_min, 1e-6), 0.0, 1.0)
        base *= max(0.1, 1.0 - z_factor)
        # Temperature reduces barrier: exp(-T)
        t_factor = np.exp(-T)
        return base * t_factor

    def fusion_yield_fraction(self, he_props: Dict[str, float], Z: float, T: float) -> float:
        """Return the fraction of the dominant species' mass that undergoes fusion.

        This fraction increases with the compression state Z and decreases
        with the stability of the species (lambda parameter).  It is
        clamped to the [0,1] range.
        """
        lam = he_props.get("HE/lambda", 0.0)
        # Basic yield from Z
        y = 0.2 + 0.4 * clamp((Z - self.cfg.Z_fuse_min) / max(self.cfg.Z_abs_max - self.cfg.Z_fuse_min, 1e-6), 0.0, 1.0)
        # Stability reduces yield
        y *= (1.0 - lam)
        return clamp(y, 0.0, 0.8)

    def radiation_fraction(self, parent_he: Dict[str, float], child_he: Dict[str, float], T: float) -> float:
        """Return fraction of released fusion energy radiated away.

        Low opacity species radiate more efficiently.  High temperature
        also encourages radiation relative to heat deposition.
        """
        op_p = parent_he.get("HE/opacity", 0.5)
        op_c = child_he.get("HE/opacity", 0.5)
        opacity_eff = 0.5 * (op_p + op_c)
        return clamp((1.0 - opacity_eff) * (T / (T + 1.0)), 0.0, 0.9)

    def E_act_decay(self, he_props: Dict[str, float], Z: float, T: float) -> float:
        """Activation energy for decay.

        Stable species (low lambda) have high decay barriers.  The
        barrier is lowered when the species is far from the preferred
        compression regime (below ``Z_fuse_min``).
        """
        lam = he_props.get("HE/lambda", 0.0)
        # Base barrier depends on stability
        base = 0.3 + 0.7 * lam
        # Lower barrier below fusion threshold
        z_factor = clamp((self.cfg.Z_fuse_min - Z) / max(self.cfg.Z_fuse_min, 1e-6), 0.0, 1.0)
        base *= max(0.05, 1.0 - z_factor)
        # Temperature increases decay probability (tunnelling)
        return base * np.exp(-T)

    def decay_fraction(self, he_props: Dict[str, float], Z: float, T: float) -> float:
        """Fraction of mass lost in a decay event.

        Unstable species (high lambda) decay more readily.  The
        fraction increases as Z moves away from the optimal compression
        window.
        """
        lam = he_props.get("HE/lambda", 0.0)
        f = 0.1 + 0.3 * lam
        # Increase with low compression (below fusion threshold)
        z_factor = clamp((self.cfg.Z_fuse_min - Z) / max(self.cfg.Z_fuse_min, 1e-6), 0.0, 1.0)
        f *= (1.0 + z_factor)
        return clamp(f, 0.0, 0.6)

    def decay_mass_split(self, parent_he: Dict[str, float], daughters: Tuple[Species], tick: int, cell: Tuple[int, int], attempt: int, entropy: Ledger) -> Tuple[float, ...]:
        """Return a tuple of fractions (summing to 1) representing how mass is
        distributed among daughter species in a decay.

        A deterministic split is derived from the species IDs hashed with
        the current tick and cell, with a small entropy component.
        """
        n = len(daughters)
        if n == 0:
            return ()
        # Use species IDs and tick to produce weights
        weights = []
        total = 0.0
        for idx, d in enumerate(daughters):
            # Deterministic weight from ID hash
            h = sum(ord(ch) for ch in d.id) + tick + idx
            w = (h % 100 + 1) / 100.0
            # Add small random jitter (deterministic if deterministic mode)
            u = entropy.entropy.sample_uniform("decay_split", tick, cell, attempt + idx, {"id": d.id})
            w += 0.1 * (u - 0.5)
            if w < 0.01:
                w = 0.01
            weights.append(w)
            total += w
        # Normalise
        return tuple((w / total) for w in weights)

    def compute_decay_energy(self, parent_he: Dict[str, float], daughters_he: Tuple[Dict[str, float], ...], Z: float) -> float:
        """Compute energy released or consumed in a decay event.

        Energy is the negative of the fusion energy required to go
        backwards: a decay to lighter species releases the difference
        in binding energy.  We sum contributions for each daughter.
        """
        if not daughters_he:
            return 0.0
        A_p = self._derive_A(parent_he)
        Bp = self._binding_curve(A_p)
        energy = 0.0
        for he in daughters_he:
            A_d = self._derive_A(he)
            Bd = self._binding_curve(A_d)
            # Each daughter releases binding difference (parent heavier)
            energy += Bp - Bd
        # Scale by Z modulation: less energy if highly compressed (consumed)
        m = 1.0 - 2.0 * clamp((Z - self.cfg.Z_star_flip) / max(self.cfg.Z_abs_max - self.cfg.Z_star_flip, 1e-6), 0.0, 1.0)
        return energy * m

    def E_act_degenerate(self, Z: float, T: float, mix: Mixture) -> float:
        """Activation energy for the degenerate transition.

        Lower when compression exceeds the degeneracy threshold.  Also
        reduced by high temperature via thermal tunnelling.
        """
        # Base barrier high; drop as Z increases
        z_factor = clamp((Z - self.cfg.Z_deg_min) / max(self.cfg.Z_abs_max - self.cfg.Z_deg_min, 1e-6), 0.0, 1.0)
        base = 1.0 - 0.8 * z_factor
        # Temperature effect
        return max(0.05, base * np.exp(-T))

    def degenerate_fraction(self, Z: float, T: float) -> float:
        """Fraction of mass converted to degenerate matter in a transition.

        Increases with compression; saturates at high Z.
        """
        return clamp(0.3 + 0.5 * clamp((Z - self.cfg.Z_deg_min) / max(self.cfg.Z_abs_max - self.cfg.Z_deg_min, 1e-6), 0.0, 1.0), 0.0, 0.8)

    def E_act_bh(self, Z: float, T: float) -> float:
        """Activation energy for black hole formation.

        As Z approaches the BH threshold, the barrier decreases.  High
        temperature also lowers the barrier slightly.
        """
        z_factor = clamp((Z - self.cfg.Z_bh_min) / max(self.cfg.Z_abs_max - self.cfg.Z_bh_min, 1e-6), 0.0, 1.0)
        base = 1.2 - z_factor
        return max(0.1, base * np.exp(-0.5 * T))

    def support_failure_metric(self, fabric: Fabric, i: int, j: int) -> float:
        """Return a crude support failure metric.

        This compares gravitational acceleration to pressure support.  We
        approximate local gravity as the gradient of influence and
        pressure support as gradient of pressure.  Higher ratios
        indicate likely collapse.
        """
        W, H = self.cfg.grid_w, self.cfg.grid_h
        # Finite difference gradient approximations using periodic wrap
        def grad(field, x, y):
            xm = (x - 1) % W
            xp = (x + 1) % W
            ym = (y - 1) % H
            yp = (y + 1) % H
            gx = 0.5 * (field[xp, y] - field[xm, y])
            gy = 0.5 * (field[x, yp] - field[x, ym])
            return np.hypot(gx, gy)
        # Approximate pressure using effective properties and cell state
        mix = fabric.mixtures[i][j]
        rho = fabric.rho[i, j]
        rho_max_eff = self.effective_property(mix, self.registry, "HE/rho_max")
        chi_eff = self.effective_property(mix, self.registry, "HE/chi")
        T = fabric.E_heat[i, j] / max(rho, 1e-8)
        r = rho / max(rho_max_eff, 1e-8)
        P = chi_eff * (r ** self.cfg.eos_gamma) + self.cfg.thermal_pressure_coeff * rho * T
        # Build synthetic pressure field around cell
        P_field = np.zeros((3, 3), dtype=float)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                x = (i + dx) % W
                y = (j + dy) % H
                mix_n = fabric.mixtures[x][y]
                rho_n = fabric.rho[x, y]
                rho_max_eff_n = self.effective_property(mix_n, self.registry, "HE/rho_max")
                chi_eff_n = self.effective_property(mix_n, self.registry, "HE/chi")
                T_n = fabric.E_heat[x, y] / max(rho_n, 1e-8)
                r_n = rho_n / max(rho_max_eff_n, 1e-8)
                P_field[dx+1, dy+1] = chi_eff_n * (r_n ** self.cfg.eos_gamma) + self.cfg.thermal_pressure_coeff * rho_n * T_n
        # Pressure gradient magnitude
        gradP = np.hypot(P_field[2,1] - P_field[0,1], P_field[1,2] - P_field[1,0]) * 0.5
        # Influence gradient magnitude
        gradI = grad(fabric.influence, i, j)
        if gradP < 1e-8:
            return float('inf')
        return gradI / gradP

    def potential_like_energy(self, fabric: Fabric, i: int, j: int) -> float:
        """Return a pseudo potential energy for BH formation.

        In this simplified model, we ignore gravitational potential and
        return zero.  A more elaborate model could integrate mass
        distribution.
        """
        return 0.0

    # Helper to derive mass index A and binding curve used in fusion/decay energy
    def _derive_A(self, he: Dict[str, float]) -> float:
        rho_max = he.get("HE/rho_max", 0.0)
        beta = he.get("HE/beta", 0.0)
        return max(0.0, 32.0 * rho_max * beta)

    def _binding_curve(self, A: float) -> float:
        b0, b1, b2 = 1.0, 1.5, 0.5
        return b0 * A - b1 * (A ** (2.0 / 3.0)) - b2 * (A ** (5.0 / 3.0))

    # ------------------------------------------------------------------
    # Event handling: degenerate, fusion, BH

    def handle_high_energy_events(
        self,
        fabric: Fabric,
        ledger: Ledger,
        registry: SpeciesRegistry,
        i: int,
        j: int,
        Z: float,
        T: float,
        tick: int,
        attempt: int,
    ) -> None:
        """Process high energy events at cell ``(i,j)``.

        This method implements degenerate transitions, fusion, decay and
        black hole formation. It uses the material's barrier functions
        and energy models to compute activation energies, available
        energy and gating probabilities. A single successful event per
        call short‑circuits further event checks. Energy is conserved
        by adjusting the cell's heat and radiation.
        """
        # no mass → no event
        rho = fabric.rho[i, j]
        if rho <= 1e-8:
            return
        mix = fabric.mixtures[i][j]
        if not mix.species_ids:
            return
        # dominant species (by mass)
        dom_sid = mix.species_ids[0]
        dom_sp = registry.species.get(dom_sid)
        if dom_sp is None:
            return
        dom_he = dom_sp.he_props
        # compute high energy stability (negative = unstable)
        S_high = self.compute_stability_high(dom_he, Z, T)
        # compute available energy from local heat and kinetic
        kin_energy = ledger.kinetic_energy(rho, Vec2(fabric.px[i, j], fabric.py[i, j]))
        avail_E = self.E_avail_local(rho, fabric.E_heat[i, j], kin_energy)
        # compute degenerate fraction
        total_mass = 0.0
        deg_mass = 0.0
        for sid, mass in zip(mix.species_ids, mix.masses):
            total_mass += mass
            if sid == self.deg_species.id:
                deg_mass += mass
        deg_frac = deg_mass / total_mass if total_mass > 0 else 0.0
        # ------------------------------------------------------------------
        # 1) Black hole formation: occurs at extreme Z with high degenerate fraction
        # Use the material's BH activation function and available energy to gate
        if Z >= self.cfg.Z_bh_min:
            E_act = self.E_act_bh(Z, T)
            # BH gating increases with degenerate fraction
            gate = clamp(deg_frac + 0.1, 0.0, 1.0)
            if ledger.barrier_crossed("bh", tick, (i, j), attempt, E_act, avail_E, T, gate):
                # absorb entire cell into BH mass
                self.absorb_into_black_hole(fabric, i, j, self.cfg)
                fabric.EH_mask[i, j] = 1.0
                return
        # ------------------------------------------------------------------
        # 2) Degenerate transition: compress into stiff state if Z large and unstable
        if Z >= self.cfg.Z_deg_min and S_high < 0.0:
            # Activation energy for degenerate transition
            E_act_deg = self.E_act_degenerate(Z, T, mix)
            # gate probability decreases if already mostly degenerate
            gate_deg = 1.0 - deg_frac
            gate_deg = clamp(gate_deg, 0.0, 1.0)
            if ledger.barrier_crossed("degenerate", tick, (i, j), attempt, E_act_deg, avail_E, T, gate_deg):
                # convert a fraction of dominant species to degenerate matter
                fraction = self.degenerate_fraction(Z, T)
                self.convert_mass_fraction(fabric, i, j, dom_sid, self.deg_species.id, fraction)
                # energy release from degeneracy
                heat_rel = self.energy_delta_degenerate(dom_he, Z, fraction, rho)
                fabric.E_heat[i, j] += heat_rel
                return
        # ------------------------------------------------------------------
        # 3) Fusion: form heavier species when compressed and unstable
        if Z >= self.cfg.Z_fuse_min and S_high < 0.0:
            # activation energy for fusion
            E_act_fus = self.E_act_fusion(dom_he, Z, T, rho)
            gate_fus = dom_he.get("HE/nu", 0.0)
            if ledger.barrier_crossed("fusion", tick, (i, j), attempt, E_act_fus, avail_E, T, gate_fus):
                # Determine child properties via drift + noise
                child_he = self.generate_fusion_child_he(dom_he, Z, tick, (i, j), attempt, ledger)
                child = registry.get_or_create_species(child_he, provenance={"parent": dom_sid, "tick": tick})
                # determine fraction of mass to fuse
                f_frac = self.fusion_yield_fraction(dom_he, Z, T)
                self.convert_mass_fraction(fabric, i, j, dom_sid, child.id, f_frac)
                # energy release or consumption from fusion
                dE = self.compute_fusion_energy(dom_he, child_he, Z, self.cfg.Z_star_flip)
                if dE != 0.0:
                    if dE > 0.0:
                        # allocate between heat and radiation
                        f_rad = self.radiation_fraction(dom_he, child_he, T)
                        fabric.E_heat[i, j] += dE * (1.0 - f_rad)
                        fabric.E_rad[i, j] += dE * f_rad
                    else:
                        # endothermic: deduct energy from heat and kinetic if possible
                        cost = -dE
                        # first remove from heat
                        take = min(cost, fabric.E_heat[i, j])
                        fabric.E_heat[i, j] -= take
                        cost -= take
                        if cost > 0:
                            # remove remaining cost by scaling momentum
                            # compute kinetic energy; reduce proportionally
                            ke = ledger.kinetic_energy(rho, Vec2(fabric.px[i, j], fabric.py[i, j]))
                            if ke > 0:
                                new_ke = max(0.0, ke - cost)
                                scale = np.sqrt(new_ke / ke) if ke > 0 else 0.0
                                fabric.px[i, j] *= scale
                                fabric.py[i, j] *= scale
                        # no negative energy beyond this point
                return
        # ------------------------------------------------------------------
        # 4) Decay: unstable species revert to lighter species at low compression
        # Only trigger decay when below fusion threshold and stability is very low
        if S_high < -0.1 and Z <= self.cfg.Z_fuse_min:
            # compute activation energy for decay
            E_act_dec = self.E_act_decay(dom_he, Z, T)
            gate_dec = dom_he.get("HE/lambda", 0.1)
            if ledger.barrier_crossed("decay", tick, (i, j), attempt, E_act_dec, avail_E, T, gate_dec):
                # compute fraction of mass to decay
                frac_dec = self.decay_fraction(dom_he, Z, T)
                # generate two daughters with drifted HE properties
                d_he1, d_he2 = self.generate_decay_daughters(dom_he, Z, tick, (i, j), attempt, ledger)
                # register daughters
                daughter1 = registry.get_or_create_species(d_he1, provenance={"parent": dom_sid, "tick": tick, "type": "decay"})
                daughter2 = registry.get_or_create_species(d_he2, provenance={"parent": dom_sid, "tick": tick, "type": "decay"})
                # split fraction between daughters equally (could be random) but we use equal share for simplicity
                frac_per = frac_dec * 0.5
                # convert mass to daughters
                self.convert_mass_fraction(fabric, i, j, dom_sid, daughter1.id, frac_per)
                self.convert_mass_fraction(fabric, i, j, dom_sid, daughter2.id, frac_per)
                # compute energy release from decay using binding differences
                dE = self.compute_decay_energy(dom_he, (d_he1, d_he2), Z)
                if dE > 0.0:
                    fabric.E_heat[i, j] += dE
                else:
                    # endothermic: deduct from heat
                    cost = -dE
                    take = min(cost, fabric.E_heat[i, j])
                    fabric.E_heat[i, j] -= take
                return

    # ------------------------------------------------------------------
    # Utility functions for mass conversion and energy

    def convert_mass_fraction(self, fabric: Fabric, i: int, j: int, from_sid: str, to_sid: str, fraction: float) -> None:
        """Convert a fraction of mass of species ``from_sid`` to species ``to_sid`` in cell (i,j).

        Mass and energies are preserved (no kinetic conversion). If the
        ``to_sid`` does not exist in the mixture, it is added. The
        mixture is pruned to the top K species after conversion.
        """
        if fraction <= 0.0 or fraction >= 1.0:
            return
        mix = fabric.mixtures[i][j]
        # locate from species
        if from_sid not in mix.species_ids:
            return
        idx = mix.species_ids.index(from_sid)
        m_from = mix.masses[idx]
        amount = m_from * fraction
        # subtract from from species
        mix.masses[idx] -= amount
        # add to to species
        mix.add_species_mass(to_sid, amount, self.cfg.mixture_top_k)
        # clean up mixture
        mix.cleanup(self.cfg.mixture_eps_merge, self.cfg.mixture_top_k)

    def generate_fusion_child_he(self, parent_he: Dict[str, float], Z: float, tick: int, cell: Tuple[int, int], attempt: int, ledger: Ledger) -> Dict[str, float]:
        """Generate new high energy properties for a child species.

        The child properties are created by copying the parent's values
        and adding a small deterministic drift and a tiny stochastic
        perturbation. The drift magnitude increases with ``Z``. The
        perturbation is derived from the Ledger's entropy source.
        """
        child = dict(parent_he)
        # drift amount proportional to how far above fuse threshold we are
        sigma = max(0.0, (Z - self.cfg.Z_fuse_min) / max(self.cfg.Z_abs_max - self.cfg.Z_fuse_min, 1e-6))
        # adjust rho_max and chi upwards to represent heavier/denser species
        child["HE/rho_max"] = min(1.0, child.get("HE/rho_max", 0.0) + 0.1 * sigma)
        child["HE/chi"] = min(1.0, child.get("HE/chi", 0.0) + 0.05 * sigma)
        # adjust opacity down slightly to let more radiation escape
        child["HE/opacity"] = max(0.0, child.get("HE/opacity", 0.0) - 0.05 * sigma)
        # adjust beta and lambda
        child["HE/beta"] = min(1.0, child.get("HE/beta", 0.0) + 0.05 * sigma)
        child["HE/lambda"] = max(0.0, child.get("HE/lambda", 0.0) - 0.05 * sigma)
        # small noise on selected props
        u = ledger.entropy.sample_uniform("fusion_child", tick, cell, attempt, {"sigma": sigma}) - 0.5
        child["HE/nu"] = min(1.0, max(0.0, child.get("HE/nu", 0.0) + u * 0.02))
        return child

    # ------------------------------------------------------------------
    # Decay daughter generation

    def generate_decay_daughters(self, parent_he: Dict[str, float], Z: float, tick: int, cell: Tuple[int, int], attempt: int, ledger: Ledger) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Generate two child HE property dictionaries for a decay event.

        The daughters are produced by scaling the parent's packing limit and
        stiffness downward to represent a lighter species. We also adjust
        opacity and binding depth to reflect lower density and less
        tightly bound matter. A small deterministic drift is applied based
        on the local extremity ``Z`` and the parent's properties. A tiny
        random perturbation derived from the ledger's entropy source adds
        diversity but remains reproducible in deterministic mode.

        Parameters
        ----------
        parent_he : Dict[str, float]
            The high energy properties of the parent species undergoing decay.
        Z : float
            Current extremity index (compression state) of the cell.
        tick : int
            Simulation tick, used for entropy seeding.
        cell : Tuple[int, int]
            Grid cell coordinates, used for entropy seeding.
        attempt : int
            Microtick attempt index, used for entropy seeding.
        ledger : Ledger
            The ledger providing access to deterministic entropy.

        Returns
        -------
        Tuple[Dict[str, float], Dict[str, float]]
            Two dictionaries containing the new HE properties for the daughter species.
        """
        # Determine scaling based on Z; lower Z means more pronounced reduction.
        sigma = 1.0 - clamp((Z - self.cfg.Z_fuse_min) / max(self.cfg.Z_abs_max - self.cfg.Z_fuse_min, 1e-6), 0.0, 1.0)
        # Ensure sigma between 0.0 and 1.0
        sigma = clamp(sigma, 0.0, 1.0)
        daughters = []
        # We'll produce two daughters with slight variations
        for idx in range(2):
            child = dict(parent_he)
            # reduce rho_max and chi to lighten the species
            child["HE/rho_max"] = max(0.05, child.get("HE/rho_max", 0.0) * (0.7 + 0.2 * sigma))
            child["HE/chi"] = max(0.05, child.get("HE/chi", 0.0) * (0.7 + 0.2 * sigma))
            # increase opacity slightly (lighter species trap less radiation)
            child["HE/opacity"] = clamp(child.get("HE/opacity", 0.0) + 0.05 * sigma, 0.0, 1.0)
            # decrease beta and lambda modestly
            child["HE/beta"] = max(0.0, child.get("HE/beta", 0.0) - 0.1 * sigma)
            child["HE/lambda"] = max(0.0, child.get("HE/lambda", 0.0) - 0.05 * sigma)
            # fusion affinity lowered for daughter (harder to fuse) and increased decay tendency slightly
            child["HE/nu"] = max(0.0, child.get("HE/nu", 0.0) - 0.05)
            child["HE/lambda"] = clamp(child["HE/lambda"] + 0.05, 0.0, 1.0)
            # deterministic drift based on index
            drift = 0.01 * (idx * 2 - 1)  # ±0.01
            child["HE/rho_max"] = clamp(child["HE/rho_max"] + drift, 0.0, 1.0)
            # small entropy perturbation on nu
            u = ledger.entropy.sample_uniform("decay_daughter", tick, cell, attempt + idx, {"sigma": sigma}) - 0.5
            child["HE/nu"] = clamp(child["HE/nu"] + u * 0.02, 0.0, 1.0)
            daughters.append(child)
        return daughters[0], daughters[1]

    # ------------------------------------------------------------------
    # Degenerate transition energy model

    def energy_delta_degenerate(self, parent_he: Dict[str, float], Z: float, fraction: float, rho: float) -> float:
        """Compute energy released when a fraction of mass transitions to degenerate matter.

        When matter is compressed beyond the degeneracy threshold, it
        enters a stiff state.  The conversion releases energy
        depending on how stiff the new state is relative to the
        parent.  A simple model uses the difference in EOS stiffness
        (``chi``) and binding depth (``beta``) between the parent and
        the degenerate species.  The energy released is scaled by the
        mass converted (``fraction * rho``).

        Parameters
        ----------
        parent_he : Dict[str, float]
            High energy properties of the species being converted.
        Z : float
            Current extremity index.
        fraction : float
            Fraction of mass being converted to degenerate matter.
        rho : float
            Total mass/occupancy in the cell.

        Returns
        -------
        float
            The amount of heat energy released.
        """
        # Target degenerate species properties
        deg_he = self.deg_species.he_props
        chi_p = parent_he.get("HE/chi", 0.0)
        beta_p = parent_he.get("HE/beta", 0.0)
        chi_deg = deg_he.get("HE/chi", 0.0)
        beta_deg = deg_he.get("HE/beta", 0.0)
        # Energy per unit mass proportional to stiffness and binding differences
        delta_chi = max(0.0, chi_deg - chi_p)
        delta_beta = max(0.0, beta_deg - beta_p)
        # The energy release increases with Z beyond the deg threshold
        z_factor = clamp((Z - self.cfg.Z_deg_min) / max(self.cfg.Z_abs_max - self.cfg.Z_deg_min, 1e-6), 0.0, 1.0)
        per_mass_energy = (delta_chi + delta_beta) * 0.5 * (1.0 + z_factor)
        return per_mass_energy * fraction * rho

    def compute_fusion_energy(self, parent_he: Dict[str, float], child_he: Dict[str, float], Z: float, Z_star_flip: float) -> float:
        """Return energy released (positive) or consumed (negative) for a fusion event.

        This implementation approximates binding energy differences using
        a crude ``mass‑number'' index derived from the species' high
        energy properties. Rather than relying solely on the beta
        parameter, we compute an effective mass index ``A`` from the
        packing and binding properties (``rho_max`` and ``beta``). The
        binding curve is approximated by a simple function with a
        maximum at an intermediate ``A`` value, loosely inspired by
        nuclear binding energy curves: ``B(A) = b0*A - b1*A^(2/3) -
        b2*A^(5/3)``.  Fusion energy is then proportional to the
        difference ``B(A_child) - B(A_parent)`` and modulated by a Z
        dependent factor that flips sign beyond ``Z_star_flip``.

        Parameters
        ----------
        parent_he : Dict[str, float]
            High energy properties of the parent species.
        child_he : Dict[str, float]
            High energy properties of the fused species.
        Z : float
            Local extremity index.
        Z_star_flip : float
            Threshold at which the sign of energy release flips due to
            extreme compression.

        Returns
        -------
        float
            Energy per unit mass released (>0) or absorbed (<0).
        """
        # Derive effective mass index A for parent and child from rho_max and beta.
        def derive_A(he: Dict[str, float]) -> float:
            # The packing limit and binding depth together hint at how
            # "heavy" a species is. Scale into a moderate range to avoid
            # numeric blow‑up.  The constants here are heuristic.
            rho_max = he.get("HE/rho_max", 0.0)
            beta = he.get("HE/beta", 0.0)
            return max(0.0, 32.0 * rho_max * beta)

        A_p = derive_A(parent_he)
        A_c = derive_A(child_he)
        # Binding energy curve parameters (heuristic).  b0 sets linear
        # growth, b1 and b2 provide decreasing returns for larger A.
        b0, b1, b2 = 1.0, 1.5, 0.5
        def binding_curve(A: float) -> float:
            return b0 * A - b1 * (A ** (2.0 / 3.0)) - b2 * (A ** (5.0 / 3.0))

        Bp = binding_curve(A_p)
        Bc = binding_curve(A_c)
        delta_B = Bc - Bp
        # Z modulation: smoothly flip the sign beyond Z_star_flip.  When
        # Z is below the flip threshold, m≈1; when far above, m≈-1.
        if self.cfg.Z_abs_max > Z_star_flip:
            m = 1.0 - 2.0 * clamp((Z - Z_star_flip) / max(self.cfg.Z_abs_max - Z_star_flip, 1e-6), 0.0, 1.0)
        else:
            m = 1.0
        return delta_B * m


    def compute_stability_high(self, he_props: Dict[str, float], Z: float, T: float) -> float:
        """Compute a stability score for high energy regime.

        Positive values mean stable; negative values mean unstable. This
        implementation uses only a few HE properties and the Z and T
        values. More complex models could be implemented by extending
        this function.
        """
        # base stability from beta and chi (stronger binding and stiffer EOS give stability)
        S0 = he_props.get("HE/beta", 0.0) + he_props.get("HE/chi", 0.0) - he_props.get("HE/lambda", 0.0)
        # penalise high Z and high T
        S = S0 - self.cfg.stability_high_coeff * max(0.0, Z - self.cfg.Z_fuse_min)
        S -= self.cfg.stability_temp_coeff * max(0.0, T - 0.5)
        return S

    # ------------------------------------------------------------------
    # Black hole absorption helper

    def absorb_into_black_hole(self, fabric: Fabric, i: int, j: int, cfg: EngineConfig) -> None:
        """Absorb the contents of a cell into its black hole mass.

        When a black hole is present in a cell, any residual mass,
        momentum and energy are converted into BH_mass. The event horizon
        mask is updated elsewhere in the Quanta subsystem. This helper
        is invoked during microticks when a BH is detected in the cell.
        """
        # add mass and energy to BH_mass
        m = fabric.rho[i, j]
        # compute kinetic energy locally: avoid Ledger dependency in this helper
        pvec_x = float(fabric.px[i, j])
        pvec_y = float(fabric.py[i, j])
        Ekin = 0.0
        if m > 1e-12:
            # Use multiplication to avoid overflow and clamp the result.
            sum_sq = pvec_x * pvec_x + pvec_y * pvec_y
            if not math.isfinite(sum_sq) or sum_sq > 1e300:
                sum_sq = 1e300
            Ekin = sum_sq / (2.0 * m)
        E = fabric.E_heat[i, j] + fabric.E_rad[i, j] + Ekin
        fabric.BH_mass[i, j] += m + cfg.BH_absorb_energy_scale * E
        # clear cell
        fabric.rho[i, j] = 0.0
        fabric.px[i, j] = 0.0
        fabric.py[i, j] = 0.0
        fabric.E_heat[i, j] = 0.0
        fabric.E_rad[i, j] = 0.0
        fabric.mixtures[i][j] = Mixture([], [])
