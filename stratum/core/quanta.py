"""
Quanta subsystem: event propagation and local microtick resolution.

Quanta orchestrates the delivery of signals (influence, radiation,
impulses) with finite speed and processes local cell interactions in
small microticks before propagating their finalised effects outwards.
This implementation follows the design described in the specification,
but simplifies some aspects for clarity and performance: influence
propagation uses a simple signal queue, radiation energy is injected
locally for each tick and then diffuses, and active cell selection
uses basic heuristics based on gradients and overfill.

Key Performance Features:
- Derived fields (rho_max_eff, T_field, etc.) are cached per tick
- Boundary handling is centralized through Fabric methods
- Active cell selection uses efficient numpy operations
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import EngineConfig
from .types import Vec2, clamp
from .ledger import Ledger
from .fabric import Fabric, Mixture
from .registry import SpeciesRegistry
from . import types as ttypes

# Numerical stability constants
#: Maximum energy delta to prevent downstream overflow in energy conversions
ENERGY_DELTA_MAX = 1e50


class SignalType:
    """Signal type constants for the propagation system."""
    INFLUENCE = 0
    RADIATION = 1
    IMPULSE = 2
    DISTURBANCE = 3


class Signal:
    """Lightweight signal representation used for propagation.
    
    Attributes:
        type: Signal type (INFLUENCE, RADIATION, IMPULSE, DISTURBANCE).
        emit_tick: Tick when the signal was emitted.
        origin: Grid coordinates (i, j) of the signal origin.
        speed: Propagation speed in cells per tick.
        attenuation: Attenuation factor per unit distance.
        radius: Radius of effect in cells.
        payload: Dictionary of signal-specific data.
        arrive_tick: Computed tick when signal arrives.
    """

    def __init__(self, sig_type: int, emit_tick: int, origin: Tuple[int, int], speed: float, attenuation: float, radius: int, payload: dict):
        self.type = sig_type
        self.emit_tick = emit_tick
        self.origin = origin
        self.speed = speed
        self.attenuation = attenuation
        self.radius = radius
        self.payload = payload
        self.arrive_tick = emit_tick  # will be set when enqueued


class SignalQueue:
    """Time-sorted queue for signal delivery."""

    def __init__(self):
        self.by_tick: Dict[int, List[Signal]] = {}

    def push(self, sig: Signal, current_tick: int, v_max: float) -> None:
        """Insert a signal into the queue computing its arrival time."""
        # compute travel time: distance / speed; in this simplified version
        # we assume signals deposit energy in the origin cell only on the next tick
        # For a real implementation distance to footprint radius would be used.
        # Here we treat radius = 0 and schedule for next tick.
        sig.arrive_tick = current_tick + 1
        self.by_tick.setdefault(sig.arrive_tick, []).append(sig)

    def pop_arrivals(self, tick: int) -> List[Signal]:
        arrivals = self.by_tick.pop(tick, [])
        return arrivals


@dataclass
class DerivedFields:
    """Cached derived fields computed once per tick.
    
    This cache eliminates redundant computation of mixture-weighted properties
    and derived quantities during microtick processing. All fields are computed
    once at the beginning of each tick.
    
    Attributes:
        rho_max_eff: Effective maximum density per cell (mixture-weighted).
        chi_eff: Effective EOS stiffness per cell.
        eta_eff: Effective viscosity per cell.
        opacity_eff: Effective opacity per cell.
        T_field: Temperature field (E_heat / rho).
        Z_field: Compression index field.
        grad_infl_mag: Magnitude of influence gradient.
        tick: Tick number for which this cache is valid.
    """
    rho_max_eff: np.ndarray
    chi_eff: np.ndarray
    eta_eff: np.ndarray
    opacity_eff: np.ndarray
    T_field: np.ndarray
    Z_field: np.ndarray
    grad_infl_mag: np.ndarray
    tick: int


class Quanta:
    """Event propagation and microtick scheduler.
    
    The Quanta subsystem handles:
    - Signal delivery: Influence, radiation, and impulse signals
    - Active region selection: Identifies cells requiring detailed simulation
    - Microtick resolution: Local physics updates within each tick
    - Mass transfer: Conservation-aware mass movement between cells
    
    Key performance optimization: Derived fields are computed once per tick
    and cached in a DerivedFields dataclass to avoid redundant computation.
    """

    def __init__(self, fabric: Fabric, ledger: Ledger, registry: SpeciesRegistry, materials: 'MaterialsFundamentals', config: EngineConfig):
        self.fabric = fabric
        self.ledger = ledger
        self.registry = registry
        self.materials = materials
        self.cfg = config
        self.queue = SignalQueue()
        # Simple counters for diagnostics
        self.total_microticks = 0
        # Derived fields cache (computed once per tick)
        self._derived_cache: Optional[DerivedFields] = None

    def _compute_derived_fields(self, tick: int) -> DerivedFields:
        """Compute and cache all derived fields for the current tick.
        
        This method computes mixture-weighted properties and derived quantities
        once per tick, avoiding redundant computation during microtick processing.
        
        Args:
            tick: Current simulation tick.
            
        Returns:
            DerivedFields dataclass containing all cached fields.
        """
        W, H = self.cfg.grid_w, self.cfg.grid_h
        
        # Compute effective properties per cell (mixture-weighted)
        rho_max_eff = np.zeros((W, H), dtype=np.float64)
        chi_eff = np.zeros((W, H), dtype=np.float64)
        eta_eff = np.zeros((W, H), dtype=np.float64)
        opacity_eff = np.zeros((W, H), dtype=np.float64)
        
        # Build property lookup tables once
        rho_max_table = {sid: s.he_props.get("HE/rho_max", 0.0) for sid, s in self.registry.species.items()}
        chi_table = {sid: s.he_props.get("HE/chi", 0.0) for sid, s in self.registry.species.items()}
        eta_table = {sid: s.he_props.get("HE/eta", 0.0) for sid, s in self.registry.species.items()}
        opacity_table = {sid: s.he_props.get("HE/opacity", 0.0) for sid, s in self.registry.species.items()}
        
        # Compute weighted properties for each cell
        for i in range(W):
            for j in range(H):
                mix = self.fabric.mixtures[i][j]
                total = mix.total_mass()
                if total <= 0.0:
                    continue
                    
                rho_max_accum = 0.0
                chi_accum = 0.0
                eta_accum = 0.0
                opacity_accum = 0.0
                
                for sid, mass in zip(mix.species_ids, mix.masses):
                    rho_max_accum += mass * rho_max_table.get(sid, 0.0)
                    chi_accum += mass * chi_table.get(sid, 0.0)
                    eta_accum += mass * eta_table.get(sid, 0.0)
                    opacity_accum += mass * opacity_table.get(sid, 0.0)
                
                rho_max_eff[i, j] = rho_max_accum / total
                chi_eff[i, j] = chi_accum / total
                eta_eff[i, j] = eta_accum / total + self.cfg.viscosity_global
                opacity_eff[i, j] = opacity_accum / total
        
        # Add global viscosity to eta
        eta_eff += self.cfg.viscosity_global
        
        # Compute temperature field
        T_field = np.divide(self.fabric.E_heat, np.maximum(self.fabric.rho, 1e-12))
        
        # Compute influence gradient
        grad_infl_x, grad_infl_y = self.fabric.gradient_scalar(self.fabric.influence)
        grad_infl_mag = np.sqrt(grad_infl_x ** 2 + grad_infl_y ** 2)
        
        # Compute overfill and Z field
        overfill = np.maximum(self.fabric.rho - rho_max_eff, 0.0)
        Z_field = (
            1.0 * np.divide(self.fabric.rho, np.maximum(rho_max_eff, 1e-12))
            + 0.5 * np.log1p(T_field)
            + 0.2 * grad_infl_mag
            + 0.3 * overfill
        )
        
        return DerivedFields(
            rho_max_eff=rho_max_eff,
            chi_eff=chi_eff,
            eta_eff=eta_eff,
            opacity_eff=opacity_eff,
            T_field=T_field,
            Z_field=Z_field,
            grad_infl_mag=grad_infl_mag,
            tick=tick
        )

    def step(self, tick: int, micro_budget: int) -> None:
        """Execute one simulation tick.
        
        This method:
        1. Delivers scheduled signals
        2. Computes and caches derived fields
        3. Selects active cells for detailed processing
        4. Runs microtick resolution on active cells
        5. Applies global operators (diffusion, etc.)
        
        Args:
            tick: Current simulation tick number.
            micro_budget: Total microtick budget for this tick.
        """
        # deliver signals for this tick
        arrivals = self.queue.pop_arrivals(tick)
        W, H = self.cfg.grid_w, self.cfg.grid_h
        # temporary accumulators
        signal_hits = np.zeros((W, H), dtype=np.int32)
        influence_add = np.zeros((W, H), dtype=np.float64)
        rad_add = np.zeros((W, H), dtype=np.float64)
        impulse_x = np.zeros((W, H), dtype=np.float64)
        impulse_y = np.zeros((W, H), dtype=np.float64)
        # accumulate
        for sig in arrivals:
            x0, y0 = sig.origin
            # For now, signals deposit in the origin cell only (simplified propagation)
            if 0 <= x0 < W and 0 <= y0 < H:
                signal_hits[x0, y0] += 1
                if sig.type == SignalType.INFLUENCE:
                    influence_add[x0, y0] += sig.payload.get("strength", 0.0)
                elif sig.type == SignalType.RADIATION:
                    rad_add[x0, y0] += sig.payload.get("energy", 0.0)
                elif sig.type == SignalType.IMPULSE:
                    dp: Vec2 = sig.payload.get("dp", Vec2(0.0, 0.0))
                    impulse_x[x0, y0] += dp.x
                    impulse_y[x0, y0] += dp.y
        # apply accumulators
        self.fabric.influence += influence_add
        self.fabric.E_rad += rad_add
        self.fabric.px += impulse_x
        self.fabric.py += impulse_y
        
        # Compute and cache derived fields for this tick
        self._derived_cache = self._compute_derived_fields(tick)
        derived = self._derived_cache
        
        # compute gradient of rho for active cell selection
        grad_rho_x, grad_rho_y = self.fabric.gradient_scalar(self.fabric.rho)
        grad_rho_mag = np.sqrt(grad_rho_x ** 2 + grad_rho_y ** 2)
        
        # Compute overfill using cached rho_max_eff
        overfill = np.maximum(self.fabric.rho - derived.rho_max_eff, 0.0)
        
        # determine active cells: top cells with high grad or overfill or high Z
        # flatten arrays to rank
        scores = grad_rho_mag + overfill + np.maximum(derived.Z_field - self.cfg.Z_fuse_min * 0.5, 0.0)
        
        # Use argpartition for O(N) top-K selection instead of O(N log N) argsort
        # When all scores are equal (e.g. uniform initial conditions), this ordering
        # is arbitrary but stable. We pick the top ``active_region_max`` cells
        # regardless of the absolute score so that microticks always run on some
        # subset of the grid even in symmetric states.
        flat_scores = scores.ravel()
        n_total = len(flat_scores)
        n_active = min(self.cfg.active_region_max, n_total)
        
        if n_active > 0 and n_total > n_active:
            # Use argpartition for O(N) selection of top K elements
            # argpartition gives us indices where the K-th element is in its final position
            # and all elements before it are smaller (unordered) and after are larger (unordered)
            # We want the LARGEST K elements, so we partition at (n_total - n_active)
            partition_idx = n_total - n_active
            partitioned_indices = np.argpartition(flat_scores, partition_idx)
            top_k_indices = partitioned_indices[partition_idx:]
            
            # Sort only the top K elements for consistent ordering (optional but good for reproducibility)
            top_k_scores = flat_scores[top_k_indices]
            sorted_within_topk = np.argsort(top_k_scores)[::-1]
            flat_indices = top_k_indices[sorted_within_topk]
        else:
            # If we want all or more cells than exist, just take all (sorted)
            flat_indices = np.argsort(flat_scores)[::-1][:n_active]
        
        active_cells: List[Tuple[int, int]] = []
        for idx in flat_indices:
            i = idx // H
            j = idx % H
            active_cells.append((i, j))
        # allocate microticks evenly for simplicity
        if len(active_cells) == 0:
            return
        micro_per_cell = max(1, micro_budget // len(active_cells))
        # main local microtick loop
        for (i, j) in active_cells:
            # skip BH cells
            if self.fabric.EH_mask[i, j] > 0.0:
                continue
            # microticks for this cell - pass cached derived fields
            M = min(micro_per_cell, self.cfg.microtick_cap_per_region)
            self.resolve_cell(i, j, M, derived, tick)
            self.total_microticks += M
        # after microticks, diffuse heat and radiation (global operators)
        self.materials.apply_global_ops(self.fabric, self.cfg)
        # reset influence for next tick
        self.fabric.reset_influence()

    def compute_effective_rho_max(self) -> np.ndarray:
        """Compute the mixture weighted effective rho_max per cell.

        Returns a numpy array of shape (W,H). Use HE/rho_max table from
        registry. If a species is unknown in the registry, zero is
        assumed, which will result in zero effective rho_max; caller
        should treat such cells carefully.
        """
        W, H = self.cfg.grid_w, self.cfg.grid_h
        rho_max_eff = np.zeros((W, H), dtype=np.float64)
        # build a property lookup table for rho_max
        rho_max_table = {sid: s.he_props.get("HE/rho_max", 0.0) for sid, s in self.registry.species.items()}
        # compute per cell
        for i in range(W):
            for j in range(H):
                mix = self.fabric.mixtures[i][j]
                total = mix.total_mass()
                if total <= 0.0:
                    rho_max_eff[i, j] = 0.0
                    continue
                accum = 0.0
                for sid, mass in zip(mix.species_ids, mix.masses):
                    accum += mass * rho_max_table.get(sid, 0.0)
                rho_max_eff[i, j] = accum / total
        return rho_max_eff

    def resolve_cell(self, i: int, j: int, M: int, derived: DerivedFields, tick: int) -> None:
        """Perform ``M`` local microticks at cell (i,j).

        This method implements a simplified version of the detailed logic
        from the specification. It computes pressure and gravitational
        accelerations, updates momentum, performs a small advection
        substep, corrects overfill and handles high‑energy events via
        Material helpers.
        
        Uses cached derived fields from DerivedFields to avoid redundant
        computation of mixture-weighted properties.
        
        Args:
            i: X coordinate of the cell.
            j: Y coordinate of the cell.
            M: Number of microticks to perform.
            derived: Cached derived fields for this tick.
            tick: Current simulation tick.
        """
        # local aliases for speed
        rho_array = self.fabric.rho
        px = self.fabric.px
        py = self.fabric.py
        E_heat = self.fabric.E_heat
        E_rad = self.fabric.E_rad
        influence = self.fabric.influence
        mix = self.fabric.mixtures[i][j]
        
        # Get cached effective properties (computed once per tick)
        rho_max_eff = derived.rho_max_eff[i, j]
        chi_eff = derived.chi_eff[i, j]
        eta_eff = derived.eta_eff[i, j]
        opacity_eff = derived.opacity_eff[i, j]
        
        # skip empty cells
        if rho_array[i, j] <= 1e-12:
            return
        for m in range(M):
            if self.fabric.EH_mask[i, j] > 0.0:
                # if a BH formed, absorb contents
                self.materials.absorb_into_black_hole(self.fabric, i, j, self.cfg)
                return
            rho = rho_array[i, j]
            if rho <= 1e-12:
                break
            # temperature
            T = E_heat[i, j] / max(rho, 1e-12)
            # pressure
            r_ratio = rho / max(rho_max_eff, 1e-12)
            P_eos = chi_eff * (r_ratio ** self.cfg.eos_gamma)
            P_th = self.cfg.thermal_pressure_coeff * rho * T
            over = max(0.0, rho - rho_max_eff)
            P_rep = self.cfg.repulsion_k * ((over / max(rho_max_eff, 1e-12)) ** self.cfg.repulsion_n) if rho_max_eff > 0 else 0.0
            P_tot = P_eos + P_th + P_rep
            # gradients of pressure and influence
            # approximate local gradient using finite differences on small stencil
            # Use boundary-aware neighbor indexing
            W, H = self.cfg.grid_w, self.cfg.grid_h
            # Get boundary-aware neighbor indices
            ip1, _, ip1_valid = self.fabric.get_neighbor_index(i, j, 1, 0)
            im1, _, im1_valid = self.fabric.get_neighbor_index(i, j, -1, 0)
            _, jp1, jp1_valid = self.fabric.get_neighbor_index(i, j, 0, 1)
            _, jm1, jm1_valid = self.fabric.get_neighbor_index(i, j, 0, -1)
            
            # partial derivatives of P (use current cell value if neighbor invalid)
            if im1_valid:
                P_im1 = chi_eff * ((rho_array[im1, j] / max(rho_max_eff, 1e-12)) ** self.cfg.eos_gamma) + self.cfg.thermal_pressure_coeff * rho_array[im1, j] * (E_heat[im1, j] / max(rho_array[im1, j], 1e-12))
            else:
                P_im1 = P_tot  # Zero gradient at boundary
            if jm1_valid:
                P_jm1 = chi_eff * ((rho_array[i, jm1] / max(rho_max_eff, 1e-12)) ** self.cfg.eos_gamma) + self.cfg.thermal_pressure_coeff * rho_array[i, jm1] * (E_heat[i, jm1] / max(rho_array[i, jm1], 1e-12))
            else:
                P_jm1 = P_tot  # Zero gradient at boundary
                
            dPdx = 0.5 * (P_tot - P_im1)
            dPdy = 0.5 * (P_tot - P_jm1)
            
            # influence gradient approximated from precomputed magnitude field
            infl_ip1 = influence[ip1, j] if ip1_valid else influence[i, j]
            infl_im1 = influence[im1, j] if im1_valid else influence[i, j]
            infl_jp1 = influence[i, jp1] if jp1_valid else influence[i, j]
            infl_jm1 = influence[i, jm1] if jm1_valid else influence[i, j]
            
            dIdx = 0.5 * (infl_ip1 - infl_im1)
            dIdy = 0.5 * (infl_jp1 - infl_jm1)
            # accelerations
            a_x = -(dPdx) / max(rho, 1e-12) - self.cfg.gravity_strength * dIdx
            a_y = -(dPdy) / max(rho, 1e-12) - self.cfg.gravity_strength * dIdy
            # update momentum with viscous damping
            dt_sub = 1.0 / float(M)
            px[i, j] += rho * a_x * dt_sub
            py[i, j] += rho * a_y * dt_sub
            px[i, j] *= max(0.0, 1.0 - eta_eff * dt_sub)
            py[i, j] *= max(0.0, 1.0 - eta_eff * dt_sub)
            # local advection: move mass to neighbours based on velocity
            v_x = px[i, j] / max(rho, 1e-12)
            v_y = py[i, j] / max(rho, 1e-12)
            # compute outflows to four neighbours
            fx_pos = max(0.0, v_x) * dt_sub
            fx_neg = max(0.0, -v_x) * dt_sub
            fy_pos = max(0.0, v_y) * dt_sub
            fy_neg = max(0.0, -v_y) * dt_sub
            total_out = fx_pos + fx_neg + fy_pos + fy_neg
            if total_out > 0.5:
                # clamp to avoid numerical explosion
                s = 0.5 / total_out
                fx_pos *= s
                fx_neg *= s
                fy_pos *= s
                fy_neg *= s
            # move mass to neighbours using boundary-aware indexing
            # right neighbour
            dm = rho * fx_pos
            if dm > 0 and ip1_valid:
                self.transfer_mass(i, j, ip1, j, dm, v_x, v_y)
            # left neighbour
            dm = rho * fx_neg
            if dm > 0 and im1_valid:
                self.transfer_mass(i, j, im1, j, dm, v_x, v_y)
            # up neighbour
            dm = rho * fy_pos
            if dm > 0 and jp1_valid:
                self.transfer_mass(i, j, i, jp1, dm, v_x, v_y)
            # down neighbour
            dm = rho * fy_neg
            if dm > 0 and jm1_valid:
                self.transfer_mass(i, j, i, jm1, dm, v_x, v_y)
            # update local cell after transfers: mass, energies, mixture
            rho = rho_array[i, j]  # updated by transfer
            if rho <= 1e-12:
                break
            # overfill correction: push outwards and heat
            over = max(0.0, rho - rho_max_eff)
            if over > 0:
                # push small fraction outward; convert kinetic into heat
                # choose arbitrary direction for repulsion (here just x)
                dm_rep = min(over, rho * 0.1)
                # move to right cell if valid
                if ip1_valid:
                    self.transfer_mass(i, j, ip1, j, dm_rep, v_x, v_y)
                # convert kinetic to heat
                pvec_before = Vec2(px[i, j], py[i, j])
                Ekin_before = self.ledger.kinetic_energy(rho, pvec_before)
                # reduce momentum magnitude
                px[i, j] *= 0.9
                py[i, j] *= 0.9
                pvec_after = Vec2(px[i, j], py[i, j])
                Ekin_after = self.ledger.kinetic_energy(rho, pvec_after)
                # Compute delta kinetic energy; ensure finite and non-negative.
                # Handle case where both energies are very large (could produce NaN)
                if not math.isfinite(Ekin_before) or not math.isfinite(Ekin_after):
                    dE = 0.0
                else:
                    raw_dE = Ekin_before - Ekin_after
                    if not math.isfinite(raw_dE) or raw_dE <= 0.0:
                        dE = 0.0
                    else:
                        dE = min(raw_dE, ENERGY_DELTA_MAX)
                # convert to heat; allocate some to radiation based on opacity and T
                f_rad = clamp((1.0 - opacity_eff) * (T / (T + 1.0)), 0.0, 1.0)
                self.ledger.convert_kinetic_to_heat(i, j, dE, f_rad)
            # shock heating due to compression (approximate negative divergence)
            # compute divergence from current velocity field (approx gradient)
            # Use boundary-aware neighbor access
            px_ip1 = px[ip1, j] if ip1_valid else px[i, j]
            px_im1 = px[im1, j] if im1_valid else px[i, j]
            py_jp1 = py[i, jp1] if jp1_valid else py[i, j]
            py_jm1 = py[i, jm1] if jm1_valid else py[i, j]
            
            dvx = (px_ip1 - px_im1) * 0.5
            dvy = (py_jp1 - py_jm1) * 0.5
            div_v = (dvx + dvy) / max(rho, 1e-12)
            if div_v < 0:
                # compression
                dE_shock = self.cfg.shock_k * (-div_v) * rho * dt_sub
                E_heat[i, j] += dE_shock
            # radiation absorption
            absorb = self.cfg.rad_to_heat_absorb_rate * E_rad[i, j] * dt_sub
            E_rad[i, j] -= absorb
            E_heat[i, j] += absorb
            # check high-energy events at lower frequency
            if m % max(1, M // 4) == 0:
                self.materials.handle_high_energy_events(
                    self.fabric, self.ledger, self.registry, i, j, derived.Z_field[i, j], T, tick, attempt=m
                )

    def transfer_mass(self, src_i: int, src_j: int, dst_i: int, dst_j: int, mass: float, v_x: float, v_y: float) -> None:
        """Move a quantity of mass from ``src`` to ``dst`` with associated momentum and energy.

        ``mass`` is removed from the source cell's density and added to
        the destination. Momentum is scaled accordingly using the
        current velocity at the source. Heat and radiation are moved
        proportionally. Species masses are moved proportionally as well.
        Mixtures are maintained with top-K filtering.
        """
        if mass <= 0.0:
            return
        # clamp mass to available
        avail = self.fabric.rho[src_i, src_j]
        if avail <= 0.0:
            return
        if mass > avail:
            mass = avail
        # remove from source
        self.fabric.rho[src_i, src_j] -= mass
        # compute fraction moved
        frac = mass / avail if avail > 0 else 0
        # move momentum
        dp_x = self.fabric.px[src_i, src_j] * frac
        dp_y = self.fabric.py[src_i, src_j] * frac
        self.fabric.px[src_i, src_j] -= dp_x
        self.fabric.py[src_i, src_j] -= dp_y
        # add to destination
        self.fabric.rho[dst_i, dst_j] += mass
        self.fabric.px[dst_i, dst_j] += dp_x
        self.fabric.py[dst_i, dst_j] += dp_y
        # move heat and radiation
        dE_heat = self.fabric.E_heat[src_i, src_j] * frac
        dE_rad = self.fabric.E_rad[src_i, src_j] * frac
        self.fabric.E_heat[src_i, src_j] -= dE_heat
        self.fabric.E_rad[src_i, src_j] -= dE_rad
        self.fabric.E_heat[dst_i, dst_j] += dE_heat
        self.fabric.E_rad[dst_i, dst_j] += dE_rad
        # move mixture
        src_mix = self.fabric.mixtures[src_i][src_j]
        dst_mix = self.fabric.mixtures[dst_i][dst_j]
        # For each species in source mixture, move mass proportionally
        if src_mix.total_mass() > 0:
            for idx in range(len(src_mix.species_ids)):
                sid = src_mix.species_ids[idx]
                mval = src_mix.masses[idx]
                moved = mval * frac
                src_mix.masses[idx] -= moved
                dst_mix.add_species_mass(sid, moved, self.cfg.mixture_top_k)
            # cleanup mixtures
            src_mix.cleanup(self.cfg.mixture_eps_merge, self.cfg.mixture_top_k)
            dst_mix.cleanup(self.cfg.mixture_eps_merge, self.cfg.mixture_top_k)