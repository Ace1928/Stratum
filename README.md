# Stratum - Emergent Layered Physical Simulation Engine

Stratum is a prototype simulation engine designed to model emergent, layered physical phenomena. It provides a framework for simulating complex physical systems with multiple interacting scales, from stellar dynamics to chemical reactions.

## Overview

The Stratum engine implements a hierarchical simulation architecture with:

- **Core Engine**: Handles the fundamental simulation mechanics including configuration, field storage, event scheduling, and signal propagation
- **Domain Modules**: Implement specific physical behaviors like high-energy materials physics
- **Scenarios**: Define complete simulation setups and run configurations

## Features

- **Grid-based simulation** with configurable dimensions and boundary conditions (periodic, reflective, open)
- **Multi-scale physics** with microtick-based local resolution and global diffusion operators
- **Species registry** for tracking material compositions with quantized high-energy properties
- **Deterministic replay** support via entropy source recording
- **High-energy event handling** including:
  - Fusion reactions
  - Decay processes
  - Degenerate matter transitions
  - Black hole formation
- **Signal propagation** system for influence, radiation, and impulse delivery
- **Dynamic level-of-detail** (LOD) for performance optimization

## Architecture

```
stratum/
├── core/                  # Core engine components
│   ├── config.py          # EngineConfig dataclass with simulation parameters
│   ├── fabric.py          # Fabric class for field storage and spatial operations
│   ├── ledger.py          # Energy conservation and entropy management
│   ├── metronome.py       # Timing and compute budget allocation
│   ├── quanta.py          # Event propagation and microtick resolution
│   ├── registry.py        # Species registry with property quantization
│   └── types.py           # Core type definitions (Vec2, Cell, etc.)
├── domains/               # Domain-specific physics modules
│   └── materials/         # High-energy materials subsystem
│       ├── __init__.py
│       └── fundamentals.py # Material rules, EOS, fusion/decay/BH logic
├── scenarios/             # Simulation scenarios
│   ├── stellar_collapse.py           # Basic stellar collapse scenario
│   ├── stellar_collapse_runtime.py   # Runtime-controlled simulation
│   └── stellar_screensaver.py        # Real-time Pygame visualizer
├── util/                  # Utilities
│   └── classregistry.py   # Code introspection utility
└── tests/                 # Comprehensive test suite
    ├── conftest.py        # Pytest configuration
    ├── test_*.py          # Unit and integration tests
```

## Installation

### Requirements

- Python 3.9+
- NumPy
- Matplotlib (for image generation)
- Pygame (optional, for real-time visualization)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Running the Basic Simulation

```bash
# Run the stellar collapse scenario
python -m scenarios.stellar_collapse

# Run with custom parameters
python -c "
from scenarios.stellar_collapse import run_stellar_collapse
run_stellar_collapse(grid_size=64, num_ticks=1000)
"
```

### Runtime-Controlled Simulation

Run a simulation for a specific wall-clock duration:

```bash
# Run for 30 seconds with snapshots every second
python -m scenarios.stellar_collapse_runtime --runtime 30 --snapshot 1.0

# With custom parameters
python -m scenarios.stellar_collapse_runtime \
    --grid 64 \
    --runtime 60 \
    --microticks 500 \
    --snapshot 2.0 \
    --output ./my_outputs
```

### Real-Time Screensaver

Launch the interactive Pygame visualization:

```bash
# Auto-size to display
python -m scenarios.stellar_screensaver

# With custom grid and FPS
python -m scenarios.stellar_screensaver --grid 256 --fps 60

# Controls:
#   ESC/Q: Quit
#   SPACE: Pause/Resume
#   R: Restart with new seed
#   TAB: Toggle log scaling
#   1/2: Bias LOD toward speed/quality
#   +/-: Adjust target FPS
```

## Configuration

The `EngineConfig` dataclass controls all simulation parameters:

```python
from core.config import EngineConfig

config = EngineConfig(
    # Grid dimensions
    grid_w=128,
    grid_h=128,
    
    # Random seed for reproducibility
    base_seed=42,
    entropy_mode=False,  # Set True for run-to-run variation
    
    # Boundary condition: PERIODIC, REFLECTIVE, or OPEN
    boundary="PERIODIC",
    
    # Physics parameters
    gravity_strength=0.05,
    eos_gamma=2.0,
    thermal_pressure_coeff=0.1,
    
    # High-energy thresholds
    Z_fuse_min=1.5,   # Fusion threshold
    Z_deg_min=3.0,    # Degeneracy threshold
    Z_bh_min=4.5,     # Black hole threshold
)
```

## Core Concepts

### Fabric

The `Fabric` class stores all spatial field data:
- `rho`: Mass density
- `px`, `py`: Momentum components
- `E_heat`, `E_rad`: Heat and radiation energy
- `influence`: Gravitational influence field
- `BH_mass`: Black hole mass accumulator
- `EH_mask`: Event horizon mask
- `mixtures`: Per-cell species composition

### Species Registry

Species are identified by quantized high-energy (HE) properties:
- `HE/rho_max`: Maximum packing density
- `HE/chi`: EOS stiffness coefficient
- `HE/eta`: Viscosity
- `HE/opacity`: Radiation opacity
- `HE/beta`: Binding depth
- `HE/lambda`: Decay instability
- And more...

### Quanta System

The Quanta subsystem handles:
1. **Signal delivery**: Influence, radiation, and impulse signals
2. **Active region selection**: Identifies cells requiring detailed simulation
3. **Microtick resolution**: Local physics updates within each tick
4. **Mass transfer**: Conservation-aware mass movement between cells

### High-Energy Events

Events are triggered based on the compression index Z and local conditions:
- **Fusion**: Creates heavier species, releases binding energy
- **Decay**: Breaks species into lighter fragments
- **Degenerate transition**: Converts to stiff degenerate matter
- **Black hole formation**: Absorbs cell contents at extreme Z

## Testing

Run the full test suite:

```bash
# Install pytest if needed
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_fabric.py -v

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=. --cov-report=term-missing
```

## Development

### Adding New Species Properties

1. Update `SpeciesRegistry` property definitions
2. Implement property calculations in `MaterialsFundamentals`
3. Use `registry.migrate_le_properties()` for existing species

### Adding New Physical Effects

1. Create a new method in `MaterialsFundamentals`
2. Hook into `handle_high_energy_events()` or `apply_global_ops()`
3. Add appropriate tests in `tests/test_materials.py`

### Extending the Core Engine

1. Add new fields to `Fabric` if needed
2. Update `EngineConfig` for new parameters
3. Implement processing in `Quanta.resolve_cell()`

## License

This project is provided for educational and research purposes.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Acknowledgments

The Stratum architecture is inspired by multi-scale physics simulation systems and emergent behavior modeling frameworks.
