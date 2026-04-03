# vasprunrs

A fast parser for VASP `vasprun.xml` files, written in Rust with Python bindings.

`vasprunrs` is a drop-in replacement for `pymatgen.io.vasp.outputs.Vasprun` that parses the same files 5–10x faster using a streaming Rust XML parser. It returns native pymatgen objects (`Structure`, `CompleteDos`, `BandStructure`, etc.) so existing code needs only a one-line import change.

## Installation

```bash
pip install vasprunrs

# with pymatgen compatibility shim (recommended)
pip install "vasprunrs[pymatgen]"
```

## Usage

### Drop-in replacement for pymatgen

```python
# Before
from pymatgen.io.vasp import Vasprun

# After
from vasprunrs.pymatgen import Vasprunrs as Vasprun
```

```python
vr = Vasprun("vasprun.xml")

structure      = vr.final_structure       # pymatgen Structure
dos            = vr.complete_dos          # pymatgen CompleteDos
band_structure = vr.get_band_structure()  # pymatgin BandStructure
eigenvalues    = vr.eigenvalues           # {Spin.up: ndarray, ...}
```

### Native API (no pymatgen required)

```python
from vasprunrs import Vasprun

vr = Vasprun("vasprun.xml", parse_dos=True, parse_eigen=True)

print(vr.version)           # "6.4.2"
print(vr.efermi)            # 4.123 (eV)
print(vr.final_structure)   # dict with lattice, positions, species
print(vr.eigenvalues.shape) # (nspins, nkpts, nbands, 2)
print(vr.dos)               # dict with efermi, total, partial
```

## Features

| Feature | Status |
|---------|--------|
| INCAR parameters (typed) | supported |
| Atomic info (mass, valence, pseudopotential) | supported |
| K-points (mesh, weights, labels) | supported |
| Initial and final structures | supported |
| Ionic steps (forces, stress, energies) | supported |
| Eigenvalues and occupancies | supported |
| Projected eigenvalues (LORBIT ≥ 10) | supported (opt-in) |
| Total DOS | supported |
| Partial DOS (site + orbital) | supported |
| Dielectric function | supported |
| Fermi level | supported |
| Spin-polarized calculations | supported |
| Non-collinear / SOC | supported |
| Line-mode k-points (band structure) | supported |
| GW post-processing | supported |
| Overflow values (NaN) | handled |
| Magnetic moments (per-atom, collinear + non-collinear) | supported |
| Force constants / phonons | not yet implemented |
| Born effective charges | not yet implemented |

## Performance

`vasprunrs` uses a streaming XML parser that does not load the entire file into memory. Parse times are typically 5–10x faster than pymatgen's Python parser.

## Parse options

Both the native `Vasprun` and the compatibility `Vasprunrs` accept the same options:

| Option | Default | Description |
|--------|---------|-------------|
| `parse_dos` | `True` | Parse density of states |
| `parse_eigen` | `True` | Parse eigenvalues |
| `parse_projected_eigen` | `False` | Parse projected eigenvalues (can be large) |
| `ionic_step_skip` | `None` | Parse every Nth ionic step |
| `ionic_step_offset` | `0` | Skip the first N ionic steps |

## Known limitations

- `converged_electronic` and `converged_ionic` are not tracked separately (both reflect the same value)
- Epsilon static/ionic tensors (`epsilon_static`, `epsilon_ionic`) are not parsed from the DFPT block
- Force constants / phonons and Born effective charges are not yet implemented

> **Note on per-atom magnetic moments:** `Vasprunrs.magnetization` reads the
> `<varray name="magnetization">` block from each ionic step.  This block is
> written by **VASP 5.x** but is **absent in VASP 6.x** output, so
> `magnetization` will return `None` for all VASP 6.x files regardless of
> whether the calculation is magnetic.  Per-atom moments for VASP 6.x can be
> recovered by integrating the partial DOS up to the Fermi energy
> (spin-up minus spin-down per atom), which requires `parse_dos=True` and
> `LORBIT ≥ 11` in the INCAR.

## Requirements

- Python 3.9+
- numpy
- pymatgen (optional, for `vasprunrs[pymatgen]`)

## License

MIT
