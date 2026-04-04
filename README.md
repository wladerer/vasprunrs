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

## Command-line interface

```bash
pip install "vasprunrs[cli]"
```

### Convergence summary

```bash
# defaults to ./vasprun.xml
vasprunrs

# explicit file
vasprunrs path/to/vasprun.xml

# watch a running calculation, refresh every 5 s
vasprunrs vasprun.xml --watch 5
```

Example output for a geometry optimization:

```
  fe_relax.xml
  ------------
  VASP 6.4.1  vasp
  NCORE=4  NPAR=2  KPAR=1
  Atoms: 2 x Fe    K-points: 29

  Calculation
  Type   : Geometry optimization (CG)
  Cell   : Relax ions + shape + volume
  Max    : NSW=50 ionic steps,  NELM=60 SCF/step
  EDIFF  : 1.0e-05 eV  (electronic convergence)
  EDIFFG : -0.01 eV/A  (force convergence)

  Ionic convergence
  #      E_wo_entrp (eV)   dE (eV)       log|dE|   Fmax (eV/A)  nSCF
  -----  ----------------  ------------  --------  -----------  -----
  1      -14.123456        --            --        0.2340       42
  2      -14.145678        -2.222e-02    -1.65     0.0980       28
  3      -14.152341        -6.663e-03    -2.18     0.0310       21
  4      -14.153012        -6.710e-04    -3.17     0.0080*      18

  Last ionic step -- electronic convergence
   SCF   E_wo_entrp (eV)       dE (eV)   log|dE|
  ----  ----------------  ------------  --------
     1        -14.160000            --        --
     ...

  Result: CONVERGED  (Fmax 0.0080 <= |EDIFFG| 0.0100 eV/A)
  Trend : -0.51 log-units/step
```

Column legend: `*` on Fmax = force criterion met; `!` on Fmax/nSCF = NELM hit (electronic convergence not reached); `+` suffix on step number = energy increased.

Optional columns:

| Flag | Column added |
|------|-------------|
| `-e` / `--toten` | TOTEN (free energy with entropy) |
| `-a` / `--favg` | Average force magnitude |
| `-x` / `--fmaxis` | Axis of largest force component |
| `-i` / `--fmidx` | Atom index with largest force (1-based) |
| `-v` / `--volume` | Cell volume per step |
| `--no-fmax` | Suppress Fmax |
| `--no-lgde` | Suppress log\|dE\| |
| `--no-magmom` | Suppress magnetic moment |
| `--no-nscf` | Suppress SCF count |

Selective dynamics are read directly from `vasprun.xml` — frozen atoms are automatically excluded from Fmax without needing a separate POSCAR file.

### Export band structure data

```bash
vasprunrs bands vasprun.xml -o bands.npz
vasprunrs bands vasprun.xml -o bands.npz --shift-efermi --projected
```

| Key | Shape | Description |
|-----|-------|-------------|
| `eigenvalues` | `(nspins, nkpts, nbands)` | Band energies in eV |
| `occupancies` | `(nspins, nkpts, nbands)` | Occupancies |
| `kpoints` | `(nkpts, 3)` | Fractional reciprocal coordinates |
| `weights` | `(nkpts,)` | K-point weights |
| `efermi` | scalar | Fermi energy in eV |
| `labels` | object array | `(kpt_index, label)` pairs for high-symmetry points |
| `projected` | `(nspins, nkpts, nbands, nions, norbitals)` | Orbital weights (`--projected` only) |
| `orbitals` | object array | Orbital label strings (`--projected` only) |

### Export DOS data

```bash
vasprunrs dos vasprun.xml -o dos.npz
```

| Key | Shape | Description |
|-----|-------|-------------|
| `energies` | `(nedos,)` | Energy grid in eV |
| `total` | `(nspins, nedos)` | Total DOS |
| `integrated` | `(nspins, nedos)` | Integrated DOS |
| `efermi` | scalar | Fermi energy in eV |
| `partial` | `(nspins, nions, norbitals, nedos)` | Partial DOS (if present) |
| `orbitals` | object array | Orbital label strings (if partial DOS present) |

### Plotting projected band structures

`scripts/plot_bands.py` reads the `.npz` from `vasprunrs bands --projected` and produces a fat band plot where dot size encodes orbital character.

```bash
# plain band structure
python scripts/plot_bands.py bands.npz

# fat bands: d-orbital character
python scripts/plot_bands.py bands.npz --orbital d

# multiple orbital groups with distinct colors
python scripts/plot_bands.py bands.npz --orbital s p d

# spin-down channel, save to file
python scripts/plot_bands.py bands.npz --orbital d --spin 1 -o fatbands.png

# adjust energy window and dot scale
python scripts/plot_bands.py bands.npz --orbital d --emin -3 --emax 3 --scale 300
```

Supported orbital group names: `s`, `p`, `d`, `f`, `sp`, `pd` — or any individual orbital name (e.g. `dxy`, `x2-y2`). High-symmetry labels and line-mode segment breaks are handled automatically.

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
