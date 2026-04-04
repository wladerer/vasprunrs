use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::IntoPyArray;
use crate::{parse_file, ParseOptions};
use crate::types::*;

// ---------------------------------------------------------------------------
// Top-level Python class
// ---------------------------------------------------------------------------

#[pyclass(name = "Vasprun")]
pub struct PyVasprun {
    inner: Vasprun,
}

#[pymethods]
impl PyVasprun {
    #[new]
    #[pyo3(signature = (
        path,
        parse_dos            = true,
        parse_eigen          = true,
        parse_projected      = false,
        ionic_step_skip      = None,
        ionic_step_offset    = 0,
    ))]
    fn new(
        path: &str,
        parse_dos: bool,
        parse_eigen: bool,
        parse_projected: bool,
        ionic_step_skip: Option<usize>,
        ionic_step_offset: usize,
    ) -> PyResult<Self> {
        let opts = ParseOptions {
            parse_dos,
            parse_eigen,
            parse_projected,
            ionic_step_skip,
            ionic_step_offset,
        };
        let inner = parse_file(path, opts)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyVasprun { inner })
    }

    // ---- generator ---------------------------------------------------------

    #[getter]
    fn program(&self) -> &str { &self.inner.generator.program }
    #[getter]
    fn version(&self) -> &str { &self.inner.generator.version }

    // ---- incar -------------------------------------------------------------

    #[getter]
    fn incar(&self, py: Python) -> PyResult<PyObject> {
        let d = PyDict::new(py);
        for (k, v) in &self.inner.incar {
            let pyval: PyObject = match v {
                IncarValue::Int(i)   => i.into_py(py),
                IncarValue::Float(f) => f.into_py(py),
                IncarValue::Bool(b)  => b.into_py(py),
                IncarValue::Str(s)   => s.into_py(py),
                IncarValue::Vec(vs)  => vs.clone().into_py(py),
            };
            d.set_item(k, pyval)?;
        }
        Ok(d.into())
    }

    // ---- atominfo ----------------------------------------------------------

    #[getter]
    fn atoms(&self, py: Python) -> PyObject {
        self.inner.atominfo.atoms.clone().into_py(py)
    }

    #[getter]
    fn atom_types(&self, py: Python) -> PyResult<PyObject> {
        let list = PyList::empty(py);
        for at in &self.inner.atominfo.atom_types {
            let d = PyDict::new(py);
            d.set_item("element",         &at.element)?;
            d.set_item("count",           at.count)?;
            d.set_item("mass",            at.mass)?;
            d.set_item("valence",         at.valence)?;
            d.set_item("pseudopotential", &at.pseudopotential)?;
            list.append(d)?;
        }
        Ok(list.into())
    }

    // ---- kpoints -----------------------------------------------------------

    #[getter]
    fn kpoints(&self, py: Python) -> PyResult<PyObject> {
        let d = PyDict::new(py);
        let kl = self.inner.kpoints.kpointlist.clone().into_pyarray(py);
        let w  = self.inner.kpoints.weights.clone().into_pyarray(py);
        d.set_item("kpointlist", kl)?;
        d.set_item("weights", w)?;
        if let Some(ref gen) = self.inner.kpoints.generation {
            d.set_item("scheme", &gen.scheme)?;
            if let Some(div) = gen.divisions {
                d.set_item("divisions", div.to_vec())?;
            }
        }
        if !self.inner.kpoints.labels.is_empty() {
            let labels: Vec<(usize, &str)> = self.inner.kpoints.labels
                .iter()
                .map(|(i, s)| (*i, s.as_str()))
                .collect();
            d.set_item("labels", labels)?;
        }
        Ok(d.into())
    }

    // ---- structures --------------------------------------------------------

    #[getter]
    fn initial_structure(&self, py: Python) -> PyResult<PyObject> {
        structure_to_dict(py, &self.inner.initial_structure)
    }

    #[getter]
    fn final_structure(&self, py: Python) -> PyResult<PyObject> {
        structure_to_dict(py, &self.inner.final_structure)
    }

    // ---- ionic steps -------------------------------------------------------

    #[getter]
    fn ionic_steps(&self, py: Python) -> PyResult<PyObject> {
        let list = PyList::empty(py);
        for step in &self.inner.ionic_steps {
            let d = PyDict::new(py);
            d.set_item("structure",   structure_to_dict(py, &step.structure)?)?;
            d.set_item("e_fr_energy", step.energy.e_fr_energy)?;
            d.set_item("e_wo_entrp",  step.energy.e_wo_entrp)?;
            d.set_item("e_0_energy",  step.energy.e_0_energy)?;
            let forces: Vec<Vec<f64>> = step.forces.iter().map(|f| f.to_vec()).collect();
            d.set_item("forces", forces)?;
            let stress: Vec<Vec<f64>> = step.stress.iter().map(|r| r.to_vec()).collect();
            d.set_item("stress", stress)?;
            if let Some(ref mag) = step.magnetization {
                d.set_item("magnetization", mag.clone())?;
            }
            let scf: Vec<_> = step.scf_steps.iter().map(|s| {
                let sd = PyDict::new(py);
                let _ = sd.set_item("e_fr_energy", s.e_fr_energy);
                let _ = sd.set_item("e_wo_entrp",  s.e_wo_entrp);
                let _ = sd.set_item("e_0_energy",  s.e_0_energy);
                sd.into_any().unbind()
            }).collect();
            d.set_item("scf_steps", scf)?;
            list.append(d)?;
        }
        Ok(list.into())
    }

    // ---- eigenvalues -------------------------------------------------------

    /// ndarray shape (nspins, nkpts, nbands, 2).  Axis-3: [energy, occupancy].
    #[getter]
    fn eigenvalues(&self, py: Python) -> PyObject {
        match &self.inner.eigenvalues {
            Some(e) => e.data.clone().into_pyarray(py).into_any().unbind(),
            None    => py.None(),
        }
    }

    #[getter]
    fn eigenvalue_shape(&self) -> Option<(usize, usize, usize)> {
        self.inner.eigenvalues.as_ref().map(|e| (e.nspins, e.nkpts, e.nbands))
    }

    // ---- projected eigenvalues ---------------------------------------------

    /// Returns a dict with keys ``"data"`` (ndarray shape
    /// ``(nspins, nkpts, nbands, nions, norbitals)``) and ``"orbitals"``
    /// (list of orbital label strings), or ``None`` if not parsed.
    #[getter]
    fn projected(&self, py: Python) -> PyResult<PyObject> {
        let Some(ref proj) = self.inner.projected else { return Ok(py.None()) };
        let d = PyDict::new(py);
        d.set_item("data",     proj.data.clone().into_pyarray(py))?;
        d.set_item("orbitals", proj.orbitals.clone())?;
        Ok(d.into())
    }

    // ---- efermi ------------------------------------------------------------

    #[getter]
    fn efermi(&self) -> Option<f64> { self.inner.efermi }

    // ---- DOS ---------------------------------------------------------------

    #[getter]
    fn dos(&self, py: Python) -> PyResult<PyObject> {
        let Some(ref dos) = self.inner.dos else { return Ok(py.None()) };
        let d = PyDict::new(py);
        d.set_item("efermi", dos.efermi)?;

        let td = PyDict::new(py);
        td.set_item("energies",   dos.total.energies.clone().into_pyarray(py))?;
        td.set_item("densities",  dos.total.densities.clone().into_pyarray(py))?;
        td.set_item("integrated", dos.total.integrated.clone().into_pyarray(py))?;
        d.set_item("total", td)?;

        if let Some(ref pdos) = dos.partial {
            let pd = PyDict::new(py);
            pd.set_item("data",     pdos.data.clone().into_pyarray(py))?;
            pd.set_item("orbitals", pdos.orbitals.clone())?;
            d.set_item("partial", pd)?;
        }
        Ok(d.into())
    }

    // ---- dielectric --------------------------------------------------------

    #[getter]
    fn dielectric(&self, py: Python) -> PyResult<PyObject> {
        let Some(ref diel) = self.inner.dielectric else { return Ok(py.None()) };
        let d = PyDict::new(py);
        d.set_item("energies", diel.energies.clone().into_pyarray(py))?;
        d.set_item("real",     diel.real.clone().into_pyarray(py))?;
        d.set_item("imag",     diel.imag.clone().into_pyarray(py))?;
        Ok(d.into())
    }

    // ---- convenience -------------------------------------------------------

    #[getter]
    fn converged(&self) -> bool {
        fn incar_f64(incar: &indexmap::IndexMap<String, IncarValue>, key: &str, default: f64) -> f64 {
            match incar.get(key) {
                Some(IncarValue::Float(f)) => *f,
                Some(IncarValue::Int(i))   => *i as f64,
                _ => default,
            }
        }
        fn incar_i64(incar: &indexmap::IndexMap<String, IncarValue>, key: &str, default: i64) -> i64 {
            match incar.get(key) {
                Some(IncarValue::Int(i))   => *i,
                Some(IncarValue::Float(f)) => *f as i64,
                _ => default,
            }
        }

        let incar  = &self.inner.incar;
        let ediff  = incar_f64(incar, "EDIFF",  1e-4);
        let ediffg = incar_f64(incar, "EDIFFG", -0.01);
        let ibrion = incar_i64(incar, "IBRION", -1);
        let nsw    = incar_i64(incar, "NSW",     0);

        let Some(last) = self.inner.ionic_steps.last() else { return false };

        let is_relax = ibrion != -1 && nsw > 0;

        if !is_relax {
            // Single-point: check electronic convergence only (last SCF dE < EDIFF)
            let scf = &last.scf_steps;
            if scf.len() >= 2 {
                let de = (scf[scf.len()-1].e_wo_entrp - scf[scf.len()-2].e_wo_entrp).abs();
                return de < ediff;
            }
            return false;
        }

        // Ionic convergence
        if ediffg < 0.0 {
            // Force criterion: Fmax over unfrozen atoms < |EDIFFG|
            let sel    = &last.structure.selective;
            let fmax   = last.forces.iter().enumerate()
                .filter(|(i, _)| {
                    sel.as_ref()
                        .and_then(|s| s.get(*i))
                        .map(|f| f.iter().any(|&b| b))
                        .unwrap_or(true)
                })
                .map(|(_, f)| (f[0]*f[0] + f[1]*f[1] + f[2]*f[2]).sqrt())
                .fold(0.0f64, f64::max);
            fmax < ediffg.abs()
        } else {
            // Energy criterion: ionic |dE| < EDIFFG
            let steps = &self.inner.ionic_steps;
            steps.len() >= 2 && {
                let de = (steps[steps.len()-1].energy.e_wo_entrp
                        - steps[steps.len()-2].energy.e_wo_entrp).abs();
                de < ediffg
            }
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Vasprun(program={}, version={}, atoms={:?}, nkpts={}, nsteps={})",
            self.inner.generator.program,
            self.inner.generator.version,
            self.inner.atominfo.atoms,
            self.inner.kpoints.nkpts(),
            self.inner.ionic_steps.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn structure_to_dict(py: Python, s: &Structure) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    let lat: Vec<Vec<f64>> = s.lattice.iter().map(|r| r.to_vec()).collect();
    d.set_item("lattice",   lat)?;
    d.set_item("volume",    s.volume)?;
    let pos: Vec<Vec<f64>> = s.positions.iter().map(|p| p.to_vec()).collect();
    d.set_item("positions", pos)?;
    d.set_item("species",   &s.species)?;
    if let Some(ref sel) = s.selective {
        let flags: Vec<Vec<bool>> = sel.iter().map(|f| f.to_vec()).collect();
        d.set_item("selective", flags)?;
    }
    Ok(d.into())
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVasprun>()?;
    Ok(())
}
