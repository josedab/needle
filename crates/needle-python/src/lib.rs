//! Thin cdylib wrapper that re-exports the Needle Python module.
//!
//! This crate exists so the root `needle` crate can use `crate-type = ["rlib"]`
//! for fast development builds while still providing a cdylib for maturin.

use pyo3::prelude::*;

#[pymodule]
fn needle(m: &Bound<'_, PyModule>) -> PyResult<()> {
    ::needle::python::needle(m)
}
