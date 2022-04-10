#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::wrap_pyfunction;
#[cfg(feature = "python")]
pub mod monomorphs;

pub mod ragged_buffer;

#[cfg(feature = "python")]
pub mod ragged_buffer_view;

#[cfg(feature = "python")]
#[pymodule]
fn ragged_buffer(_py: Python, m: &PyModule) -> PyResult<()> {
    // New exports also have to be added to __init__.py
    m.add_class::<monomorphs::RaggedBufferF32>()?;
    m.add_class::<monomorphs::RaggedBufferI64>()?;
    m.add_class::<monomorphs::RaggedBufferBool>()?;
    m.add_function(wrap_pyfunction!(translate_rotate, m)?)?;
    Ok(())
}

#[cfg(feature = "python")]
#[pyfunction]
fn translate_rotate(
    source: &monomorphs::RaggedBufferF32,
    translation: monomorphs::RaggedBufferF32,
    rotation: monomorphs::RaggedBufferF32,
) -> PyResult<()> {
    ragged_buffer_view::translate_rotate(&source.0, &translation.0, &rotation.0)
}
