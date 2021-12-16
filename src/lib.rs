use pyo3::prelude::*;

pub mod monomorphs;
pub mod ragged_buffer;

#[pymodule]
fn ragged_buffer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<monomorphs::RaggedBufferF32>()?;
    m.add_class::<monomorphs::RaggedBufferI64>()?;
    m.add_class::<monomorphs::RaggedBufferBool>()?;
    Ok(())
}
