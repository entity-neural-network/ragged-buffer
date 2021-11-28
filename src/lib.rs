use pyo3::{prelude::*, wrap_pyfunction};

pub mod monomorphs;
pub mod ragged_buffer;

#[pyfunction]
pub fn test(py: Python, s: &str) -> PyResult<PyObject> {
    Ok(format!("Hello from Rust: {}", s).into_py(py))
}

#[pymodule]
fn ragged_buffer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test, m)?).unwrap();
    m.add_class::<monomorphs::RaggedBufferF32>()?;
    m.add_class::<monomorphs::RaggedBufferI64>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
