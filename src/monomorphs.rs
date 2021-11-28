use numpy::PyReadonlyArrayDyn;
use pyo3::{prelude::*, PyObjectProtocol};

use crate::ragged_buffer::RaggedBuffer;

#[pyclass]
pub struct RaggedBufferF32(RaggedBuffer<f32>);

#[pymethods]
impl RaggedBufferF32 {
    #[new]
    pub fn new(features: usize) -> Self {
        RaggedBufferF32(RaggedBuffer::new(features))
    }

    fn push(&mut self, features: PyReadonlyArrayDyn<f32>) {
        self.0.push(features);
    }

    fn clear(&mut self) {
        self.0.clear();
    }
}

#[pyproto]
impl PyObjectProtocol for RaggedBufferF32 {
    fn __str__(&self) -> PyResult<String> {
        self.0.__str__()
    }
}

#[pyclass]
pub struct RaggedBufferI64(RaggedBuffer<i64>);

#[pymethods]
impl RaggedBufferI64 {
    #[new]
    pub fn new(features: usize) -> Self {
        RaggedBufferI64(RaggedBuffer::new(features))
    }

    fn push(&mut self, features: PyReadonlyArrayDyn<i64>) {
        self.0.push(features);
    }

    fn clear(&mut self) {
        self.0.clear();
    }
}

#[pyproto]
impl PyObjectProtocol for RaggedBufferI64 {
    fn __str__(&self) -> PyResult<String> {
        self.0.__str__()
    }
}
