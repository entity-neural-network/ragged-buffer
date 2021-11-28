use numpy::{PyReadonlyArray3, PyReadonlyArrayDyn};
use pyo3::types::PyType;
use pyo3::{prelude::*, PyNumberProtocol, PyObjectProtocol};

use crate::ragged_buffer::RaggedBuffer;

#[pyclass]
pub struct RaggedBufferF32(RaggedBuffer<f32>);

#[pymethods]
impl RaggedBufferF32 {
    #[new]
    pub fn new(features: usize) -> Self {
        RaggedBufferF32(RaggedBuffer::new(features))
    }
    #[classmethod]
    fn from_array(_cls: &PyType, array: PyReadonlyArray3<f32>) -> Self {
        RaggedBufferF32(RaggedBuffer::from_array(array))
    }
    fn push(&mut self, features: PyReadonlyArrayDyn<f32>) {
        self.0.push(features);
    }

    fn clear(&mut self) {
        self.0.clear();
    }

    fn as_array<'a>(
        &self,
        py: Python<'a>,
    ) -> &'a numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 2]>> {
        self.0.as_array(py)
    }

    fn extend(&mut self, other: &RaggedBufferF32) -> PyResult<()> {
        self.0.extend(&other.0)
    }
    fn size0(&self) -> usize {
        self.0.size0()
    }
    fn size1(&self, i: usize) -> PyResult<usize> {
        self.0.size1(i)
    }
    fn size2(&self) -> usize {
        self.0.size2()
    }
}

#[pyproto]
impl PyObjectProtocol for RaggedBufferF32 {
    fn __str__(&self) -> PyResult<String> {
        self.0.__str__()
    }
    fn __repr__(&self) -> PyResult<String> {
        self.0.__str__()
    }
}

#[pyproto]
impl PyNumberProtocol for RaggedBufferI64 {
    fn __add__(lhs: RaggedBufferI64, rhs: RaggedBufferI64) -> PyResult<RaggedBufferI64> {
        Ok(RaggedBufferI64(lhs.0.add(&rhs.0)?))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct RaggedBufferI64(RaggedBuffer<i64>);

#[pymethods]
impl RaggedBufferI64 {
    #[new]
    pub fn new(features: usize) -> Self {
        RaggedBufferI64(RaggedBuffer::new(features))
    }
    #[classmethod]
    fn from_array(_cls: &PyType, array: PyReadonlyArray3<i64>) -> Self {
        RaggedBufferI64(RaggedBuffer::from_array(array))
    }

    fn push(&mut self, features: PyReadonlyArrayDyn<i64>) {
        self.0.push(features);
    }

    fn clear(&mut self) {
        self.0.clear();
    }

    fn as_array<'a>(
        &self,
        py: Python<'a>,
    ) -> &'a numpy::PyArray<i64, numpy::ndarray::Dim<[usize; 2]>> {
        self.0.as_array(py)
    }

    fn extend(&mut self, other: &RaggedBufferI64) -> PyResult<()> {
        self.0.extend(&other.0)
    }
    fn size0(&self) -> usize {
        self.0.size0()
    }
    fn size1(&self, i: usize) -> PyResult<usize> {
        self.0.size1(i)
    }
    fn size2(&self) -> usize {
        self.0.size2()
    }
}

#[pyproto]
impl PyObjectProtocol for RaggedBufferI64 {
    fn __str__(&self) -> PyResult<String> {
        self.0.__str__()
    }
    fn __repr__(&self) -> PyResult<String> {
        self.0.__str__()
    }
}
