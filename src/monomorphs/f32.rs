use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArrayDyn};
use pyo3::basic::CompareOp;
use pyo3::types::PyType;
use pyo3::{prelude::*, PyMappingProtocol, PyNumberProtocol, PyObjectProtocol};

use crate::ragged_buffer::RaggedBuffer;

use super::IndicesOrInt;

#[pyclass]
#[derive(Clone)]
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
    #[classmethod]
    fn from_flattened(
        _cls: &PyType,
        flattened: PyReadonlyArray2<f32>,
        lengths: PyReadonlyArray1<i64>,
    ) -> Self {
        RaggedBufferF32(RaggedBuffer::from_flattened(flattened, lengths))
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
    fn size1(&self, py: Python, i: Option<usize>) -> PyResult<PyObject> {
        match i {
            Some(i) => self.0.size1(i).map(|s| s.into_py(py)),
            None => Ok(self.0.lengths(py).into_py(py)),
        }
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

    fn __richcmp__(&self, other: RaggedBufferF32, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.0 == other.0),
            CompareOp::Ne => Ok(self.0 != other.0),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Only == and != are supported",
            )),
        }
    }
}

#[pyproto]
impl PyNumberProtocol for RaggedBufferF32 {
    fn __add__(lhs: RaggedBufferF32, rhs: RaggedBufferF32) -> PyResult<RaggedBufferF32> {
        Ok(RaggedBufferF32(lhs.0.add(&rhs.0)?))
    }
}

#[pyproto]
impl<'p> PyMappingProtocol for RaggedBufferF32 {
    fn __getitem__(&self, index: IndicesOrInt<'p>) -> PyResult<RaggedBufferF32> {
        match index {
            IndicesOrInt::Indices(indices) => Ok(RaggedBufferF32(self.0.swizzle(indices)?)),
            IndicesOrInt::Int(i) => Ok(RaggedBufferF32(self.0.get(i))),
        }
    }
}
