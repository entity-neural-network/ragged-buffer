use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArrayDyn};
use pyo3::basic::CompareOp;
use pyo3::types::PyType;
use pyo3::{prelude::*, PyMappingProtocol, PyObjectProtocol};

use crate::ragged_buffer::RaggedBuffer;

use super::IndicesOrInt;

#[pyclass]
#[derive(Clone)]
pub struct RaggedBufferI64(pub RaggedBuffer<i64>);

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
    #[classmethod]
    fn from_flattened(
        _cls: &PyType,
        flattened: PyReadonlyArray2<i64>,
        lengths: PyReadonlyArray1<i64>,
    ) -> PyResult<Self> {
        Ok(RaggedBufferI64(RaggedBuffer::from_flattened(
            flattened, lengths,
        )?))
    }
    fn push(&mut self, items: PyReadonlyArrayDyn<i64>) -> PyResult<()> {
        if items.ndim() == 1 && items.len() == 0 {
            self.0.push_empty();
            Ok(())
        } else if items.ndim() == 2 {
            self.0.push(
                &items
                    .reshape((items.shape()[0], items.shape()[1]))?
                    .readonly(),
            )
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "Expected 2 dimensional array",
            ))
        }
    }
    fn push_empty(&mut self) {
        self.0.push_empty();
    }

    fn clear(&mut self) {
        self.0.clear();
    }

    fn as_array<'a>(
        &self,
        py: Python<'a>,
    ) -> PyResult<&'a numpy::PyArray<i64, numpy::ndarray::Dim<[usize; 2]>>> {
        self.0.as_array(py)
    }

    fn extend(&mut self, other: &RaggedBufferI64) -> PyResult<()> {
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
    fn indices(&self, dim: usize) -> PyResult<RaggedBufferI64> {
        Ok(RaggedBufferI64(self.0.indices(dim)?))
    }
    fn flat_indices(&self) -> PyResult<RaggedBufferI64> {
        Ok(RaggedBufferI64(self.0.flat_indices()?))
    }
    #[classmethod]
    fn cat(_cls: &PyType, buffers: Vec<PyRef<RaggedBufferI64>>, dim: usize) -> PyResult<Self> {
        Ok(RaggedBufferI64(RaggedBuffer::cat(
            &buffers.iter().map(|b| &b.0).collect::<Vec<_>>(),
            dim,
        )?))
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

    fn __richcmp__(&self, other: RaggedBufferI64, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.0 == other.0),
            CompareOp::Ne => Ok(self.0 != other.0),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Only == and != are supported",
            )),
        }
    }
}

#[derive(FromPyObject)]
pub enum RaggedBufferI64OrI64 {
    RB(RaggedBufferI64),
    Scalar(i64),
}

#[cfg(all())]
#[pyproto]
impl pyo3::PyNumberProtocol for RaggedBufferI64 {
    fn __add__(lhs: RaggedBufferI64, rhs: RaggedBufferI64OrI64) -> PyResult<RaggedBufferI64> {
        match rhs {
            RaggedBufferI64OrI64::RB(rhs) => Ok(RaggedBufferI64(
                lhs.0.binop::<crate::ragged_buffer::BinOpAdd>(&rhs.0)?,
            )),
            RaggedBufferI64OrI64::Scalar(rhs) => Ok(RaggedBufferI64(
                lhs.0.op_scalar::<crate::ragged_buffer::BinOpAdd>(rhs),
            )),
        }
    }
    fn __mul__(lhs: RaggedBufferI64, rhs: RaggedBufferI64OrI64) -> PyResult<RaggedBufferI64> {
        match rhs {
            RaggedBufferI64OrI64::RB(rhs) => Ok(RaggedBufferI64(
                lhs.0.binop::<crate::ragged_buffer::BinOpMul>(&rhs.0)?,
            )),
            RaggedBufferI64OrI64::Scalar(rhs) => Ok(RaggedBufferI64(
                lhs.0.op_scalar::<crate::ragged_buffer::BinOpMul>(rhs),
            )),
        }
    }
}

#[pyproto]
impl<'p> PyMappingProtocol for RaggedBufferI64 {
    fn __getitem__(&self, index: IndicesOrInt<'p>) -> PyResult<RaggedBufferI64> {
        match index {
            IndicesOrInt::Indices(indices) => Ok(RaggedBufferI64(self.0.swizzle(indices)?)),
            IndicesOrInt::Int(i) => Ok(RaggedBufferI64(self.0.get(i))),
        }
    }
}
