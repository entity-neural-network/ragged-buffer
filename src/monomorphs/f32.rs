use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArrayDyn, ToPyArray};
use pyo3::basic::CompareOp;
use pyo3::types::PyType;
use pyo3::{prelude::*, PyMappingProtocol, PyObjectProtocol};

use crate::monomorphs::RaggedBufferI64;
use crate::ragged_buffer_view::RaggedBufferView;

use super::{Index, MultiIndex, PadpackResult};

#[pyclass]
#[derive(Clone)]
pub struct RaggedBufferF32(pub RaggedBufferView<f32>);

#[pymethods]
impl RaggedBufferF32 {
    #[new]
    pub fn new(features: usize) -> Self {
        RaggedBufferF32(RaggedBufferView::new(features))
    }
    #[classmethod]
    fn from_array(_cls: &PyType, array: PyReadonlyArray3<f32>) -> Self {
        RaggedBufferF32(RaggedBufferView::from_array(array))
    }
    #[classmethod]
    fn from_flattened(
        _cls: &PyType,
        flattened: PyReadonlyArray2<f32>,
        lengths: PyReadonlyArray1<i64>,
    ) -> PyResult<Self> {
        Ok(RaggedBufferF32(RaggedBufferView::from_flattened(
            flattened, lengths,
        )?))
    }
    fn push(&mut self, items: PyReadonlyArrayDyn<f32>) -> PyResult<()> {
        if items.ndim() == 1 && items.len() == 0 {
            self.0.push_empty()
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
    fn push_empty(&mut self) -> PyResult<()> {
        self.0.push_empty()
    }

    fn clear(&mut self) -> PyResult<()> {
        self.0.clear()
    }

    fn as_array<'a>(
        &self,
        py: Python<'a>,
    ) -> PyResult<&'a numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 2]>>> {
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
            None => self.0.lengths(py).map(|ok| ok.into_py(py)),
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
    fn cat(_cls: &PyType, buffers: Vec<PyRef<RaggedBufferF32>>, dim: usize) -> PyResult<Self> {
        Ok(RaggedBufferF32(RaggedBufferView::cat(
            &buffers.iter().map(|b| &b.0).collect::<Vec<_>>(),
            dim,
        )?))
    }
    #[allow(clippy::type_complexity)]
    fn padpack<'a>(&self, py: Python<'a>) -> PadpackResult<'a> {
        match self.0.padpack()? {
            Some((padbpack_index, padpack_batch, padpack_inverse_index, dims)) => Ok(Some((
                padbpack_index.to_pyarray(py).reshape(dims)?,
                padpack_batch.to_pyarray(py).reshape(dims)?,
                padpack_inverse_index
                    .to_pyarray(py)
                    .reshape(self.0.len()?)?,
            ))),
            _ => Ok(None),
        }
    }
    fn items(&self) -> PyResult<usize> {
        self.0.items()
    }
    fn clone(&self) -> Self {
        RaggedBufferF32(self.0.deepclone())
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

#[derive(FromPyObject)]
pub enum RaggedBufferF32OrF32 {
    RB(RaggedBufferF32),
    Scalar(f32),
}

// TODO: pass by reference?
#[cfg(all())]
#[pyproto]
impl pyo3::PyNumberProtocol for RaggedBufferF32 {
    fn __add__(lhs: RaggedBufferF32, rhs: RaggedBufferF32OrF32) -> PyResult<RaggedBufferF32> {
        match rhs {
            RaggedBufferF32OrF32::RB(rhs) => Ok(RaggedBufferF32(
                lhs.0.binop::<crate::ragged_buffer::BinOpAdd>(&rhs.0)?,
            )),
            RaggedBufferF32OrF32::Scalar(rhs) => Ok(RaggedBufferF32(
                lhs.0.op_scalar::<crate::ragged_buffer::BinOpAdd>(rhs)?,
            )),
        }
    }
    fn __mul__(lhs: RaggedBufferF32, rhs: RaggedBufferF32OrF32) -> PyResult<RaggedBufferF32> {
        match rhs {
            RaggedBufferF32OrF32::RB(rhs) => Ok(RaggedBufferF32(
                lhs.0.binop::<crate::ragged_buffer::BinOpMul>(&rhs.0)?,
            )),
            RaggedBufferF32OrF32::Scalar(rhs) => Ok(RaggedBufferF32(
                lhs.0.op_scalar::<crate::ragged_buffer::BinOpMul>(rhs)?,
            )),
        }
    }

    fn __isub__(&mut self, rhs: RaggedBufferF32) -> PyResult<()> {
        self.0.binop_mut::<crate::ragged_buffer::BinOpSub>(&rhs.0)
    }
}

#[pyproto]
impl<'p> PyMappingProtocol for RaggedBufferF32 {
    fn __getitem__(&self, index: MultiIndex<'p>) -> PyResult<RaggedBufferF32> {
        match index {
            MultiIndex::Index1(index) => match index {
                Index::PermutationNP(indices) => Ok(RaggedBufferF32(self.0.swizzle(indices)?)),
                Index::Permutation(_indices) => panic!("oh no"), //Ok(RaggedBufferF32(self.0.swizzle(indices)?)),
                Index::Int(i) => Ok(RaggedBufferF32(self.0.get_sequence(i)?)),
                Index::Slice(slice) => panic!("{:?}", slice),
            },
            MultiIndex::Index3((i0, i1, i2)) => Ok(RaggedBufferF32(Python::with_gil(|py| {
                self.0.get_slice(py, i0, i1, i2)
            })?)),
            x => panic!("{:?}", x),
        }
    }
    fn __len__(&self) -> PyResult<usize> {
        self.0.len()
    }
}
