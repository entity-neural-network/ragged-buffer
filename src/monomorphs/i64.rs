use numpy::{
    PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyReadonlyArrayDyn, PyUntypedArrayMethods, ToPyArray,
};
use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::ragged_buffer_view::RaggedBufferView;

use super::{Index, MultiIndex, PadpackResult};

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct RaggedBufferI64(pub RaggedBufferView<i64>);

#[pymethods]
impl RaggedBufferI64 {
    #[new]
    pub fn new(features: usize) -> Self {
        RaggedBufferI64(RaggedBufferView::new(features))
    }
    #[classmethod]
    fn from_array(_cls: &Bound<'_, PyType>, array: PyReadonlyArray3<'_, i64>) -> Self {
        RaggedBufferI64(RaggedBufferView::from_array(array))
    }
    #[classmethod]
    fn from_flattened(
        _cls: &Bound<'_, PyType>,
        flattened: PyReadonlyArray2<'_, i64>,
        lengths: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<Self> {
        Ok(RaggedBufferI64(RaggedBufferView::from_flattened(
            flattened, lengths,
        )?))
    }
    fn push(&mut self, items: PyReadonlyArrayDyn<'_, i64>) -> PyResult<()> {
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

    fn as_array<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        self.0.as_array(py)
    }

    fn extend(&mut self, other: PyRef<'_, RaggedBufferI64>) -> PyResult<()> {
        self.0.extend(&other.0)
    }
    fn size0(&self) -> usize {
        self.0.size0()
    }
    #[pyo3(signature = (i=None))]
    fn size1(&mut self, py: Python<'_>, i: Option<usize>) -> PyResult<Py<PyAny>> {
        match i {
            Some(i) => Ok(self.0.size1(i)?.into_pyobject(py)?.into_any().unbind()),
            None => Ok(self.0.lengths(py)?.into_any().unbind()),
        }
    }
    fn size2(&self) -> usize {
        self.0.size2()
    }
    fn indices(&mut self, dim: usize) -> PyResult<RaggedBufferI64> {
        Ok(RaggedBufferI64(self.0.indices(dim)?))
    }
    fn flat_indices(&mut self) -> PyResult<RaggedBufferI64> {
        Ok(RaggedBufferI64(self.0.flat_indices()?))
    }
    #[classmethod]
    fn cat(
        _cls: &Bound<'_, PyType>,
        buffers: Vec<PyRef<'_, RaggedBufferI64>>,
        dim: usize,
    ) -> PyResult<Self> {
        Ok(RaggedBufferI64(RaggedBufferView::cat(
            &buffers.iter().map(|b| &b.0).collect::<Vec<_>>(),
            dim,
        )?))
    }
    #[allow(clippy::type_complexity)]
    fn padpack<'py>(&mut self, py: Python<'py>) -> PadpackResult<'py> {
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
    fn items(&mut self) -> PyResult<usize> {
        self.0.items()
    }
    fn clone(&self) -> Self {
        RaggedBufferI64(self.0.deepclone())
    }
    fn materialize(&self) -> Self {
        RaggedBufferI64(self.0.materialize())
    }
    fn __str__(&self) -> PyResult<String> {
        self.0.__str__()
    }

    fn __repr__(&self) -> PyResult<String> {
        self.0.__str__()
    }

    fn __richcmp__(&self, other: PyRef<'_, RaggedBufferI64>, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.0 == other.0),
            CompareOp::Ne => Ok(self.0 != other.0),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Only == and != are supported",
            )),
        }
    }

    // BEGIN ARITHMETIC (removed from bool.rs by monomorphize.sh)
    fn __add__(
        lhs: PyRef<'_, RaggedBufferI64>,
        rhs: RaggedBufferI64OrI64<'_>,
    ) -> PyResult<RaggedBufferI64> {
        match rhs {
            RaggedBufferI64OrI64::RB(rhs) => Ok(RaggedBufferI64(
                lhs.0.binop::<crate::ragged_buffer::BinOpAdd>(&rhs.0)?,
            )),
            RaggedBufferI64OrI64::Scalar(rhs) => Ok(RaggedBufferI64(
                lhs.0.op_scalar::<crate::ragged_buffer::BinOpAdd>(rhs)?,
            )),
        }
    }

    fn __mul__(
        lhs: PyRef<'_, RaggedBufferI64>,
        rhs: RaggedBufferI64OrI64<'_>,
    ) -> PyResult<RaggedBufferI64> {
        match rhs {
            RaggedBufferI64OrI64::RB(rhs) => Ok(RaggedBufferI64(
                lhs.0.binop::<crate::ragged_buffer::BinOpMul>(&rhs.0)?,
            )),
            RaggedBufferI64OrI64::Scalar(rhs) => Ok(RaggedBufferI64(
                lhs.0.op_scalar::<crate::ragged_buffer::BinOpMul>(rhs)?,
            )),
        }
    }

    fn __isub__(&mut self, rhs: PyRef<'_, RaggedBufferI64>) -> PyResult<()> {
        self.0.binop_mut::<crate::ragged_buffer::BinOpSub>(&rhs.0)
    }
    // END ARITHMETIC

    fn __getitem__(&self, py: Python<'_>, index: MultiIndex<'_>) -> PyResult<RaggedBufferI64> {
        match index {
            MultiIndex::Index1(index) => match index {
                Index::PermutationNP(indices) => Ok(RaggedBufferI64(self.0.swizzle(indices)?)),
                Index::Permutation(indices) => Ok(RaggedBufferI64(self.0.swizzle_usize(&indices)?)),
                Index::Int(i) => Ok(RaggedBufferI64(self.0.get_sequence(i)?)),
                Index::Slice(slice) => panic!("{:?}", slice),
            },
            MultiIndex::Index3((i0, i1, i2)) => {
                Ok(RaggedBufferI64(self.0.get_slice(py, i0, i1, i2)?))
            }
            x => panic!("{:?}", x),
        }
    }
    fn __len__(&self) -> PyResult<usize> {
        self.0.len()
    }
}

// BEGIN ARITHMETIC (removed from bool.rs by monomorphize.sh)
#[derive(FromPyObject)]
pub enum RaggedBufferI64OrI64<'py> {
    RB(PyRef<'py, RaggedBufferI64>),
    Scalar(i64),
}
// END ARITHMETIC
