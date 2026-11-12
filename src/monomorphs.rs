#![allow(clippy::format_push_string)] // Caused by FromPyObject macro
use numpy::PyReadonlyArray1;
use pyo3::types::PySlice;
use pyo3::{Bound, FromPyObject, Py, PyResult};

mod bool;
mod f32;
mod i64;

pub use self::bool::RaggedBufferBool;
pub use self::f32::RaggedBufferF32;
pub use self::i64::RaggedBufferI64;

#[derive(FromPyObject)]
pub enum Index<'py> {
    PermutationNP(PyReadonlyArray1<'py, i64>),
    Permutation(Vec<usize>),
    Int(usize),
    Slice(Py<PySlice>),
}

impl<'py> std::fmt::Debug for Index<'py> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PermutationNP(arg0) => f
                .debug_tuple("PermutationNP")
                .field(&arg0.as_array().to_vec())
                .finish(),
            Self::Permutation(arg0) => f.debug_tuple("Permutation").field(arg0).finish(),
            Self::Int(arg0) => f.debug_tuple("Int").field(arg0).finish(),
            Self::Slice(arg0) => f.debug_tuple("Slice").field(arg0).finish(),
        }
    }
}

#[derive(FromPyObject, Debug)]
pub enum MultiIndex<'py> {
    Index1(Index<'py>),
    Index2((Index<'py>, Index<'py>)),
    Index3((Index<'py>, Index<'py>, Index<'py>)),
}

type PyArray<'py, T, D> = Bound<'py, numpy::PyArray<T, numpy::ndarray::Dim<D>>>;
type PadpackResult<'py> = PyResult<
    Option<(
        PyArray<'py, i64, [usize; 2]>,
        PyArray<'py, f32, [usize; 2]>,
        PyArray<'py, i64, [usize; 1]>,
    )>,
>;
