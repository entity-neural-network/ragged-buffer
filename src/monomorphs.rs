use numpy::PyReadonlyArray1;
use pyo3::types::PySlice;
use pyo3::{FromPyObject, Py, PyResult};

mod bool;
mod f32;
mod i64;

pub use self::bool::RaggedBufferBool;
pub use self::f32::RaggedBufferF32;
pub use self::i64::RaggedBufferI64;

#[derive(FromPyObject)]
pub enum Index<'a> {
    PermutationNP(PyReadonlyArray1<'a, i64>),
    Permutation(Vec<usize>),
    Int(usize),
    Slice(Py<PySlice>),
}

impl<'a> std::fmt::Debug for Index<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PermutationNP(arg0) => f
                .debug_tuple("PermutationNP")
                .field(&arg0.to_vec().unwrap())
                .finish(),
            Self::Permutation(arg0) => f.debug_tuple("Permutation").field(arg0).finish(),
            Self::Int(arg0) => f.debug_tuple("Int").field(arg0).finish(),
            Self::Slice(arg0) => f.debug_tuple("Slice").field(arg0).finish(),
        }
    }
}

#[derive(FromPyObject, Debug)]
pub enum MultiIndex<'a> {
    Index1(Index<'a>),
    Index2((Index<'a>, Index<'a>)),
    Index3((Index<'a>, Index<'a>, Index<'a>)),
}

type PyArray<'a, T, D> = &'a numpy::PyArray<T, numpy::ndarray::Dim<D>>;
type PadpackResult<'a> = PyResult<
    Option<(
        PyArray<'a, i64, [usize; 2]>,
        PyArray<'a, f32, [usize; 2]>,
        PyArray<'a, i64, [usize; 1]>,
    )>,
>;
