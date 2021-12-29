use numpy::PyReadonlyArray1;
use pyo3::{FromPyObject, PyResult};

mod bool;
mod f32;
mod i64;

pub use self::bool::RaggedBufferBool;
pub use self::f32::RaggedBufferF32;
pub use self::i64::RaggedBufferI64;

#[derive(FromPyObject)]
pub enum IndicesOrInt<'a> {
    Indices(PyReadonlyArray1<'a, i64>),
    Int(usize),
}

type PyArray<'a, T, D> = &'a numpy::PyArray<T, numpy::ndarray::Dim<D>>;
type PadpackResult<'a> = PyResult<
    Option<(
        PyArray<'a, i64, [usize; 2]>,
        PyArray<'a, f32, [usize; 2]>,
        PyArray<'a, i64, [usize; 1]>,
    )>,
>;
