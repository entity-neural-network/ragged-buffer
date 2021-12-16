use numpy::PyReadonlyArray1;
use pyo3::FromPyObject;

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
