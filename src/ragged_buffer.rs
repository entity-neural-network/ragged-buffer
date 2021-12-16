use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use std::fmt::{Display, Write};
use std::ops::{Add, Mul, Range};

use pyo3::prelude::*;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct RaggedBuffer<T> {
    data: Vec<T>,
    subarrays: Vec<Range<usize>>,
    features: usize,
    items: usize,
}

pub trait BinOp<T> {
    fn op(lhs: T, rhs: T) -> T;
}

pub struct BinOpAdd;

impl<T: Add<T, Output = T>> BinOp<T> for BinOpAdd {
    #[inline]
    fn op(lhs: T, rhs: T) -> T {
        lhs + rhs
    }
}

pub struct BinOpMul;

impl<T: Mul<T, Output = T>> BinOp<T> for BinOpMul {
    #[inline]
    fn op(lhs: T, rhs: T) -> T {
        lhs * rhs
    }
}

impl<T: numpy::Element + Copy + Display + std::fmt::Debug> RaggedBuffer<T> {
    pub fn new(features: usize) -> Self {
        RaggedBuffer {
            data: Vec::new(),
            subarrays: Vec::new(),
            features,
            items: 0,
        }
    }

    pub fn from_array(data: PyReadonlyArray3<T>) -> Self {
        let data = data.as_array();
        let features = data.shape()[2];
        RaggedBuffer {
            data: data.iter().cloned().collect(),
            subarrays: (0..data.shape()[0])
                .map(|i| i * data.shape()[1]..(i + 1) * data.shape()[1])
                .collect(),
            features,
            items: data.shape()[0] * data.shape()[1],
        }
    }

    pub fn from_flattened(
        data: PyReadonlyArray2<T>,
        lengths: PyReadonlyArray1<i64>,
    ) -> PyResult<Self> {
        let data = data.as_array();
        let lenghts = lengths.as_array();
        let features = data.shape()[1];
        let mut subarrays = Vec::new();
        let mut item = 0;
        for len in lenghts.iter().cloned() {
            subarrays.push(item..(item + len as usize));
            item += len as usize;
        }
        if item != data.shape()[0] {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Lengths array specifies {} items, but data array has {} items",
                item,
                data.shape()[0]
            )))
        } else {
            Ok(RaggedBuffer {
                data: data.iter().cloned().collect(),
                subarrays,
                features,
                items: item,
            })
        }
    }

    pub fn extend(&mut self, other: &RaggedBuffer<T>) -> PyResult<()> {
        if self.features != other.features {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Features mismatch: {} != {}",
                self.features, other.features
            )));
        }
        let item = self.items;
        self.data.extend(other.data.iter());
        self.subarrays
            .extend(other.subarrays.iter().map(|r| r.start + item..r.end + item));
        self.items += other.items;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.subarrays.clear();
        self.items = 0;
    }

    pub fn as_array<'a>(
        &self,
        py: Python<'a>,
    ) -> PyResult<&'a numpy::PyArray<T, numpy::ndarray::Dim<[usize; 2]>>> {
        self.data
            .to_pyarray(py)
            .reshape((self.items, self.features))
    }

    pub fn push(&mut self, x: &PyReadonlyArray2<T>) -> PyResult<()> {
        let data = x.as_array();
        if data.dim().1 != self.features {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Features mismatch: {} != {}",
                self.features,
                data.dim().1
            )));
        }
        self.subarrays.push(self.items..(self.items + data.dim().0));
        match data.as_slice() {
            Some(slice) => self.data.extend_from_slice(slice),
            None => {
                for x in data.iter() {
                    self.data.push(*x);
                }
            }
        }
        self.items += data.dim().0;
        Ok(())
    }

    pub fn push_empty(&mut self) {
        self.subarrays.push(self.items..self.items);
    }

    pub fn swizzle(&self, indices: PyReadonlyArray1<i64>) -> PyResult<RaggedBuffer<T>> {
        let indices = indices.as_array();
        let indices = indices.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Indices must be a **contiguous** 1D array")
        })?;
        let mut subarrays = Vec::with_capacity(indices.len());
        let mut item = 0usize;
        for i in indices {
            let sublen = self.subarrays[*i as usize].end - self.subarrays[*i as usize].start;
            subarrays.push(item..(item + sublen));
            item += sublen;
        }
        let mut data = Vec::with_capacity(item * self.features);
        for i in indices {
            let Range { start, end } = self.subarrays[*i as usize];
            data.extend_from_slice(&self.data[start * self.features..end * self.features]);
        }
        Ok(RaggedBuffer {
            data,
            subarrays,
            features: self.features,
            items: item,
        })
    }

    pub fn get(&self, i: usize) -> RaggedBuffer<T> {
        let subarray = self.subarrays[i].clone();
        let Range { start, end } = subarray;
        RaggedBuffer {
            subarrays: vec![0..subarray.len()],
            data: self.data[start * self.features..end * self.features].to_vec(),
            features: self.features,
            items: subarray.len(),
        }
    }

    pub fn size0(&self) -> usize {
        self.subarrays.len()
    }

    pub fn lengths<'a>(
        &self,
        py: Python<'a>,
    ) -> &'a numpy::PyArray<i64, numpy::ndarray::Dim<[usize; 1]>> {
        self.subarrays
            .iter()
            .map(|r| (r.end - r.start) as i64)
            .collect::<Vec<_>>()
            .to_pyarray(py)
    }

    pub fn size1(&self, i: usize) -> PyResult<usize> {
        if i >= self.subarrays.len() {
            Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Index {} out of range",
                i
            )))
        } else {
            Ok(self.subarrays[i].end - self.subarrays[i].start)
        }
    }

    pub fn size2(&self) -> usize {
        self.features
    }

    pub fn __str__(&self) -> PyResult<String> {
        let mut array = String::new();
        array.push_str("RaggedBuffer([");
        array.push('\n');
        for range in &self.subarrays {
            let slice = range.start * self.features..range.end * self.features;
            if range.start == range.end {
                writeln!(array, "    [],").unwrap();
            } else if range.start + 1 == range.end {
                writeln!(array, "    [{:?}],", &self.data[slice]).unwrap();
            } else {
                writeln!(array, "    [").unwrap();
                for i in slice.clone() {
                    if i % self.features == 0 {
                        if i != slice.start {
                            writeln!(array, "],").unwrap();
                        }
                        write!(array, "        [").unwrap();
                    }
                    write!(array, "{}", self.data[i]).unwrap();
                    if i % self.features != self.features - 1 {
                        write!(array, ", ").unwrap();
                    }
                }
                writeln!(array, "],").unwrap();
                writeln!(array, "    ],").unwrap();
            }
        }
        write!(
            array,
            "], '{} * var * {} * {})",
            self.subarrays.len(),
            self.features,
            std::any::type_name::<T>(),
        )
        .unwrap();

        Ok(array)
    }

    pub fn binop<Op: BinOp<T>>(&self, rhs: &RaggedBuffer<T>) -> PyResult<RaggedBuffer<T>> {
        if self.features == rhs.features && self.subarrays == rhs.subarrays {
            let mut data = Vec::with_capacity(self.data.len());
            for i in 0..self.data.len() {
                data.push(Op::op(self.data[i], rhs.data[i]));
            }
            Ok(RaggedBuffer {
                data,
                subarrays: self.subarrays.clone(),
                features: self.features,
                items: self.items,
            })
        } else if self.features == rhs.features
            && self.subarrays.len() == rhs.subarrays.len()
            && rhs.subarrays.iter().all(|r| r.end - r.start == 1)
        {
            let mut data = Vec::with_capacity(self.data.len());
            for (subarray, rhs_subarray) in self.subarrays.iter().zip(rhs.subarrays.iter()) {
                for item in subarray.clone() {
                    let lhs_offset = item * self.features;
                    let rhs_offset = rhs_subarray.start * self.features;
                    for i in 0..self.features {
                        data.push(Op::op(self.data[lhs_offset + i], rhs.data[rhs_offset + i]));
                    }
                }
            }
            Ok(RaggedBuffer {
                data,
                subarrays: self.subarrays.clone(),
                features: self.features,
                items: self.items,
            })
        } else if self.features == rhs.features
            && self.subarrays.len() == rhs.subarrays.len()
            && self.subarrays.iter().all(|r| r.end - r.start == 1)
        {
            rhs.binop::<Op>(self)
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Dimensions mismatch: ({}, {:?}, {}) != ({}, {:?}, {})",
                self.size0(),
                self.subarrays
                    .iter()
                    .map(|r| r.end - r.start)
                    .collect::<Vec<_>>(),
                self.size2(),
                rhs.size0(),
                rhs.subarrays
                    .iter()
                    .map(|r| r.end - r.start)
                    .collect::<Vec<_>>(),
                rhs.size2(),
            )))
        }
    }

    pub fn op_scalar<Op: BinOp<T>>(&self, scalar: T) -> RaggedBuffer<T> {
        RaggedBuffer {
            data: self.data.iter().map(|x| Op::op(*x, scalar)).collect(),
            subarrays: self.subarrays.clone(),
            features: self.features,
            items: self.items,
        }
    }

    pub fn indices(&self, dim: usize) -> PyResult<RaggedBuffer<i64>> {
        match dim {
            0 => {
                let mut indices = Vec::with_capacity(self.items);
                for (index, subarray) in self.subarrays.iter().enumerate() {
                    for _ in subarray.clone() {
                        indices.push(index as i64);
                    }
                }
                Ok(RaggedBuffer {
                    subarrays: self.subarrays.clone(),
                    data: indices,
                    features: 1,
                    items: self.items,
                })
            }
            1 => {
                let mut indices = Vec::with_capacity(self.items);
                for subarray in &self.subarrays {
                    for (i, _) in subarray.clone().enumerate() {
                        indices.push(i as i64);
                    }
                }
                Ok(RaggedBuffer {
                    subarrays: self.subarrays.clone(),
                    data: indices,
                    features: 1,
                    items: self.items,
                })
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid dimension {}",
                dim
            ))),
        }
    }

    pub fn flat_indices(&self) -> PyResult<RaggedBuffer<i64>> {
        Ok(RaggedBuffer {
            subarrays: self.subarrays.clone(),
            data: (0..self.items).map(|i| i as i64).collect(),
            features: 1,
            items: self.items,
        })
    }

    pub fn cat(buffers: &[&RaggedBuffer<T>], dim: usize) -> PyResult<RaggedBuffer<T>> {
        match dim {
            0 => {
                if buffers.iter().any(|b| b.features != buffers[0].features) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "All buffers must have the same number of features, but found {}",
                        buffers
                            .iter()
                            .map(|b| b.features.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )));
                }
                let mut data = Vec::with_capacity(buffers.iter().map(|b| b.data.len()).sum());
                for buffer in buffers {
                    data.extend_from_slice(&buffer.data);
                }
                let mut subarrays =
                    Vec::with_capacity(buffers.iter().map(|b| b.subarrays.len()).sum());
                let mut item = 0;
                for buffer in buffers {
                    subarrays.extend_from_slice(
                        &buffer
                            .subarrays
                            .iter()
                            .map(|r| {
                                let start = r.start + item;
                                let end = r.end + item;
                                start..end
                            })
                            .collect::<Vec<_>>(),
                    );
                    item += buffer.items;
                }
                Ok(RaggedBuffer {
                    data,
                    subarrays,
                    features: buffers[0].features,
                    items: item,
                })
            }
            1 => {
                if buffers
                    .iter()
                    .any(|b| b.subarrays.len() != buffers[0].subarrays.len())
                {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "All buffers must have the same number of subarrays, but found {}",
                        buffers
                            .iter()
                            .map(|b| b.subarrays.len().to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )));
                }
                if buffers.iter().any(|b| b.features != buffers[0].features) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "All buffers must have the same number of features, but found {}",
                        buffers
                            .iter()
                            .map(|b| b.features.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )));
                }
                let mut data = Vec::with_capacity(buffers.iter().map(|b| b.data.len()).sum());
                let mut subarrays =
                    Vec::with_capacity(buffers.iter().map(|b| b.subarrays.len()).sum());
                let mut item = 0;
                let mut last_item = 0;
                for i in 0..buffers[0].subarrays.len() {
                    for buffer in buffers {
                        let Range { start, end } = &buffer.subarrays[i];
                        data.extend_from_slice(
                            &buffer.data[start * buffer.features..end * buffer.features],
                        );
                        item += end - start;
                    }
                    subarrays.push(Range {
                        start: last_item,
                        end: item,
                    });
                    last_item = item;
                }
                Ok(RaggedBuffer {
                    data,
                    subarrays,
                    features: buffers[0].features,
                    items: item,
                })
            }
            2 => Err(pyo3::exceptions::PyValueError::new_err(
                "Concatenation along dimension 2 not yet implemented",
            )),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid dimension {}, RaggedBuffer only has 3 dimensions",
                dim
            ))),
        }
    }
}
