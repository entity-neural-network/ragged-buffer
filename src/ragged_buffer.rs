use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArrayDyn, ToPyArray};
use std::fmt::{Display, Write};
use std::ops::{Add, Range};

use pyo3::prelude::*;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct RaggedBuffer<T> {
    data: Vec<T>,
    subarrays: Vec<Range<usize>>,
    features: usize,
}

impl<T: numpy::Element + Copy + Display + Add<Output = T> + std::fmt::Debug> RaggedBuffer<T> {
    pub fn new(features: usize) -> Self {
        RaggedBuffer {
            data: Vec::new(),
            subarrays: Vec::new(),
            features,
        }
    }

    pub fn from_array(data: PyReadonlyArray3<T>) -> Self {
        let data = data.as_array();
        let features = data.shape()[2];
        RaggedBuffer {
            data: data.iter().cloned().collect(),
            subarrays: (0..data.shape()[0])
                .map(|i| i * features * data.shape()[1]..(i + 1) * features * data.shape()[1])
                .collect(),
            features,
        }
    }

    pub fn from_flattened(data: PyReadonlyArray2<T>, lenghts: PyReadonlyArray1<i64>) -> Self {
        let data = data.as_array();
        let lenghts = lenghts.as_array();
        let features = data.shape()[1];
        let mut subarrays = Vec::new();
        let mut data_index = 0;
        for len in lenghts.iter().cloned() {
            let l = len as usize * features;
            subarrays.push(data_index..(data_index + l));
            data_index += l;
        }
        RaggedBuffer {
            data: data.iter().cloned().collect(),
            subarrays,
            features,
        }
    }

    pub fn extend(&mut self, other: &RaggedBuffer<T>) -> PyResult<()> {
        if self.features != other.features {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Features mismatch: {} != {}",
                self.features, other.features
            )));
        }
        let len = self.data.len();
        self.data.extend(other.data.iter());
        self.subarrays
            .extend(other.subarrays.iter().map(|r| r.start + len..r.end + len));
        Ok(())
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.subarrays.clear();
    }

    pub fn as_array<'a>(
        &self,
        py: Python<'a>,
    ) -> &'a numpy::PyArray<T, numpy::ndarray::Dim<[usize; 2]>> {
        self.data
            .to_pyarray(py)
            .reshape((self.data.len() / self.features, self.features))
            .unwrap()
    }

    pub fn push(&mut self, x: PyReadonlyArrayDyn<T>) {
        let data = x.as_array();
        assert!(data.len() % self.features == 0);
        let start = self.data.len();
        let len = data.len();
        self.subarrays.push(start..(start + len));
        match data.as_slice() {
            Some(slice) => self.data.extend_from_slice(slice),
            None => {
                for x in data.iter() {
                    self.data.push(*x);
                }
            }
        }
    }

    pub fn swizzle(&self, indices: PyReadonlyArray1<i64>) -> PyResult<RaggedBuffer<T>> {
        let indices = indices.as_array();
        let indices = indices.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Indices must be a **contiguous** 1D array")
        })?;
        let mut subarrays = Vec::with_capacity(indices.len());
        let mut len = 0usize;
        for i in indices {
            let sublen = self.subarrays[*i as usize].end - self.subarrays[*i as usize].start;
            subarrays.push(len..(len + sublen));
            len += sublen;
        }
        let mut data = Vec::with_capacity(len);
        for i in indices {
            data.extend_from_slice(&self.data[self.subarrays[*i as usize].clone()]);
        }
        Ok(RaggedBuffer {
            data,
            subarrays,
            features: self.features,
        })
    }

    pub fn get(&self, i: usize) -> RaggedBuffer<T> {
        let subarray = self.subarrays[i].clone();
        RaggedBuffer {
            subarrays: vec![0..subarray.len()],
            data: self.data[subarray].to_vec(),
            features: self.features,
        }
    }

    pub fn size0(&self) -> usize {
        self.subarrays.len()
    }

    pub fn lengths<'a>(
        &self,
        py: Python<'a>,
    ) -> &'a numpy::PyArray<usize, numpy::ndarray::Dim<[usize; 1]>> {
        self.subarrays
            .iter()
            .map(|r| (r.end - r.start) / self.features)
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
            Ok((self.subarrays[i].end - self.subarrays[i].start) / self.features)
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
            if range.start == range.end {
                writeln!(array, "    [],").unwrap();
            } else if range.end - range.start == self.features {
                writeln!(array, "    [{:?}],", &self.data[range.clone()]).unwrap();
            } else {
                writeln!(array, "    [").unwrap();
                for i in range.clone() {
                    if i % self.features == 0 {
                        if i != range.start {
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

    pub fn add(&self, rhs: &RaggedBuffer<T>) -> PyResult<RaggedBuffer<T>> {
        if self.features == rhs.features && self.subarrays == rhs.subarrays {
            let mut data = Vec::with_capacity(self.data.len());
            for i in 0..self.data.len() {
                data.push(self.data[i] + rhs.data[i]);
            }
            Ok(RaggedBuffer {
                data,
                subarrays: self.subarrays.clone(),
                features: self.features,
            })
        } else if self.features == rhs.features
            && self.subarrays.len() == rhs.subarrays.len()
            && self
                .subarrays
                .iter()
                .all(|r| (r.end - r.start) == self.features)
        {
            let mut data = Vec::with_capacity(self.data.len());
            for (subarray, rhs_subarray) in self.subarrays.iter().zip(rhs.subarrays.iter()) {
                let mut i = subarray.start;
                while i < subarray.end {
                    for j in rhs_subarray.clone() {
                        data.push(self.data[i] + rhs.data[j]);
                    }
                    i += self.features;
                }
            }
            Ok(RaggedBuffer {
                data,
                subarrays: self.subarrays.clone(),
                features: self.features,
            })
        } else if self.features == rhs.features
            && rhs
                .subarrays
                .iter()
                .all(|r| (r.end - r.start) == self.features)
        {
            rhs.add(self)
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Dimensions mismatch: ({}, {:?}, {}) != ({}, {:?}, {})",
                self.size0(),
                self.subarrays
                    .iter()
                    .map(|r| (r.end - r.start) / self.features)
                    .collect::<Vec<_>>(),
                self.size2(),
                rhs.size0(),
                rhs.subarrays
                    .iter()
                    .map(|r| (r.end - r.start) / self.features)
                    .collect::<Vec<_>>(),
                rhs.size2(),
            )))
        }
    }

    pub fn indices(&self, dim: usize) -> PyResult<RaggedBuffer<i64>> {
        match dim {
            0 => {
                let mut indices = Vec::with_capacity(self.data.len() / self.features);
                for (index, subarray) in self.subarrays.iter().enumerate() {
                    for _ in 0..(subarray.end - subarray.start) / self.features {
                        indices.push(index as i64);
                    }
                }
                Ok(RaggedBuffer {
                    subarrays: self
                        .subarrays
                        .iter()
                        .map(|r| r.start / self.features..r.end / self.features)
                        .collect(),
                    data: indices,
                    features: 1,
                })
            }
            1 => Err(pyo3::exceptions::PyValueError::new_err(
                "Sequence indices not yet implemented",
            )),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid dimension {}",
                dim
            ))),
        }
    }

    pub fn flat_indices(&self) -> PyResult<RaggedBuffer<i64>> {
        Ok(RaggedBuffer {
            subarrays: self
                .subarrays
                .iter()
                .map(|r| r.start / self.features..r.end / self.features)
                .collect(),
            data: (0..self.data.len() / self.features)
                .map(|i| i as i64)
                .collect(),
            features: 1,
        })
    }
}
