use numpy::{PyReadonlyArrayDyn, ToPyArray};
use std::fmt::{Display, Write};
use std::ops::Range;

use pyo3::prelude::*;

pub struct RaggedBuffer<T> {
    data: Vec<T>,
    subarrays: Vec<Range<usize>>,
    features: usize,
}

impl<T: numpy::Element + Copy + Display> RaggedBuffer<T> {
    pub fn new(features: usize) -> RaggedBuffer<T> {
        RaggedBuffer {
            data: Vec::new(),
            subarrays: Vec::new(),
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

    pub fn __str__(&self) -> PyResult<String> {
        let mut array = String::new();
        array.push_str("RaggedBuffer([");
        array.push('\n');
        for range in &self.subarrays {
            if range.start == range.end {
                writeln!(array, "    [],").unwrap();
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
            "], '{} * var * {} * f32)",
            self.subarrays.len(),
            self.features
        )
        .unwrap();

        Ok(array)
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

    pub fn multi_index(&self, indices: &[usize]) -> RaggedBuffer<T> {
        let mut subarrays = Vec::with_capacity(indices.len());
        let mut len = 0usize;
        for i in indices {
            let sublen = self.subarrays[*i].end - self.subarrays[*i].start;
            subarrays.push(len..(len + sublen));
            len += sublen;
        }
        let mut data = Vec::with_capacity(len);
        for i in indices {
            data.extend_from_slice(&self.data[self.subarrays[*i].clone()]);
        }
        RaggedBuffer {
            data,
            subarrays,
            features: self.features,
        }
    }
}
