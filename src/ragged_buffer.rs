use numpy::PyReadonlyArrayDyn;
use std::fmt::{Display, Write};
use std::ops::Range;

use numpy::ndarray::ArrayViewD;
use pyo3::prelude::*;

pub struct RaggedBuffer<T> {
    data: Vec<T>,
    subarrays: Vec<Range<u32>>,
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

    pub fn push(&mut self, x: PyReadonlyArrayDyn<T>) {
        self._push(x.as_array());
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
                    if i as usize % self.features == 0 {
                        if i != range.start {
                            writeln!(array, "],").unwrap();
                        }
                        write!(array, "        [").unwrap();
                    }
                    write!(array, "{}", self.data[i as usize]).unwrap();
                    if i as usize % self.features != self.features - 1 {
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
}

impl<T: numpy::Element + Copy> RaggedBuffer<T> {
    fn _push(&mut self, data: ArrayViewD<'_, T>) {
        assert!(data.len() % self.features == 0);
        let start = self.data.len() as u32;
        let len = data.len() as u32;
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
            subarrays.push(len as u32..(len as u32 + sublen));
            len += sublen as usize;
        }
        let mut data = Vec::with_capacity(len);
        for i in indices {
            let Range { start, end } = self.subarrays[*i];
            data.extend_from_slice(&self.data[start as usize..end as usize]);
        }
        RaggedBuffer {
            data,
            subarrays,
            features: self.features,
        }
    }
}
