use std::cmp::Ordering;
use std::collections::{binary_heap, BinaryHeap};
use std::fmt::{Display, Write};
use std::ops::{Add, Mul, Range, Sub};

use ndarray::{ArrayView1, ArrayView2, ArrayView3};

#[derive(Debug)]
pub enum Error {
    Generic(String),
}

impl Error {
    fn generic<S: Into<String>>(s: S) -> Self {
        Self::Generic(s.into())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct RaggedBuffer<T> {
    pub data: Vec<T>,
    // Each element of `subarrays` gives the start/end index of the items within that subarray (step size 1).
    // The start index of the data of an item is obtained by multiplying its index by `features`.
    pub subarrays: Vec<Range<usize>>,
    pub features: usize,
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

pub struct BinOpSub;

impl<T: Sub<T, Output = T>> BinOp<T> for BinOpSub {
    #[inline]
    fn op(lhs: T, rhs: T) -> T {
        lhs - rhs
    }
}

pub struct BinOpMul;

impl<T: Mul<T, Output = T>> BinOp<T> for BinOpMul {
    #[inline]
    fn op(lhs: T, rhs: T) -> T {
        lhs * rhs
    }
}

impl<T: Copy + Display + std::fmt::Debug> RaggedBuffer<T> {
    pub fn new(features: usize) -> Self {
        RaggedBuffer {
            data: Vec::new(),
            subarrays: Vec::new(),
            features,
        }
    }

    pub fn from_array(data: ArrayView3<T>) -> Self {
        let features = data.shape()[2];
        RaggedBuffer {
            data: data.iter().cloned().collect(),
            subarrays: (0..data.shape()[0])
                .map(|i| i * data.shape()[1]..(i + 1) * data.shape()[1])
                .collect(),
            features,
        }
    }

    pub fn from_flattened(data: ArrayView2<T>, lengths: ArrayView1<i64>) -> Result<Self> {
        let features = data.shape()[1];
        let mut subarrays = Vec::new();
        let mut item = 0;
        for len in lengths.iter().cloned() {
            subarrays.push(item..(item + len as usize));
            item += len as usize;
        }
        if item != data.shape()[0] {
            Err(Error::generic(format!(
                "Lengths array specifies {} items, but data array has {} items",
                item,
                data.shape()[0]
            )))
        } else {
            Ok(RaggedBuffer {
                data: data.iter().cloned().collect(),
                subarrays,
                features,
            })
        }
    }

    pub fn extend(&mut self, other: &RaggedBuffer<T>) -> Result<()> {
        if self.features != other.features {
            return Err(Error::generic(format!(
                "Features mismatch: {} != {}",
                self.features, other.features
            )));
        }
        let item = self.items();
        self.data.extend(other.data.iter());
        self.subarrays
            .extend(other.subarrays.iter().map(|r| r.start + item..r.end + item));
        Ok(())
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.subarrays.clear();
    }

    // pub fn as_array<'a>(
    //     &self,
    //     py: Python<'a>,
    // ) -> PyResult<&'a numpy::PyArray<T, numpy::ndarray::Dim<[usize; 2]>>> {
    //     self.data
    //         .to_pyarray(py)
    //         .reshape((self.items, self.features))
    // }

    pub fn push(&mut self, data: &ArrayView2<T>) -> Result<()> {
        if data.dim().1 != self.features {
            return Err(Error::generic(format!(
                "Features mismatch: {} != {}",
                self.features,
                data.dim().1
            )));
        }
        self.subarrays
            .push(self.items()..(self.items() + data.dim().0));
        match data.as_slice() {
            Some(slice) => self.data.extend_from_slice(slice),
            None => {
                for x in data.iter() {
                    self.data.push(*x);
                }
            }
        }
        Ok(())
    }

    pub fn push_empty(&mut self) {
        self.subarrays.push(self.items()..self.items());
    }

    pub fn swizzle(&self, indices: ArrayView1<i64>) -> Result<RaggedBuffer<T>> {
        let indices = indices
            .as_slice()
            .ok_or_else(|| Error::generic("Indices must be a **contiguous** 1D array"))?;
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
        })
    }

    // TODO: dedupe with swizzle
    pub fn swizzle_usize(&self, indices: &[usize]) -> Result<RaggedBuffer<T>> {
        let mut subarrays = Vec::with_capacity(indices.len());
        let mut item = 0usize;
        for &i in indices {
            let sublen = self.subarrays[i].end - self.subarrays[i].start;
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
        })
    }

    pub fn get(&self, i: usize) -> RaggedBuffer<T> {
        let subarray = self.subarrays[i].clone();
        let Range { start, end } = subarray;
        RaggedBuffer {
            subarrays: vec![0..subarray.len()],
            data: self.data[start * self.features..end * self.features].to_vec(),
            features: self.features,
        }
    }

    pub fn size0(&self) -> usize {
        self.subarrays.len()
    }

    pub fn lengths(&self) -> Vec<i64> {
        self.subarrays
            .iter()
            .map(|r| (r.end - r.start) as i64)
            .collect::<Vec<_>>()
    }

    pub fn size1(&self, i: usize) -> Result<usize> {
        if i >= self.subarrays.len() {
            Err(Error::generic(format!("Index {} out of range", i)))
        } else {
            Ok(self.subarrays[i].end - self.subarrays[i].start)
        }
    }

    pub fn size2(&self) -> usize {
        self.features
    }

    pub fn __str__(&self) -> Result<String> {
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

    pub fn binop<Op: BinOp<T>>(&self, rhs: &RaggedBuffer<T>) -> Result<RaggedBuffer<T>> {
        if self.features == rhs.features && self.subarrays == rhs.subarrays {
            let mut data = Vec::with_capacity(self.data.len());
            for i in 0..self.data.len() {
                data.push(Op::op(self.data[i], rhs.data[i]));
            }
            Ok(RaggedBuffer {
                data,
                subarrays: self.subarrays.clone(),
                features: self.features,
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
            })
        } else if self.features == rhs.features
            && self.subarrays.len() == rhs.subarrays.len()
            && self.subarrays.iter().all(|r| r.end - r.start == 1)
        {
            rhs.binop::<Op>(self)
        } else {
            Err(Error::generic(format!(
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
        }
    }

    pub fn indices(&self, dim: usize) -> Result<RaggedBuffer<i64>> {
        match dim {
            0 => {
                let mut indices = Vec::with_capacity(self.items());
                for (index, subarray) in self.subarrays.iter().enumerate() {
                    for _ in subarray.clone() {
                        indices.push(index as i64);
                    }
                }
                Ok(RaggedBuffer {
                    subarrays: self.subarrays.clone(),
                    data: indices,
                    features: 1,
                })
            }
            1 => {
                let mut indices = Vec::with_capacity(self.items());
                for subarray in &self.subarrays {
                    for (i, _) in subarray.clone().enumerate() {
                        indices.push(i as i64);
                    }
                }
                Ok(RaggedBuffer {
                    subarrays: self.subarrays.clone(),
                    data: indices,
                    features: 1,
                })
            }
            _ => Err(Error::generic(format!("Invalid dimension {}", dim))),
        }
    }

    pub fn flat_indices(&self) -> Result<RaggedBuffer<i64>> {
        Ok(RaggedBuffer {
            subarrays: self.subarrays.clone(),
            data: (0..self.items()).map(|i| i as i64).collect(),
            features: 1,
        })
    }

    pub fn cat(buffers: &[&RaggedBuffer<T>], dim: usize) -> Result<RaggedBuffer<T>> {
        match dim {
            0 => {
                if buffers.iter().any(|b| b.features != buffers[0].features) {
                    return Err(Error::generic(format!(
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
                    item += buffer.items();
                }
                Ok(RaggedBuffer {
                    data,
                    subarrays,
                    features: buffers[0].features,
                })
            }
            1 => {
                if buffers
                    .iter()
                    .any(|b| b.subarrays.len() != buffers[0].subarrays.len())
                {
                    return Err(Error::generic(format!(
                        "All buffers must have the same number of subarrays, but found {}",
                        buffers
                            .iter()
                            .map(|b| b.subarrays.len().to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )));
                }
                if buffers.iter().any(|b| b.features != buffers[0].features) {
                    return Err(Error::generic(format!(
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
                })
            }
            2 => {
                // TODO: disallow broadcasting on some sequences but not other?
                // TODO: think more about empty sequences
                let sequences = buffers[0].size0();
                if buffers.iter().any(|b| b.size0() != sequences) {
                    return Err(Error::generic(format!(
                        "All buffers must have the same number of sequences, but found {}",
                        buffers
                            .iter()
                            .map(|b| b.size0().to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )));
                }

                let features = buffers.iter().map(|b| b.features).sum();
                let mut subarrays = Vec::with_capacity(sequences);
                let mut data = Vec::with_capacity(sequences * features);
                let mut items = 0;
                for iseq in 0..sequences {
                    let seqlen = if buffers.iter().any(|b| {
                        b.size1(iseq)
                            .expect("All sequences should be the same length.")
                            == 0
                    }) {
                        0
                    } else {
                        buffers
                            .iter()
                            .map(|b| {
                                b.size1(iseq)
                                    .expect("All sequences should be the same length.")
                            })
                            .max()
                            .expect("There should be at least one buffer.")
                    };
                    subarrays.push(items..items + seqlen);
                    items += seqlen;
                    for iitem in 0..seqlen {
                        for (ibuf, buffer) in buffers.iter().enumerate() {
                            let _items = buffer.subarrays[iseq].len();
                            if _items == 1 {
                                data.extend_from_slice(
                                    &buffer.data[buffer.subarrays[iseq].start * buffer.features
                                        ..buffer.subarrays[iseq].end * buffer.features],
                                );
                            } else {
                                if _items != seqlen {
                                    return Err(Error::generic(format!(
                                        "Buffer {} has {} items for sequence {}, but expected {}",
                                        ibuf, _items, iseq, seqlen
                                    )));
                                }
                                let start_item = buffer.subarrays[iseq].start + iitem;
                                data.extend_from_slice(
                                    &buffer.data[start_item * buffer.features
                                        ..(start_item + 1) * buffer.features],
                                );
                            }
                        }
                    }
                }

                Ok(RaggedBuffer {
                    data,
                    subarrays,
                    features,
                })
            }
            _ => Err(Error::generic(format!(
                "Invalid dimension {}, RaggedBuffer only has 3 dimensions",
                dim
            ))),
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn padpack(&self) -> Option<(Vec<i64>, Vec<f32>, Vec<i64>, (usize, usize))> {
        if self.subarrays.is_empty()
            || self
                .subarrays
                .iter()
                .all(|r| r.end - r.start == self.subarrays[0].end - self.subarrays[0].start)
        {
            return None;
        }

        let mut padbpack_index = vec![];
        let mut padpack_batch = vec![];
        let mut padpack_inverse_index = vec![];
        let max_seq_len = self
            .subarrays
            .iter()
            .map(|r| r.end - r.start)
            .max()
            .unwrap();
        let mut sequences: BinaryHeap<Sequence> = binary_heap::BinaryHeap::new();

        for (batch_index, subarray) in self.subarrays.iter().enumerate() {
            let (free, packed_batch_index) = match sequences.peek().cloned() {
                Some(seq) if seq.free >= subarray.end - subarray.start => {
                    sequences.pop();
                    (seq.free, seq.batch_index)
                }
                _ => {
                    for _ in 0..max_seq_len {
                        padbpack_index.push(0);
                        padpack_batch.push(f32::NAN);
                    }
                    (max_seq_len, sequences.len())
                }
            };

            for (i, item) in subarray.clone().enumerate() {
                let packed_index = packed_batch_index * max_seq_len + max_seq_len - free + i;
                padbpack_index[packed_index] = item as i64;
                padpack_batch[packed_index] = batch_index as f32;
                padpack_inverse_index.push(packed_index as i64);
            }
            sequences.push(Sequence {
                batch_index: packed_batch_index,
                free: free - (subarray.end - subarray.start),
            });
        }

        Some((
            padbpack_index,
            padpack_batch,
            padpack_inverse_index,
            (sequences.len(), max_seq_len),
        ))
    }

    pub fn items(&self) -> usize {
        self.subarrays.last().map(|r| r.end).unwrap_or(0)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct Sequence {
    free: usize,
    batch_index: usize,
}

impl Ord for Sequence {
    fn cmp(&self, other: &Self) -> Ordering {
        self.free
            .cmp(&other.free)
            .then_with(|| other.batch_index.cmp(&self.batch_index))
    }
}

impl PartialOrd for Sequence {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
