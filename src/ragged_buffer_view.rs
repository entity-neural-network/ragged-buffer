use std::fmt::Display;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use pyo3::{exceptions, PyErr, PyResult, Python};

use crate::monomorphs::Index;
use crate::ragged_buffer::{BinOp, Error, RaggedBuffer};

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum Slice {
    Range {
        start: usize,
        end: usize,
        step: usize,
    },
    Permutation(Vec<usize>),
}

impl Slice {
    fn into_iter(self) -> Box<dyn Iterator<Item = usize>> {
        match self {
            Slice::Range { start, end, step } => Box::new((start..end).step_by(step)),
            Slice::Permutation(permutation) => Box::new(permutation.into_iter()),
        }
    }

    fn len(&self) -> usize {
        match self {
            Slice::Range { start, end, step } => (end - start + step - 1) / step,
            Slice::Permutation(permutation) => permutation.len(),
        }
    }
}

// TODO: Eq/PartialEq/Hash
#[derive(Clone, Debug)]
pub struct RaggedBufferView<T> {
    pub inner: Arc<RwLock<RaggedBuffer<T>>>,
    view: Option<(Slice, Slice, Slice)>,
}

impl<T: numpy::Element + Copy + Display + std::fmt::Debug> RaggedBufferView<T> {
    pub fn new(features: usize) -> Self {
        RaggedBufferView {
            inner: Arc::new(RwLock::new(RaggedBuffer::new(features))),
            view: None,
        }
    }

    pub fn get_slice<'a>(
        &self,
        py: Python<'a>,
        i0: Index,
        i1: Index,
        i2: Index,
    ) -> PyResult<RaggedBufferView<T>> {
        // TODO: Check that i0, i1, i2 are valid indices
        let materialized = self.materialize();
        let v0 = match i0 {
            Index::PermutationNP(np) => {
                Slice::Permutation(np.to_vec()?.into_iter().map(|x| x as usize).collect())
            }
            Index::Permutation(p) => Slice::Permutation(p),
            Index::Int(i) => Slice::Range {
                start: i,
                end: i + 1,
                step: 1,
            },
            Index::Slice(slice) => {
                let indices = slice
                    .as_ref(py)
                    .indices(materialized.size0().try_into().unwrap())?;
                Slice::Range {
                    start: indices.start as usize,
                    end: indices.stop as usize,
                    step: indices.step as usize,
                }
            }
        };
        let v1 = match i1 {
            Index::PermutationNP(np) => {
                Slice::Permutation(np.to_vec()?.into_iter().map(|x| x as usize).collect())
            }
            Index::Permutation(p) => Slice::Permutation(p),
            Index::Int(i) => Slice::Range {
                start: i,
                end: i + 1,
                step: 1,
            },
            Index::Slice(slice) => {
                let indices = slice
                    .as_ref(py)
                    .indices(materialized.len()?.try_into().unwrap())?;
                Slice::Range {
                    start: indices.start as usize,
                    end: indices.stop as usize,
                    step: indices.step as usize,
                }
            }
        };
        let v2 = match i2 {
            Index::PermutationNP(np) => {
                Slice::Permutation(np.to_vec()?.into_iter().map(|x| x as usize).collect())
            }
            Index::Permutation(p) => Slice::Permutation(p),
            Index::Int(i) => Slice::Range {
                start: i,
                end: i + 1,
                step: 1,
            },
            Index::Slice(slice) => {
                let indices = slice
                    .as_ref(py)
                    .indices(materialized.size2().try_into().unwrap())?;
                Slice::Range {
                    start: indices.start as usize,
                    end: indices.stop as usize,
                    step: indices.step as usize,
                }
            }
        };

        Ok(RaggedBufferView {
            inner: materialized.inner,
            view: Some((v0, v1, v2)),
        })
    }

    fn get(&self) -> RwLockReadGuard<RaggedBuffer<T>> {
        self.inner.read().unwrap()
    }

    fn get_mut(&self) -> RwLockWriteGuard<RaggedBuffer<T>> {
        self.inner.write().unwrap()
    }

    fn make_contiguous(&mut self) {
        let materialized = self.materialize();
        self.inner = materialized.inner;
        self.view = None;
    }
    fn require_contiguous(&self, method_name: &str) -> PyResult<()> {
        match self.view {
            Some(_) => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Cannot call method {} on a view. Call .materialize() first to get a materialized copy of the view.",
                method_name
            ))),
            None => Ok(()),
        }
    }

    pub fn from_array(data: PyReadonlyArray3<T>) -> Self {
        RaggedBufferView {
            inner: Arc::new(RwLock::new(RaggedBuffer::from_array(data.as_array()))),
            view: None,
        }
    }

    pub fn from_flattened(
        data: PyReadonlyArray2<T>,
        lengths: PyReadonlyArray1<i64>,
    ) -> PyResult<Self> {
        Ok(RaggedBufferView {
            inner: Arc::new(RwLock::new(RaggedBuffer::from_flattened(
                data.as_array(),
                lengths.as_array(),
            )?)),
            view: None,
        })
    }

    pub fn extend(&mut self, other: &RaggedBufferView<T>) -> PyResult<()> {
        self.make_contiguous();
        let other = other.materialize();
        let other = other.get();
        self.get_mut().extend(&*other).map_err(Into::into)
    }

    pub fn clear(&mut self) -> PyResult<()> {
        self.make_contiguous();
        self.get_mut().clear();
        Ok(())
    }

    pub fn as_array<'a>(
        &self,
        py: Python<'a>,
    ) -> PyResult<&'a numpy::PyArray<T, numpy::ndarray::Dim<[usize; 2]>>> {
        match self.view {
            None => {
                let inner = self.get();
                inner
                    .data
                    .to_pyarray(py)
                    .reshape((inner.items(), inner.features))
                    .map_err(Into::into)
            }
            _ => self.materialize().as_array(py),
        }
    }

    pub fn materialize(&self) -> RaggedBufferView<T> {
        match self.view.clone() {
            Some((
                Slice::Range {
                    start: start0,
                    end: end0,
                    step: step0,
                },
                Slice::Range {
                    start: start1,
                    end: end1,
                    step: step1,
                },
                Slice::Range {
                    start: start2,
                    end: end2,
                    step: step2,
                },
            )) => {
                let mut data = Vec::new();
                let mut subarrays = Vec::new();
                let mut item = 0;
                let inner = self.get();
                for i0 in (start0..end0).step_by(step0) {
                    let mut items = 0;
                    for i1 in inner.subarrays[i0]
                        .clone()
                        .skip(start1)
                        .take(end1 - start1)
                        .step_by(step1)
                    {
                        for i2 in (start2..end2).step_by(step2) {
                            data.push(inner.data[i1 * inner.features + i2]);
                        }
                        items += 1;
                    }
                    subarrays.push(item..item + items);
                    item += items;
                }
                let features = (end2 - start2 + step2 - 1) / step2;
                let materialized = RaggedBuffer {
                    data,
                    subarrays,
                    features,
                };
                RaggedBufferView {
                    inner: Arc::new(RwLock::new(materialized)),
                    view: None,
                }
            }
            Some((v0, v1, v2)) => {
                let mut data = Vec::new();
                let mut items = 0;
                let mut subarrays = Vec::new();
                let inner = self.get();
                for i0 in v0.into_iter() {
                    let item_start = items;
                    let subarray = inner.subarrays[i0].clone();
                    for i1 in v1.clone().into_iter() {
                        if i1 >= subarray.len() {
                            break;
                        }
                        let offset = (subarray.start + i1) * inner.features;
                        for i2 in v2.clone().into_iter() {
                            data.push(inner.data[offset + i2]);
                        }
                        items += 1;
                    }
                    subarrays.push(item_start..items);
                }
                let features = v2.len();
                let materialized = RaggedBuffer {
                    data,
                    subarrays,
                    features,
                };
                RaggedBufferView {
                    inner: Arc::new(RwLock::new(materialized)),
                    view: None,
                }
            }
            None => self.clone(),
        }
    }

    pub fn push(&mut self, x: &PyReadonlyArray2<T>) -> PyResult<()> {
        self.make_contiguous();
        self.get_mut().push(&x.as_array()).map_err(Into::into)
    }

    pub fn push_empty(&mut self) -> PyResult<()> {
        self.make_contiguous();
        self.get_mut().push_empty();
        Ok(())
    }

    pub fn swizzle(&self, indices: PyReadonlyArray1<i64>) -> PyResult<RaggedBufferView<T>> {
        match self.view {
            Some((_, _, _)) => todo!(),
            None => Ok(self.get().swizzle(indices.as_array())?.view()),
        }
    }

    pub fn swizzle_usize(&self, indices: &[usize]) -> PyResult<RaggedBufferView<T>> {
        match self.view {
            Some((_, _, _)) => todo!(),
            None => Ok(self.get().swizzle_usize(indices)?.view()),
        }
    }

    pub fn get_sequence(&self, i: usize) -> PyResult<RaggedBufferView<T>> {
        self.require_contiguous("get_sequence")?;
        Ok(self.get().get(i).view())
    }

    pub fn size0(&self) -> usize {
        match &self.view {
            Some((s0, _, _)) => s0.len(),
            None => self.get().size0(),
        }
    }

    pub fn size2(&self) -> usize {
        match &self.view {
            Some((_, _, s2)) => s2.len(),
            None => self.get().size2(),
        }
    }

    pub fn lengths<'a>(
        &self,
        py: Python<'a>,
    ) -> PyResult<&'a numpy::PyArray<i64, numpy::ndarray::Dim<[usize; 1]>>> {
        match self.view {
            None => Ok(self.get().lengths().to_pyarray(py)),
            Some((
                Slice::Range {
                    start: start0,
                    end: end0,
                    step: step0,
                },
                Slice::Range {
                    start: start1,
                    end: end1,
                    step: step1,
                },
                _,
            )) => {
                let mut lengths = Vec::with_capacity((end0 - start0) / step0);
                let inner = self.get();
                for i0 in (start0..end0).step_by(step0) {
                    let end1 = std::cmp::min(end1, inner.subarrays[i0].len());
                    if end1 > start1 {
                        let stepsf = (end1 - start1) / step1;
                        lengths.push(
                            stepsf as i64 + if stepsf * step1 < end1 - start1 { 1 } else { 0 },
                        );
                    } else {
                        lengths.push(0);
                    }
                }
                Ok(lengths.to_pyarray(py))
            }
            _ => {
                self.require_contiguous("lengths")?;
                Ok(self.get().lengths().to_pyarray(py))
            }
        }
    }

    pub fn size1(&mut self, i: usize) -> PyResult<usize> {
        self.make_contiguous();
        self.get().size1(i).map_err(Into::into)
    }

    pub fn __str__(&self) -> PyResult<String> {
        self.materialize().get().__str__().map_err(Into::into)
    }

    pub fn binop<Op: BinOp<T>>(&self, rhs: &RaggedBufferView<T>) -> PyResult<RaggedBufferView<T>> {
        self.require_contiguous("binop")?;
        Ok(self.get().binop::<Op>(&*rhs.get())?.view())
    }

    pub fn op_scalar<Op: BinOp<T>>(&self, scalar: T) -> PyResult<RaggedBufferView<T>> {
        self.require_contiguous("op_scalar")?;
        Ok(self.get().op_scalar::<Op>(scalar).view())
    }

    pub fn indices(&mut self, dim: usize) -> PyResult<RaggedBufferView<i64>> {
        self.make_contiguous();
        Ok(self.get().indices(dim)?.view())
    }

    pub fn flat_indices(&mut self) -> PyResult<RaggedBufferView<i64>> {
        self.make_contiguous();
        Ok(self.get().flat_indices()?.view())
    }

    pub fn cat(buffers: &[&RaggedBufferView<T>], dim: usize) -> PyResult<RaggedBufferView<T>> {
        if buffers.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cat requires at least one ragged buffer",
            ));
        }
        let mut rbs = Vec::new();
        for b in buffers {
            b.require_contiguous("cat")?;
            rbs.push(b.get());
        }
        let rb = RaggedBuffer::cat(&rbs.iter().map(|r| &**r).collect::<Vec<_>>(), dim)?;
        Ok(RaggedBufferView {
            inner: Arc::new(RwLock::new(rb)),
            view: None,
        })
    }

    #[allow(clippy::type_complexity)]
    pub fn padpack(&mut self) -> PyResult<Option<(Vec<i64>, Vec<f32>, Vec<i64>, (usize, usize))>> {
        self.make_contiguous();
        Ok(self.get().padpack())
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> PyResult<usize> {
        self.require_contiguous("len")?;
        Ok(self.get().len())
    }

    pub fn is_empty(&mut self) -> PyResult<bool> {
        self.make_contiguous();
        Ok(self.get().is_empty())
    }

    pub fn items(&mut self) -> PyResult<usize> {
        self.make_contiguous();
        Ok(self.get().items())
    }

    pub fn binop_mut<Op: BinOp<T>>(&self, rhs: &RaggedBufferView<T>) -> PyResult<()> {
        let (lhs_i0, lhs_i1, lhs_i2) = self.view.clone().unwrap();
        let (rhs_i0, rhs_i1, rhs_i2) = rhs.view.clone().unwrap();

        let (lhs_iter_0, rhs_iter_0) = if self.size0() == rhs.size0() {
            (lhs_i0.into_iter(), rhs_i0.into_iter())
        } else {
            return Err(exceptions::PyValueError::new_err(format!(
                "size mismatch in first dimension: {} != {}",
                self.size0(),
                rhs.size0(),
            )));
        };
        assert!(matches!(lhs_i1, Slice::Range { .. }));
        assert!(matches!(rhs_i1, Slice::Range { .. }));
        if self.size2() != rhs.size2() {
            return Err(exceptions::PyValueError::new_err(format!(
                "size mismatch in third dimension: {} != {}",
                self.size2(),
                rhs.size2(),
            )));
        };

        let stride2l = self.get().size2();
        let stride2r = rhs.get().size2();

        let mut lhs = self.get_mut();
        let rhs = rhs.get();
        for (l0, r0) in lhs_iter_0.zip(rhs_iter_0) {
            let (lhs_iter_1, rhs_iter_1): (
                Box<dyn Iterator<Item = usize>>,
                Box<dyn Iterator<Item = usize>>,
            ) = if lhs.subarrays[l0].len() != rhs.subarrays[r0].len() {
                if lhs.subarrays[l0].len() == 1 {
                    (
                        Box::new(
                            vec![lhs.subarrays[l0].start; rhs.subarrays[r0].len()].into_iter(),
                        ),
                        Box::new(rhs.subarrays[r0].clone()),
                    )
                } else if rhs.subarrays[r0].len() == 1 {
                    (
                        Box::new(lhs.subarrays[l0].clone()),
                        Box::new(
                            vec![rhs.subarrays[r0].start; lhs.subarrays[l0].len()].into_iter(),
                        ),
                    )
                } else {
                    return Err(exceptions::PyValueError::new_err(format!(
                        "size mismatch between {}th and {}th sequence: {} != {}",
                        l0,
                        r0,
                        lhs.subarrays[l0].len(),
                        rhs.subarrays[r0].len(),
                    )));
                }
            } else {
                (
                    Box::new(lhs.subarrays[l0].clone()),
                    Box::new(rhs.subarrays[r0].clone()),
                )
            };
            for (l1, r1) in lhs_iter_1.zip(rhs_iter_1) {
                for (l2, r2) in lhs_i2.clone().into_iter().zip(rhs_i2.clone().into_iter()) {
                    lhs.data[l1 * stride2l + l2] =
                        Op::op(lhs.data[l1 * stride2l + l2], rhs.data[r1 * stride2r + r2]);
                }
            }
        }

        Ok(())
    }

    pub fn deepclone(&self) -> RaggedBufferView<T> {
        let inner = self.get().clone();
        RaggedBufferView {
            inner: Arc::new(RwLock::new(inner)),
            view: self.view.clone(),
        }
    }
}

pub fn translate_rotate(
    source: &RaggedBufferView<f32>,
    translation: &RaggedBufferView<f32>,
    rotation: &RaggedBufferView<f32>,
) -> PyResult<()> {
    if source.size0() != translation.size0() {
        return Err(exceptions::PyValueError::new_err(format!(
            "size mismatch in first dimension: {} != {}",
            source.size0(),
            translation.size0(),
        )));
    }
    if source.size2() != 2 {
        return Err(exceptions::PyValueError::new_err(format!(
            "expected 2D source, got {}D",
            source.size2(),
        )));
    }
    if translation.size2() != 2 {
        return Err(exceptions::PyValueError::new_err(format!(
            "expected 2D translation, got {}D",
            translation.size2(),
        )));
    }
    if rotation.size2() != 2 {
        return Err(exceptions::PyValueError::new_err(format!(
            "expected rotation to be a 2D direction, got {}D",
            rotation.size2(),
        )));
    }
    let (s0, _, s2) = source.view.clone().unwrap();
    let (t0, _, t2) = translation.view.clone().unwrap();
    let (r0, _, r2) = rotation.view.clone().unwrap();
    let mut source = source.get_mut();
    let translation = translation.get();
    let rotation = rotation.get();

    let ss0 = source.size0();
    let ts0 = translation.size0();
    let rs0 = rotation.size0();
    match s0 {
        Slice::Range { start, end, step } if start == 0 && end == ss0 && step == 1 => {}
        _ => {
            return Err(exceptions::PyValueError::new_err(
                "view on first dimension of source not supported".to_string(),
            ))
        }
    }
    match t0 {
        Slice::Range { start, end, step } if start == 0 && end == ts0 && step == 1 => {}
        _ => {
            return Err(exceptions::PyValueError::new_err(
                "view on first dimension of translation not supported".to_string(),
            ))
        }
    }
    match r0 {
        Slice::Range { start, end, step } if start == 0 && end == rs0 && step == 1 => {}
        _ => {
            return Err(exceptions::PyValueError::new_err(
                "view on first dimension of rotation not supported".to_string(),
            ))
        }
    }
    let (sxi, syi) = match s2 {
        Slice::Range { start, step, .. } => (start, start + step),
        Slice::Permutation(indices) => (indices[0], indices[1]),
    };
    let (txi, tyi) = match t2 {
        Slice::Range { start, step, .. } => (start, start + step),
        Slice::Permutation(indices) => (indices[0], indices[1]),
    };
    let (rxi, ryi) = match r2 {
        Slice::Range { start, step, .. } => (start, start + step),
        Slice::Permutation(indices) => (indices[0], indices[1]),
    };
    let sstride = source.features;
    for i0 in 0..source.size0() {
        if translation.size1(i0)? != 1 || rotation.size1(i0)? != 1 {
            return Err(exceptions::PyValueError::new_err(format!(
                "must have single item in translation and rotation for each sequence, but got {} and {} items for sequence {}",
                translation.size1(i0)?, rotation.size1(i0)?, i0,
            )));
        }
        // TODO: check no view on dim 1
        for i1 in source.subarrays[i0].clone() {
            let sstart = i1 * sstride;
            source.data[sstart + sxi] -= translation.data[i0 * translation.features + txi];
            source.data[sstart + syi] -= translation.data[i0 * translation.features + tyi];
            let rx = rotation.data[i0 * rotation.features + rxi];
            let ry = rotation.data[i0 * rotation.features + ryi];
            let sx = source.data[sstart + sxi];
            let sy = source.data[sstart + syi];
            source.data[sstart + sxi] = rx * sx + ry * sy;
            source.data[sstart + syi] = -ry * sx + rx * sy;
        }
    }
    Ok(())
}

impl<T: numpy::Element + Copy + Display + std::fmt::Debug + PartialEq> PartialEq
    for RaggedBufferView<T>
{
    fn eq(&self, other: &RaggedBufferView<T>) -> bool {
        // TODO: implement for views
        self.require_contiguous("eq").unwrap();
        other.require_contiguous("eq").unwrap();
        *self.get() == *other.get()
    }
}

impl<T: numpy::Element + Copy + Display + std::fmt::Debug + Eq> Eq for RaggedBufferView<T> {}

impl<T> RaggedBuffer<T> {
    pub fn view(self) -> RaggedBufferView<T> {
        RaggedBufferView {
            inner: Arc::new(RwLock::new(self)),
            view: None,
        }
    }
}

impl From<Error> for PyErr {
    fn from(Error::Generic(msg): Error) -> PyErr {
        exceptions::PyValueError::new_err(msg)
    }
}
