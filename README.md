# ENN Ragged Buffer

[![Actions Status](https://github.com/entity-neural-network/ragged-buffer/workflows/Test/badge.svg)](https://github.com/entity-neural-network/ragged-buffer/actions)
[![PyPI](https://img.shields.io/pypi/v/ragged-buffer.svg?style=flat-square)](https://pypi.org/project/ragged-buffer/)
[![Discord](https://img.shields.io/discord/913497968701747270?style=flat-square)](https://discord.gg/SjVqhSW4Qf)

This Python package implements an efficient `RaggedBuffer` datatype that is similar to
a 3D numpy array, but which allows for variable sequence length in the second
dimension. It was created primarily for use in [ENN-PPO](https://github.com/entity-neural-network/incubator/tree/main/enn_ppo) 
and currently only supports a small selection of the numpy array methods.

![Ragged Buffer](https://user-images.githubusercontent.com/12845088/143787823-c6a585de-aeda-429c-9824-f4b4a98e6cea.png)

## User Guide

Install the package with `pip install ragged-buffer`.
The package currently supports three `RaggedBuffer` variants, `RaggedBufferF32`, `RaggedBufferI64`, and `RaggedBufferBool`.

<!-- no toc -->
- [Creating a RaggedBuffer](#creating-a-raggedbuffer)
- [Get size](#get-size)
- [Convert to numpy array](#convert-to-numpy-array)
- [Indexing](#indexing)
- [Addition](#addition)
- [Concatenation](#concatentation)
- [Clear](#clear)

### Creating a RaggedBuffer

There are three ways to create a `RaggedBuffer`:
- `RaggedBufferF32(features: int)` creates an empty `RaggedBuffer` with the specified number of features.
- `RaggedBufferF32.from_flattened(flattened: np.ndarray, lenghts: np.ndarray)` creates a `RaggedBuffer` from a flattened 2D numpy array and a 1D numpy array of lengths.
- `RaggedBufferF32.from_array` creates a `RaggedBuffer` (with equal sequence lenghts) from a 3D numpy array.

Creating an empty buffer and pushing each row:

```python
import numpy as np
from ragged_buffer import RaggedBufferF32

# Create an empty RaggedBuffer with a feature size of 3
buffer = RaggedBufferF32(3)
# Push sequences with 3, 5, 0, and 1 elements
buffer.push(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32))
buffer.push(np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]], dtype=np.float32))
buffer.push(np.array([], dtype=np.float32))  # Alternative: `buffer.push_empty()`
buffer.push(np.array([[25, 25, 27]], dtype=np.float32))
```

Creating a RaggedBuffer from a flat 2D numpy array which combines the first and second dimension,
and an array of sequence lengths:

```python
import numpy as np
from ragged_buffer import RaggedBufferF32

buffer = RaggedBufferF32.from_flattened(
    np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 25, 27]], dtype=np.float32),
    np.array([3, 5, 0, 1], dtype=np.int64))
)
```

Creating a RaggedBuffer from a 3D numpy array (all sequences have the same length):

```python
import numpy as np
from ragged_buffer import RaggedBufferF32

buffer = RaggedBufferF32.from_array(np.zeros((4, 5, 3), dtype=np.float32))
```

### Get size

The `size0`, `size1`, and `size2` methods return the number of sequences, the number of elements in a sequence, and the number of features respectively.

```python
import numpy as np
from ragged_buffer import RaggedBufferF32

buffer = RaggedBufferF32.from_flattened(
    np.zeros((9, 64), dtype=np.float32),
    np.array([3, 5, 0, 1], dtype=np.int64))
)

# Get size of the first/batch dimension.
assert buffer.size0() == 10
# Get size of individual sequences.
assert buffer.size1(1) == 5
assert buffer.size1(2) == 0
# Get size of the last/feature dimension.
assert buffer.size2() == 64
```

### Convert to numpy array

`as_aray` converts a `RaggedBuffer` to a flat 2D numpy array that combines the first and second dimension.

```python
import numpy as np
from ragged_buffer import RaggedBufferI64

buffer = RaggedBufferI64(1)
buffer.push(np.array([[1], [1], [1]], dtype=np.int64))
buffer.push(np.array([[2], [2]], dtype=np.int64))
assert np.all(buffer.as_array(), np.array([[1], [1], [1], [2], [2]], dtype=np.int64))
```

### Indexing

You can index a `RaggedBuffer` with a single integer (returning a `RaggedBuffer` with a single sequence), or with a numpy array of integers selecting/permuting multiple sequences.

```python
import numpy as np
from ragged_buffer import RaggedBufferF32

# Create a new `RaggedBufferF32`
buffer = RaggedBufferF32.from_flattened(
    np.arange(0, 40, dtype=np.float32).reshape(10, 4),
    np.array([3, 5, 0, 1], dtype=np.int64)
)

# Retrieve the first sequence.
assert np.all(
    buffer[0].as_array() ==
    np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], dtype=np.float32)
)

# Get a RaggedBatch with 2 randomly selected sequences.
buffer[np.random.permutation(4)[:2]]
```

### Addition

You can add two `RaggedBuffer`s with the `+` operator if they have the same number of sequences, sequence lengths, and features. You can also add a `RaggedBuffer` where all sequences have a length of 1 to a `RaggedBuffer` with variable length sequences, broadcasting along each sequence.

```python
import numpy as np
from ragged_buffer import RaggedBufferF32

# Create ragged buffer with dimensions (3, [1, 3, 2], 1)
rb3 = RaggedBufferI64(1)
rb3.push(np.array([[0]], dtype=np.int64))
rb3.push(np.array([[0], [1], [2]], dtype=np.int64))
rb3.push(np.array([[0], [5]], dtype=np.int64))

# Create ragged buffer with dimensions (3, [1, 1, 1], 1)
rb4 = RaggedBufferI64.from_array(np.array([0, 3, 10], dtype=np.int64).reshape(3, 1, 1))

# Add rb3 and rb4, broadcasting along the sequence dimension.
rb5 = rb3 + rb4
assert np.all(
    rb5.as_array() == np.array([[0], [3], [4], [5], [10], [15]], dtype=np.int64)
)
```

### Concatenation

The `extend` method can be used to mutate a `RaggedBuffer` by appending another `RaggedBuffer` to it.

```python
import numpy as np
from ragged_buffer import RaggedBufferF32


rb1 = RaggedBufferF32.from_array(np.zeros((4, 5, 3), dtype=np.float32))
rb2 = RaggedBufferF32.from_array(np.zeros((2, 5, 3), dtype=np.float32))
rb1.extend(r2)
assert rb1.size0() == 6
```

### Clear

The `clear` method removes all elements from a `RaggedBuffer` without deallocating the underlying memory.


```python
import numpy as np
from ragged_buffer import RaggedBufferF32

rb = RaggedBufferF32.from_array(np.zeros((4, 5, 3), dtype=np.float32))
rb.clear()
assert rb.size0() == 0
```

## License

ENN Ragged Buffer dual-licensed under Apache-2.0 and MIT.