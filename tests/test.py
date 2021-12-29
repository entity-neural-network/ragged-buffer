from typing import TypeVar
import numpy as np
from ragged_buffer import (
    RaggedBufferF32,
    RaggedBufferI64,
    RaggedBufferBool,
    RaggedBuffer,
)
import ragged_buffer

rba = RaggedBufferF32(3)

ScalarType = TypeVar("ScalarType", bound=np.generic)


def generic_len(r: RaggedBuffer[ScalarType]) -> int:
    return sum([r.size1(s) for s in range(r.size0())]) * r.size2()


expected = """RaggedBuffer([
], '0 * var * 3 * f32)"""
assert str(rba) == expected

rba.push(np.array([[2.0, 3.0, 1.0], [1.0, 2.0, 3.0]], dtype=np.float32))
rba.push(
    np.array([[2.0, 3.0, 1.0], [1.0, 2.0, 3.0], [1.4, 2.4, 3.4]], dtype=np.float32)
)
rba.push(
    np.array(
        [[2.0, 3.0, 1.0], [1.0, 2.0, 3.0], [1.4, 2.4, 3.4], [1.4, 2.4, 3.4]],
        dtype=np.float32,
    )
)
rba.push(np.array([], dtype=np.float32))
rba.push_empty()

assert generic_len(rba) == 27, f"Expected 27 elements, got {generic_len(rba)}"

expected = """RaggedBuffer([
    [
        [2, 3, 1],
        [1, 2, 3],
    ],
    [
        [2, 3, 1],
        [1, 2, 3],
        [1.4, 2.4, 3.4],
    ],
    [
        [2, 3, 1],
        [1, 2, 3],
        [1.4, 2.4, 3.4],
        [1.4, 2.4, 3.4],
    ],
    [],
    [],
], '5 * var * 3 * f32)"""


assert str(rba) == expected, str(rba)
flattened = np.array(
    [
        [2.0, 3.0, 1.0],
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 1.0],
        [1.0, 2.0, 3.0],
        [1.4, 2.4, 3.4],
        [2.0, 3.0, 1.0],
        [1.0, 2.0, 3.0],
        [1.4, 2.4, 3.4],
        [1.4, 2.4, 3.4],
    ],
    dtype=np.float32,
)
assert np.all(rba.as_array() == flattened)
assert rba == RaggedBufferF32.from_flattened(
    flattened=flattened, lengths=np.array([2, 3, 4, 0, 0], dtype=np.int64),
)
assert RaggedBufferF32(3) == RaggedBufferF32(3)


rba2 = RaggedBufferF32(3)
rba2.push(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32))
rba2.push(
    np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
)
rba.extend(rba2)
assert rba == RaggedBufferF32.from_flattened(
    flattened=np.concatenate([flattened, np.zeros((5, 3), dtype=np.float32)]),
    lengths=np.array([2, 3, 4, 0, 0, 2, 3], dtype=np.int64),
)
rba[np.random.permutation(rba.size0())]

assert rba.size0() == 7
assert rba.size1(0) == 2
assert rba.size1(1) == 3
assert rba.size1(2) == 4
assert rba.size1(3) == 0
assert rba.size1(4) == 0
assert rba.size1(5) == 2
assert rba.size1(6) == 3
assert np.all(rba.size1() == np.array([2, 3, 4, 0, 0, 2, 3], dtype=np.int64))
assert rba.size2() == 3

rba.clear()
assert rba == RaggedBufferF32(3)

rb3 = RaggedBufferI64(1)
rb3.push(np.array([[0]], dtype=np.int64))
rb3.push(np.array([[0], [1], [2]], dtype=np.int64))
rb3.push(np.array([[0], [5]], dtype=np.int64))
assert rb3 == RaggedBufferI64.from_flattened(
    flattened=np.array([[0], [0], [1], [2], [0], [5]], dtype=np.int64),
    lengths=np.array([1, 3, 2], dtype=np.int64),
)
# Shuffle
rb3[np.random.permutation(rb3.size0())]

rb4 = RaggedBufferI64.from_array(np.array([0, 3, 10], dtype=np.int64).reshape(3, 1, 1))
assert rb4 == RaggedBufferI64.from_flattened(
    flattened=np.array([[0], [3], [10]], dtype=np.int64),
    lengths=np.array([1, 1, 1], dtype=np.int64),
)
rb5 = rb3 + rb4
assert np.all(
    rb5.as_array() == np.array([[0], [3], [4], [5], [10], [15]], dtype=np.int64)
), f"{rb5.as_array()}"
assert rb3 + rb4 == rb4 + rb3
assert rb5 * 10 == RaggedBufferI64.from_flattened(
    flattened=np.array([[0], [30], [40], [50], [100], [150]], dtype=np.int64),
    lengths=np.array([1, 3, 2], dtype=np.int64),
)
assert rb5.indices(1) == RaggedBufferI64.from_flattened(
    flattened=np.array([[0], [0], [1], [2], [0], [1]], dtype=np.int64),
    lengths=np.array([1, 3, 2], dtype=np.int64),
)


rb6 = RaggedBufferF32.from_flattened(
    np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ],
        dtype=np.float32,
    ),
    np.array([3, 0, 2, 1], dtype=np.int64),
)
assert rb6[np.array([1, 3, 0], dtype=np.int64)] == RaggedBufferF32.from_flattened(
    np.array([[13.0, 14.0, 15.0], [0, 0, 0], [1, 2, 3], [4, 5, 6],], dtype=np.float32),
    np.array([0, 1, 3], dtype=np.int64),
)

assert np.all(
    rb6[2].as_array()
    == np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32)
), f"{rb6[2]}"


entities1 = RaggedBufferF32.from_flattened(
    np.zeros((6, 64), dtype=np.float32), np.array([3, 1, 2], dtype=np.int64)
)
entities2 = RaggedBufferF32.from_flattened(
    np.zeros((3, 64), dtype=np.float32), np.array([1, 2, 0], dtype=np.int64)
)
assert np.all(entities1.size1() == np.array([3, 1, 2], dtype=np.int64))
assert np.all(entities2.size1() == np.array([1, 2, 0], dtype=np.int64))
print("TEST 1 PASSED")

bi1 = entities1.indices(0).as_array().flatten()
bi2 = entities2.indices(0).as_array().flatten()
assert np.all(bi1 == np.array([0, 0, 0, 1, 2, 2], dtype=np.int64)), f"{bi1}"
assert np.all(bi2 == np.array([0, 1, 1], dtype=np.int64)), f"{bi2}"
print("TEST 2 PASSED")

flati1 = entities1.flat_indices()
print("TEST 3 PASSED")
flati2 = entities2.flat_indices() + 6
print("TEST 4 PASSED")
flat = ragged_buffer.cat([flati1, flati2, flati1, flati2], dim=1).as_array().flatten()
assert np.all(
    flat
    == np.array([0, 1, 2, 6, 0, 1, 2, 6, 3, 7, 8, 3, 7, 8, 4, 5, 4, 5], dtype=np.int64),
), f"{flat} {ragged_buffer.cat([flati1, flati2, flati1, flati2], dim=1)}"
print("TEST 5 PASSED")


mask = RaggedBufferI64.from_array(np.zeros((4, 1, 1), dtype=np.int64))
offset = RaggedBufferI64.from_flattened(
    np.array([0, 1, 2, 3, 13, 22, 32, 41, 42, 43, 44, 45,], dtype=np.int64).reshape(
        -1, 1,
    ),
    np.ones(12, dtype=np.int64),
)
try:
    mask = mask + offset
except ValueError as e:
    pass
else:
    assert False, "Did not raise ValueError"


zerofeats = RaggedBufferF32(features=0)
zerofeats.push(np.zeros((1, 0), dtype=np.float32))
zerofeats.push(np.zeros((0, 0), dtype=np.float32))
assert zerofeats.as_array().shape == (1, 0), f"{zerofeats.as_array().shape}"


boolrb = RaggedBufferBool.from_flattened(
    np.array([[True, False, True], [False, True, False]], dtype=np.bool_),
    np.array([2, 0], dtype=np.int64),
)
assert boolrb.as_array().shape == (2, 3), f"{boolrb.as_array().shape}"
assert np.all(
    boolrb.as_array()
    == np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
), f"{boolrb.as_array()}"

batch_index = RaggedBufferI64.from_flattened(
    np.array(
        [[0], [0], [0], [0], [0], [0], [1], [1], [1], [2], [2], [2], [2], [4]],
        dtype=np.int64,
    ),
    np.array([6, 3, 4, 0, 1], dtype=np.int64),
)

padpack = batch_index.padpack()
assert padpack is not None
padpack_index, padpack_batch, padpack_inverse_index = padpack

assert np.all(
    padpack_index
    == np.array(
        [[0, 1, 2, 3, 4, 5], [6, 7, 8, 13, 0, 0], [9, 10, 11, 12, 0, 0]], dtype=np.int64
    ),
), f"{padpack_index}"


np.testing.assert_equal(
    padpack_batch,
    np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 4.0, np.NaN, np.NaN],
            [2.0, 2.0, 2.0, 2.0, np.NaN, np.NaN],
        ],
        dtype=np.float32,
    ),
)

np.testing.assert_equal(
    padpack_inverse_index,
    np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 9], dtype=np.int64),
)
