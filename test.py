import numpy as np
from ragged_buffer import RaggedBufferF32, RaggedBufferI64

rba = RaggedBufferF32(3)

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
rba.push(np.array([[]], dtype=np.float32))

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
], '4 * var * 3 * f32)"""


assert str(rba) == expected
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
    flattened=flattened, lengths=np.array([2, 3, 4, 0], dtype=np.int64),
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
    lengths=np.array([2, 3, 4, 0, 2, 3], dtype=np.int64),
)

assert rba.size0() == 6
assert rba.size1(0) == 2
assert rba.size1(1) == 3
assert rba.size1(2) == 4
assert rba.size1(3) == 0
assert rba.size1(4) == 2
assert rba.size1(5) == 3
assert rba.size2() == 3

rb3 = RaggedBufferI64(1)
rb3.push(np.array([[0]], dtype=np.int64))
rb3.push(np.array([[0], [1], [2]], dtype=np.int64))
rb3.push(np.array([[0], [5]], dtype=np.int64))
assert rb3 == RaggedBufferI64.from_flattened(
    flattened=np.array([[0], [0], [1], [2], [0], [5]], dtype=np.int64),
    lengths=np.array([1, 3, 2], dtype=np.int64),
)

rb4 = RaggedBufferI64.from_array(np.array([0, 3, 10], dtype=np.int64).reshape(3, 1, 1))
assert rb4 == RaggedBufferI64.from_flattened(
    flattened=np.array([[0], [3], [10]], dtype=np.int64),
    lengths=np.array([1, 1, 1], dtype=np.int64),
)
rb5 = rb3 + rb4
assert np.all(
    rb5.as_array() == np.array([[0], [3], [4], [5], [10], [15]], dtype=np.int64)
), f"{rb5.as_array()}"

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
