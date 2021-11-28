import numpy as np
import ragged_buffer
from ragged_buffer import RaggedBufferF32, RaggedBufferI64

print(ragged_buffer.test("yoyoyo"))

rba = RaggedBufferF32(3)
print(rba)
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
print(rba)

print(rba.as_array())


rba2 = RaggedBufferF32(3)
rba2.push(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32))
rba2.push(
    np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
)

rba.extend(rba2)
print(rba)

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
print(rb3)
rb4 = RaggedBufferI64.from_array(np.array([0, 3, 10], dtype=np.int64).reshape(3, 1, 1))
print(rb4)
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
print(rb6)
print(rb6[np.array([1, 3, 0], dtype=np.int64)])
